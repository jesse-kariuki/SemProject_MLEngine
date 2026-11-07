# FloodGuard backend

Live flood risk and SMS alerts for Nairobi. Every hour it reads rainfall forecasts, river discharge and official
Kenya Met warnings, then alerts subscribers near the rain. Evidence and limitations are in
[evidence/VALIDATION.md](evidence/VALIDATION.md).

## Run it

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
copy .env.example .env
.venv\Scripts\python -m uvicorn app.main:app --port 8000
```

Open http://localhost:8000/docs for the interactive API. With the default `SMS_PROVIDER=console`, messages go to the
log instead of phones, including verification codes. A first cycle runs at startup.

```powershell
.venv\Scripts\python -m pytest -q
```

## Deploy

The `Dockerfile` is at the repo root, so a git-based host can use its default build path and the repo root as the build
context. The root `.dockerignore` keeps the models and raw data out of the build. To build it locally, from the repo root:

```powershell
docker build -t floodguard-api .
docker run -p 8000:8000 --env-file backend/.env -v floodguard-data:/srv/data floodguard-api
```

On the host:

- Set `ADMIN_API_KEY`. Use a Postgres `DATABASE_URL`, or mount a persistent volume at `/srv/data`. Without one of these,
  subscribers are lost on every redeploy.
- Run exactly one always-on instance. The scheduler runs inside the process, so extra instances send duplicate SMS and
  scale-to-zero stops monitoring.
- The app listens on `$PORT`, defaulting to 8000.
- Use `/health/live` for the platform health check. Use `/health` for uptime monitoring: it reports `degraded` when data
  goes stale, which should alert you but should not restart the container.

## How it works

Each cycle is independent per source, so one dead feed does not stop the others:

1. Fetch 72 h of past and 24 h of forecast rain for 31 zones, plus GloFAS river discharge.
2. Give each zone a rain band from the wettest forecast within 15 km.
3. Raise the band one step if the ground is saturated or the river is above its 99th percentile.
4. Combine the band with local road susceptibility to get each subscriber's alert level.
5. Send SMS for new or rising alerts, and forward Kenya Met alerts covering the subscriber.

| Rain band (mm/24 h) | Low | Medium | High |
|---|---|---|---|
| Under 20 | None | None | None |
| 20 to 49 | Watch | Watch | Warning |
| 50 to 79 | Warning | Warning | Severe |
| 80 and over | Warning | Severe | Severe |

Susceptibility can raise urgency but never silences an alert, because most real flood sites tested score Low.

Safety rules, each covered by tests:

- Missing or stale data is reported as `unknown`, never as no risk, and never produces an all-clear.
- Unchanged alerts repeat at most every 24 h; rising alerts go out at once.
- An all-clear needs at least 6 h since the last alert, so one dry forecast run cannot trigger it.
- A failed SMS is retried next cycle. The per-cycle cap sends the most severe alerts first.

## Data sources

| Source | Used for | Access |
|---|---|---|
| [Open-Meteo Forecast](https://open-meteo.com/en/docs) | Hourly rain, the main alert signal | Live, free for non-commercial use |
| [Open-Meteo Flood](https://open-meteo.com/en/docs/flood-api) | GloFAS river discharge | Live |
| [Kenya Met CAP feed](https://meteo.go.ke/api/cap/rss.xml) | Official warnings, forwarded by SMS | Live and public, listed in the [WMO register](https://alertingauthority.wmo.int/authorities.php?recId=80) |
| [GDACS](https://www.gdacs.org/) | Kenya flood events, shown for context only | Live and public |
| `models/H_rf` | Road susceptibility, precomputed into `assets/` | Rerun `scripts/export_assets.py` after retraining |

No public live river gauge or rain gauge API for Kenya was found. The Water Resources Authority, Kenya Met station data
and TAHMO are worth approaching for access.

## API

| Method and path | Purpose |
|---|---|
| `GET /health`, `GET /health/live` | Data freshness; process liveness |
| `GET /api/v1/status` | Sources, last cycle, model version |
| `GET /api/v1/zones` | Current assessment for every zone |
| `GET /api/v1/risk?latitude=&longitude=` | Risk at a location |
| `GET /api/v1/roads/search?q=` | Find a road to subscribe to |
| `GET /api/v1/roads/at-risk?min_level=` | Named roads at or above a level |
| `GET /api/v1/alerts/official` | Active Kenya Met alerts and GDACS events |
| `POST /api/v1/subscribers` | Start a subscription; sends a code |
| `POST /api/v1/subscribers/verify` | Confirm the code; returns a token |
| `GET`, `PATCH`, `DELETE /api/v1/subscribers/me` | Manage or delete a subscription (Bearer token) |
| `GET /api/v1/subscribers/me/alerts` | Alert history |
| `POST /webhooks/sms/inbound?secret=` | Handle STOP replies |
| `POST /admin/pipeline/run` | Run a cycle now (`X-Admin-Key`) |
| `POST /admin/drill` | Preview a rain scenario; can text one subscriber a labelled drill |
| `GET /admin/stats` | Subscribers, deliveries, cycles |

For a dry demo day, use `/admin/drill`. It changes no live state and marks any SMS as a drill.

## Before launch

1. **ThriveWave:** replace the placeholder request in `ThriveWaveProvider.send` (`app/sms.py`) with the real API. Its
   docs are not public. The class is the only code to change.
2. **STOP replies:** connect the provider's inbound messages to the webhook, or change `SMS_OPT_OUT_HINT` so messages
   do not promise an opt-out that is not wired up.
3. **Sender ID:** register one with the provider. Kenyan alphanumeric sender IDs need registration.
4. **Kiswahili:** have a native speaker review `app/messages.py`.
5. **Data protection:** phone numbers and locations are personal data under Kenya's Data Protection Act 2019. Add a
   privacy notice and confirm any registration duties.
6. **Rate limits:** the sign-up limits are per number and per hour globally, with no per-IP limit.

Not an official warning service. See the disclaimer returned by the API.
