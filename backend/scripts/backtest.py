"""Would the alert rules have caught real Nairobi floods, using only forecasts available at the time?

Replays Open-Meteo Previous Runs forecasts over every zone and scores them against evidence/flood_events.csv.
River discharge is left out because past forecasts of it are not available, so this tests the rain rules only.

    python -m scripts.backtest          # from backend/; responses are cached in evidence/cache

Sources compared, all as rolling 24 h totals, alerting if ANY zone crosses the threshold:
    day_ahead   forecast made about 24 h before each hour: the skill a day's warning would really have
    latest_run  most recent run before each hour: the skill of short-notice hourly monitoring
    era5        reanalysis, known only afterwards: the upper limit if forecasts were perfect at this scale
"""
from __future__ import annotations

import csv
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np

from app.risk import RAIN_BAND_MM, SATURATED_MIN_NEXT_24H_MM, SATURATED_PREV_72H_MM

BACKEND = Path(__file__).resolve().parents[1]
EVIDENCE = BACKEND / "evidence"
CACHE = EVIDENCE / "cache"
PREVIOUS_RUNS = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
START, END = date(2024, 2, 1), date(2026, 11, 5)
CHUNK_DAYS, BATCH = 180, 10
ERA5_EVERY_NTH_ZONE = 5             # ERA5 cells are about 25 km: all zones would repeat a handful of cells
EVENT_LEAD_DAYS = 1                 # an alert the day before an event counts as a hit
SWEEP_MM = (5, 10, 15, 20, 30, 40, 50)
NEIGHBOUR_RADII_KM = (0, 5, 10, 15, 100)   # 100 km = the whole city


def fetch(client: httpx.Client, url: str, params: dict, name: str) -> list[dict]:
    path = CACHE / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(6):
        response = client.get(url, params=params, timeout=180)
        if response.status_code == 200:
            payload = response.json()
            payload = payload if isinstance(payload, list) else [payload]
            path.write_text(json.dumps(payload))
            time.sleep(15)          # stays well inside the free API's per-minute limit
            return payload
        print(f"  {name}: HTTP {response.status_code}, waiting")
        time.sleep(70)
    raise RuntimeError(f"{name} failed: {response.text[:200]}")


def chunks() -> list[tuple[date, date]]:
    out, cursor = [], START
    while cursor <= END:
        stop = min(cursor + timedelta(days=CHUNK_DAYS - 1), END)
        out.append((cursor, stop))
        cursor = stop + timedelta(days=1)
    return out


def load_series(client: httpx.Client, url: str, zones: list[dict], variable: str, tag: str) -> tuple[list[str], np.ndarray]:
    """Return hourly local times and a [zones, hours] array of rain, with NaN where missing."""
    all_times: list[str] = []
    rows: list[list[float]] = []
    for c, (a, b) in enumerate(chunks()):
        chunk_rows: list[list[float]] = []
        times: list[str] = []
        for i in range(0, len(zones), BATCH):
            batch = zones[i:i + BATCH]
            print(f"{tag} chunk {c + 1}/{len(chunks())} zones {i + 1}-{i + len(batch)}")
            hourly_vars = ("precipitation,precipitation_previous_day1" if url == PREVIOUS_RUNS else "precipitation")
            payload = fetch(client, url, {
                "latitude": ",".join(str(z["lat"]) for z in batch),
                "longitude": ",".join(str(z["lon"]) for z in batch),
                "hourly": hourly_vars, "start_date": a.isoformat(), "end_date": b.isoformat(),
                "timezone": "Africa/Nairobi"}, f"{tag.split('_')[0] if url == PREVIOUS_RUNS else tag}_{c}_{i}")
            for loc in payload:
                times = loc["hourly"]["time"]
                chunk_rows.append([np.nan if v is None else float(v) for v in loc["hourly"][variable]])
        all_times += times
        rows = chunk_rows if not rows else [r + cr for r, cr in zip(rows, chunk_rows)]
    return all_times, np.array(rows, dtype=float)


def rolling_sum(rain: np.ndarray, hours: int) -> np.ndarray:
    """Sum of the `hours` values from each index on. NaN where over 10% of the window is missing."""
    ok = ~np.isnan(rain)
    filled = np.nan_to_num(rain)
    n_zones, n_hours = rain.shape
    pad = np.zeros((n_zones, 1))
    total = np.hstack([pad, np.cumsum(filled, axis=1)])
    count = np.hstack([pad, np.cumsum(ok, axis=1)])
    out = np.full(rain.shape, np.nan)
    out[:, :n_hours - hours + 1] = total[:, hours:] - total[:, :-hours]
    present = np.full(rain.shape, 0.0)
    present[:, :n_hours - hours + 1] = count[:, hours:] - count[:, :-hours]
    out[present < 0.9 * hours] = np.nan
    return out


def alert_days(times: list[str], next_24: np.ndarray, threshold: float, prev_72: np.ndarray | None = None) -> set[date]:
    """Local days on which any zone reaches `threshold`. With prev_72, saturated ground lowers it one band."""
    fired = next_24 >= threshold
    if prev_72 is not None:
        lower = max([t for t in (0.0,) + RAIN_BAND_MM if t < threshold])
        saturated = (prev_72 >= SATURATED_PREV_72H_MM) & (next_24 >= max(lower, SATURATED_MIN_NEXT_24H_MM))
        fired = fired | saturated
    return {datetime.fromisoformat(times[h]).date() for h in np.flatnonzero(fired.any(axis=0))}


def episodes(days: set[date]) -> list[tuple[date, date]]:
    """Group consecutive days. People experience a run of alert days as one alert."""
    out: list[tuple[date, date]] = []
    for d in sorted(days):
        if out and d - out[-1][1] <= timedelta(days=1):
            out[-1] = (out[-1][0], d)
        else:
            out.append((d, d))
    return out


def score(days: set[date], events: list[dict], total_days: int) -> dict:
    hits = {}
    window_days = []
    for e in events:
        window = [e["start"] - timedelta(days=EVENT_LEAD_DAYS) + timedelta(days=i)
                  for i in range((e["end"] - e["start"]).days + EVENT_LEAD_DAYS + 1)]
        window_days.append(len(window))
        hits[e["event_id"]] = any(d in days for d in window)
    runs = episodes(days)
    matched = sum(any(a - timedelta(days=1) <= e["end"] and b + timedelta(days=1) >= e["start"] for e in events)
                  for a, b in runs)
    rate = len(days) / total_days
    chance = float(np.mean([1 - (1 - rate) ** w for w in window_days]))
    return {"events_detected": sum(hits.values()), "events_total": len(events), "detection_rate": round(np.mean(list(hits.values())), 3),
            "expected_by_chance": round(chance, 3), "alert_days": len(days), "alert_days_per_year": round(rate * 365.25, 1),
            "alert_episodes": len(runs), "episodes_matching_a_known_event": matched,
            "episode_precision_lower_bound": round(matched / len(runs), 3) if runs else None, "hits": hits}


def per_subscriber(times: list[str], next_24: np.ndarray, zones: list[dict], events: list[dict], total_days: int,
                   threshold: float) -> dict:
    """Detection as one subscriber sees it: their zone only, or any zone within a radius of it."""
    lat = np.radians([z["lat"] for z in zones])
    lon = np.radians([z["lon"] for z in zones])
    km = 6371 * np.hypot(lat[:, None] - lat[None, :], (lon[:, None] - lon[None, :]) * np.cos(lat.mean()))
    fired = next_24 >= threshold
    out = {}
    for radius in NEIGHBOUR_RADII_KM:
        per_zone = []
        for z in range(len(zones)):
            near = fired[km[z] <= radius].any(axis=0)
            days = {datetime.fromisoformat(times[h]).date() for h in np.flatnonzero(near)}
            per_zone.append(score(days, events, total_days))
        out[f"{radius}km"] = {
            "mean_events_detected": round(float(np.mean([r["events_detected"] for r in per_zone])), 2),
            "worst_zone_events_detected": min(r["events_detected"] for r in per_zone),
            "mean_alert_days_per_year": round(float(np.mean([r["alert_days_per_year"] for r in per_zone])), 1),
            "mean_episodes_per_year": round(float(np.mean([r["alert_episodes"] for r in per_zone])) * 365.25 / total_days, 1)}
    return out


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    zones = json.loads((BACKEND / "assets" / "zones.json").read_text())["zones"]
    with open(EVIDENCE / "flood_events.csv", encoding="utf-8", newline="") as fh:
        events = [{**r, "start": date.fromisoformat(r["date_start"]), "end": date.fromisoformat(r["date_end"])}
                  for r in csv.DictReader(fh)]
    total_days = (END - START).days + 1
    sources = {}
    with httpx.Client(headers={"User-Agent": "FloodGuard-backtest/0.1"}) as client:
        sources["day_ahead"] = load_series(client, PREVIOUS_RUNS, zones, "precipitation_previous_day1", "prev_day_ahead")
        sources["latest_run"] = load_series(client, PREVIOUS_RUNS, zones, "precipitation", "prev_latest")
        sources["era5"] = load_series(client, ARCHIVE, zones[::ERA5_EVERY_NTH_ZONE], "precipitation", "era5")

    results: dict = {"period": [START.isoformat(), END.isoformat()], "days": total_days, "events": len(events),
                     "sources": {}}
    per_event = {e["event_id"]: {"date": e["date_start"]} for e in events}
    for name, (times, rain) in sources.items():
        next_24 = rolling_sum(rain, 24)
        prev_72 = np.hstack([np.full((rain.shape[0], 72), np.nan), rolling_sum(rain, 72)[:, :-72]])
        sweep = {f"{mm}mm": score(alert_days(times, next_24, mm), events, total_days) for mm in SWEEP_MM}
        rules = {f"band{b + 1}_{RAIN_BAND_MM[b]:.0f}mm_with_saturation":
                 score(alert_days(times, next_24, RAIN_BAND_MM[b], prev_72), events, total_days) for b in range(2)}
        results["sources"][name] = {"coverage": round(float(np.mean(~np.isnan(rain))), 4), "sweep": sweep, "rules": rules}
        if name != "era5":
            results["sources"][name]["per_subscriber_20mm"] = per_subscriber(times, next_24, zones, events, total_days, 20.0)
        for e in events:
            mask = [e["start"] - timedelta(days=EVENT_LEAD_DAYS) <= datetime.fromisoformat(t).date() <= e["end"]
                    for t in times]
            peak = np.nanmax(next_24[:, np.array(mask)])
            per_event[e["event_id"]][f"{name}_peak_24h_mm"] = round(float(peak), 1)
    results["per_event_peak_any_zone"] = per_event
    (EVIDENCE / "backtest_results.json").write_text(json.dumps(results, indent=1, default=str))
    print(json.dumps({k: results["sources"][k]["sweep"] for k in results["sources"]}, default=str)[:300])
    print(json.dumps(per_event, indent=1))


if __name__ == "__main__":
    main()
