"""Open-Meteo: hourly rainfall forecast and GloFAS river discharge."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone

import httpx

from app.config import get_settings
from app.geo import Zone
from app.sources.http import SourceError, get_with_retries

BATCH_SIZE = 10
RIVER_LOOKAHEAD_DAYS = 3
PAST_DAYS = 3                       # enough history for the 72 h antecedent window


@dataclass
class HourlySeries:
    times: list[datetime]
    precipitation: list[float | None]
    probability: list[float | None]


def _batches(zones: list[Zone]) -> list[list[Zone]]:
    return [zones[i:i + BATCH_SIZE] for i in range(0, len(zones), BATCH_SIZE)]


def _locations(zones: list[Zone]) -> dict:
    return {"latitude": ",".join(str(z.lat) for z in zones), "longitude": ",".join(str(z.lon) for z in zones)}


def _as_list(payload) -> list[dict]:
    return payload if isinstance(payload, list) else [payload]


def fetch_forecast(client: httpx.Client, zones: list[Zone]) -> dict[str, HourlySeries]:
    """Fetch hourly rain for every zone. Raises SourceError if the feed is unusable."""
    out: dict[str, HourlySeries] = {}
    for batch in _batches(zones):
        response = get_with_retries(client, get_settings().forecast_url, {
            **_locations(batch), "hourly": "precipitation,precipitation_probability",
            "past_days": PAST_DAYS, "forecast_days": 3, "timezone": "UTC"})
        locations = _as_list(response.json())
        if len(locations) != len(batch):
            raise SourceError(f"Open-Meteo returned {len(locations)} locations for {len(batch)} requested")
        for zone, loc in zip(batch, locations):
            hourly = loc.get("hourly") or {}
            if "time" not in hourly or "precipitation" not in hourly:
                raise SourceError(f"Open-Meteo response for {zone.zone_id} has no hourly precipitation")
            times = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in hourly["time"]]
            out[zone.zone_id] = HourlySeries(times, hourly["precipitation"],
                                             hourly.get("precipitation_probability") or [None] * len(times))
    return out


def fetch_river(client: httpx.Client, zones: list[Zone], today: date) -> dict[str, float]:
    """Peak forecast discharge (m3/s) from today over the next few days, by zone."""
    out: dict[str, float] = {}
    for batch in _batches(zones):
        response = get_with_retries(client, get_settings().flood_url, {
            **_locations(batch), "daily": "river_discharge", "forecast_days": RIVER_LOOKAHEAD_DAYS + 1})
        for zone, loc in zip(batch, _as_list(response.json())):
            daily = loc.get("daily") or {}
            pairs = zip(daily.get("time", []), daily.get("river_discharge", []))
            values = [q for day, q in pairs if q is not None and day >= today.isoformat()]
            if values:
                out[zone.zone_id] = max(values)
    return out
