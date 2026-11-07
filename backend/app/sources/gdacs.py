"""GDACS flood events in Kenya. Coarse and infrequent, so shown for context and never used to trigger SMS."""
from __future__ import annotations

from datetime import date, timedelta

import httpx

from app.config import get_settings
from app.sources.http import get_with_retries

LOOKBACK_DAYS = 30


def fetch_events(client: httpx.Client, today: date) -> list[dict]:
    response = get_with_retries(client, get_settings().gdacs_url, {
        "eventlist": "FL", "country": "Kenya", "fromDate": (today - timedelta(days=LOOKBACK_DAYS)).isoformat(),
        "toDate": today.isoformat()})
    if not response.content:        # GDACS answers with an empty body when there are no events
        return []
    events = []
    for feature in response.json().get("features", []):
        p = feature.get("properties", {})
        events.append({"event_id": p.get("eventid"), "alert_level": p.get("alertlevel"), "name": p.get("name"),
                       "from": p.get("fromdate"), "to": p.get("todate"), "is_current": p.get("iscurrent"),
                       "url": (p.get("url") or {}).get("report")})
    return events
