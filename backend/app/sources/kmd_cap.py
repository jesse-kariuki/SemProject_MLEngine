"""Official Kenya Meteorological Department warnings, from its public CAP 1.2 feed."""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx

from app.config import get_settings
from app.sources.http import SourceError, get_with_retries

log = logging.getLogger("floodguard.kmd")
CAP = {"cap": "urn:oasis:names:tc:emergency:cap:1.2"}
RELEVANT_WORDS = ("rain", "flood", "storm")
MAX_ITEMS = 10


@dataclass
class CapAlert:
    identifier: str
    event: str
    headline: str
    description: str
    instruction: str
    severity: str
    area: str
    url: str
    sent: datetime | None
    expires: datetime | None
    polygons: list[list[tuple[float, float]]] = field(default_factory=list)  # (lat, lon) rings


def point_in_ring(lat: float, lon: float, ring: list[tuple[float, float]]) -> bool:
    """Ray casting. CAP polygons are 'lat,lon' pairs."""
    inside = False
    for (lat_a, lon_a), (lat_b, lon_b) in zip(ring, ring[1:] + ring[:1]):
        if (lat_a > lat) != (lat_b > lat) and lon < (lon_b - lon_a) * (lat - lat_a) / (lat_b - lat_a) + lon_a:
            inside = not inside
    return inside


def covers(polygons: list[list[tuple[float, float]]], lat: float, lon: float) -> bool:
    return any(point_in_ring(lat, lon, ring) for ring in polygons)


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def parse_rss(xml: str) -> list[str]:
    root = ET.fromstring(xml)
    return [(item.findtext("link") or "").strip() for item in root.iter("item")]


def _text(parent: ET.Element, tag: str) -> str:
    return (parent.findtext(f"cap:{tag}", default="", namespaces=CAP) or "").strip()


def parse_cap(xml: str, url: str) -> CapAlert | None:
    """Parse one CAP document. Returns None for anything that is not a real, public, relevant alert."""
    root = ET.fromstring(xml)
    if _text(root, "status") != "Actual" or _text(root, "scope") != "Public" or _text(root, "msgType") == "Cancel":
        return None
    info = root.find("cap:info", CAP)
    if info is None:
        return None
    event = _text(info, "event")
    if not any(word in f"{event} {_text(info, 'headline')}".lower() for word in RELEVANT_WORDS):
        return None
    areas, polygons = [], []
    for area in info.findall("cap:area", CAP):
        areas.append(_text(area, "areaDesc"))
        for poly in area.findall("cap:polygon", CAP):
            ring = [tuple(float(v) for v in pair.split(",")[:2]) for pair in (poly.text or "").split()]
            if len(ring) >= 3:
                polygons.append(ring)
    return CapAlert(_text(root, "identifier"), event, _text(info, "headline"), _text(info, "description"),
                    _text(info, "instruction"), _text(info, "severity"), "; ".join(a for a in areas if a),
                    _text(info, "web") or url, _parse_time(_text(root, "sent")), _parse_time(_text(info, "expires")),
                    polygons)


def fetch_alerts(client: httpx.Client) -> list[CapAlert]:
    """Current relevant alerts. One malformed document is skipped, not fatal."""
    try:
        links = parse_rss(get_with_retries(client, get_settings().kmd_cap_rss_url).text)
    except ET.ParseError as err:
        raise SourceError(f"KMD RSS is not valid XML: {err}") from err
    alerts = []
    for link in [link for link in links if link][:MAX_ITEMS]:
        try:
            alert = parse_cap(get_with_retries(client, link).text, link)
        except (SourceError, ET.ParseError, ValueError) as err:
            log.warning("Skipping KMD alert %s: %s", link, err)
            continue
        if alert:
            alerts.append(alert)
    return alerts
