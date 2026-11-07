"""Alert rules: how rainfall and road susceptibility combine into an alert level.

Pure functions with no I/O, so every rule can be tested and explained. This is a transparent rule
matrix, not a calibrated probability: no time-stamped flood labels exist yet to train one.
Sources and assumptions are listed in backend/README.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

# Rain bands on forecast mm/24 h. 20 and 50 are the figures Kenya Met uses in its advisories.
# 80 is the top of the 50-80 mm range in its May 2024 Nairobi advisory.
RAIN_BAND_MM = (20.0, 50.0, 80.0)
BAND_NAMES = ("below_advisory", "moderate", "heavy", "very_heavy")

# Assumptions, not sourced thresholds: wet ground makes the same rain more dangerous.
SATURATED_PREV_72H_MM = 50.0
SATURATED_MIN_NEXT_24H_MM = 10.0

# Forecasts place storms kilometres off, so a zone takes the wettest forecast within this radius. Per subscriber
# this catches about 5.6 of 7 past floods, against 3 of 7 for the zone alone (evidence/VALIDATION.md).
NEIGHBOUR_RADIUS_KM = 15.0

MIN_HOUR_COVERAGE = 0.9             # same rule as forecast_features.min_hour_coverage in settings.yaml

NONE, WATCH, WARNING, SEVERE = 0, 1, 2, 3
LEVEL_NAMES = {NONE: "none", WATCH: "watch", WARNING: "warning", SEVERE: "severe"}

# level = MATRIX[tier][rain band]. Susceptibility can raise urgency but never silences an alert: 5 of 8 places
# that really flooded score Low (evidence/site_check.json), because the training labels ignore drainage.
MATRIX = {
    "Low":    (NONE, WATCH, WARNING, WARNING),
    "Medium": (NONE, WATCH, WARNING, SEVERE),
    "High":   (NONE, WARNING, SEVERE, SEVERE),
}
UNMODELLED_TIER = "Medium"          # segments the model could not score: neither ignore nor over-alarm


@dataclass
class ZoneRisk:
    zone_id: str
    available: bool
    rain_next_6h_mm: float | None = None
    rain_next_24h_mm: float | None = None
    rain_area_24h_mm: float | None = None       # wettest forecast within NEIGHBOUR_RADIUS_KM: this drives the band
    rain_prev_72h_mm: float | None = None
    max_hourly_mm: float | None = None
    precip_probability_max: float | None = None
    river_discharge_m3s: float | None = None
    river_p99_m3s: float | None = None
    rain_band: int = 0
    reasons: list[str] = field(default_factory=list)


def _window(by_time: dict[datetime, float | None], start: datetime, hours: int) -> tuple[float | None, float | None]:
    """Sum and peak over `hours` from `start`, or (None, None) if too many hours are missing."""
    values = [by_time.get(start + timedelta(hours=h)) for h in range(hours)]
    present = [v for v in values if v is not None]
    if len(present) < MIN_HOUR_COVERAGE * hours:
        return None, None
    return round(sum(present), 1), round(max(present), 1)


def rain_band(rain_24h_mm: float) -> int:
    return sum(rain_24h_mm >= t for t in RAIN_BAND_MM)


def apply_modifiers(band: int, rain_24h_mm: float, rain_prev_72h_mm: float | None, river_q: float | None,
                    river_p99: float | None) -> tuple[int, list[str]]:
    """Raise the band one step for saturated ground or a very high river. The signals overlap, so never two."""
    reasons = [f"rain_{BAND_NAMES[band]}"] if band else []
    saturated = (rain_prev_72h_mm or 0) >= SATURATED_PREV_72H_MM and rain_24h_mm >= SATURATED_MIN_NEXT_24H_MM
    river_high = river_q is not None and river_p99 is not None and river_q >= river_p99
    if saturated:
        reasons.append("ground_saturated")
    if river_high:
        reasons.append("river_above_p99")
    if saturated or river_high:
        band = min(band + 1, len(RAIN_BAND_MM))
    return band, reasons


def assess_zone(zone_id: str, times: list[datetime], precipitation: list[float | None], issue: datetime,
                probability: list[float | None] | None = None, river_q: float | None = None,
                river_p99: float | None = None) -> ZoneRisk:
    """Assess one zone from its hourly UTC series. `issue` is the forecast start, floored to the hour."""
    series = dict(zip(times, precipitation))
    rain_24, peak = _window(series, issue, 24)
    if rain_24 is None:
        return ZoneRisk(zone_id, available=False, reasons=["forecast_incomplete"])
    rain_6, _ = _window(series, issue, 6)
    prev_72, _ = _window(series, issue - timedelta(hours=72), 72)
    prob = None
    if probability:
        in_window = [p for t, p in zip(times, probability)
                     if issue <= t < issue + timedelta(hours=24) and p is not None]
        prob = max(in_window) if in_window else None
    band, reasons = apply_modifiers(rain_band(rain_24), rain_24, prev_72, river_q, river_p99)
    return ZoneRisk(zone_id, True, rain_6, rain_24, rain_24, prev_72, peak, prob, river_q, river_p99, band, reasons)


def spread_to_neighbours(risks: dict[str, ZoneRisk], neighbours: dict[str, list[str]]) -> None:
    """Re-band each usable zone on the wettest forecast near it. Its own wet-ground and river state still apply."""
    own_rain = {z: r.rain_next_24h_mm for z, r in risks.items() if r.available}
    for zone_id, risk in risks.items():
        if not risk.available:
            continue
        area = max(own_rain[n] for n in neighbours.get(zone_id, [zone_id]) if n in own_rain)
        risk.rain_area_24h_mm = area
        risk.rain_band, risk.reasons = apply_modifiers(rain_band(area), area, risk.rain_prev_72h_mm,
                                                       risk.river_discharge_m3s, risk.river_p99_m3s)
        if area > (risk.rain_next_24h_mm or 0) and risk.rain_band:
            risk.reasons.append("rain_nearby")


def local_level(band: int, tier: str) -> int:
    return MATRIX.get(tier, MATRIX[UNMODELLED_TIER])[band]
