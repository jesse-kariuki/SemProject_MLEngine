from datetime import datetime, timedelta, timezone

import pytest

from app import messages
from app.alerts import ALERT, ALL_CLEAR, decide
from app.phone import normalise_phone
from app.risk import MATRIX, NONE, SEVERE, WARNING, WATCH, assess_zone, local_level, rain_band, spread_to_neighbours
from app.sources.kmd_cap import covers, parse_cap, point_in_ring
from tests.conftest import cap_xml

ISSUE = datetime(2026, 4, 28, 9, tzinfo=timezone.utc)
TIMES = [ISSUE - timedelta(hours=72) + timedelta(hours=h) for h in range(96)]


def series(past=0.0, future=0.0):
    return [past] * 72 + [future] * 24


@pytest.mark.parametrize("mm,band", [(0, 0), (19.9, 0), (20, 1), (49.9, 1), (50, 2), (79.9, 2), (80, 3), (300, 3)])
def test_rain_band_boundaries(mm, band):
    assert rain_band(mm) == band


def test_no_rain_means_no_alert_on_any_road():
    assert all(local_level(0, tier) == NONE for tier in MATRIX)


def test_levels_never_fall_as_rain_or_susceptibility_rises():
    tiers = ["Low", "Medium", "High"]
    for band in range(4):
        assert [local_level(band, t) for t in tiers] == sorted(local_level(band, t) for t in tiers)
    for tier in tiers:
        assert list(MATRIX[tier]) == sorted(MATRIX[tier])


def test_low_susceptibility_never_silences_heavy_rain():
    assert local_level(1, "Low") >= WATCH
    assert local_level(2, "Low") >= WARNING
    assert local_level(3, "Low") >= WARNING


def test_unscored_segments_are_treated_as_medium():
    assert local_level(2, "Unavailable") == local_level(2, "Medium")


def test_assess_sums_only_the_next_24_hours():
    z = assess_zone("Z", TIMES, series(past=5, future=1), ISSUE)
    assert z.rain_next_24h_mm == 24 and z.rain_next_6h_mm == 6 and z.rain_prev_72h_mm == 360


def test_saturated_ground_raises_band_one_step():
    dry = assess_zone("Z", TIMES, series(past=0, future=1), ISSUE)
    wet = assess_zone("Z", TIMES, series(past=1, future=1), ISSUE)
    assert (dry.rain_band, wet.rain_band) == (1, 2) and "ground_saturated" in wet.reasons


def test_saturation_alone_does_not_alert_without_forecast_rain():
    assert assess_zone("Z", TIMES, series(past=2, future=0.2), ISSUE).rain_band == 0


def test_river_above_p99_alerts_even_without_local_rain():
    z = assess_zone("Z", TIMES, series(), ISSUE, river_q=30, river_p99=20)
    assert z.rain_band == 1 and z.reasons == ["river_above_p99"]


def test_wet_ground_and_high_river_only_add_one_step():
    z = assess_zone("Z", TIMES, series(past=1, future=1), ISSUE, river_q=30, river_p99=20)
    assert z.rain_band == 2


def test_band_is_capped_at_the_top():
    assert assess_zone("Z", TIMES, series(past=1, future=5), ISSUE).rain_band == 3


def test_incomplete_forecast_is_unavailable_not_zero():
    rain = series(future=1)
    rain[72:80] = [None] * 8
    z = assess_zone("Z", TIMES, rain, ISSUE)
    assert not z.available and z.rain_next_24h_mm is None


def test_a_few_missing_hours_are_tolerated():
    rain = series(future=1)
    rain[80] = None
    assert assess_zone("Z", TIMES, rain, ISSUE).rain_next_24h_mm == 23


def test_rain_in_a_neighbouring_zone_raises_the_band():
    risks = {"A": assess_zone("A", TIMES, series(future=2.5), ISSUE), "B": assess_zone("B", TIMES, series(), ISSUE)}
    spread_to_neighbours(risks, {"A": ["A", "B"], "B": ["A", "B"]})
    assert risks["B"].rain_band == 2 and risks["B"].rain_area_24h_mm == 60 and "rain_nearby" in risks["B"].reasons
    assert risks["B"].rain_next_24h_mm == 0


def test_rain_in_a_distant_zone_does_not_spread():
    risks = {"A": assess_zone("A", TIMES, series(future=2.5), ISSUE), "B": assess_zone("B", TIMES, series(), ISSUE)}
    spread_to_neighbours(risks, {"A": ["A"], "B": ["B"]})
    assert risks["B"].rain_band == 0


def test_a_zone_with_no_forecast_stays_unavailable_beside_wet_neighbours():
    gap = [None] * 96
    risks = {"A": assess_zone("A", TIMES, series(future=2.5), ISSUE), "B": assess_zone("B", TIMES, gap, ISSUE)}
    spread_to_neighbours(risks, {"A": ["A", "B"], "B": ["A", "B"]})
    assert not risks["B"].available and risks["A"].rain_band == 2


NOW = ISSUE


@pytest.mark.parametrize("level,last,hours_ago,expected", [
    (2, 0, None, ALERT),            # first alert
    (2, 2, 1, None),                # unchanged: stay quiet
    (2, 2, 13, ALERT),              # unchanged, cooldown passed: remind
    (3, 2, 1, ALERT),               # escalation goes straight out
    (2, 3, 1, None),                # easing but still active: no noise
    (0, 2, 1, None),                # one low run: too soon for an all-clear
    (0, 2, 7, ALL_CLEAR),           # stayed low
    (0, 0, None, None),             # nothing was ever sent
    (1, 0, None, None),             # below this subscriber's threshold
])
def test_decide(level, last, hours_ago, expected):
    at = NOW - timedelta(hours=hours_ago) if hours_ago is not None else None
    assert decide(level, 2, last, at, NOW, 12) == expected


@pytest.mark.parametrize("language", ["en", "sw"])
@pytest.mark.parametrize("level", [WATCH, WARNING, SEVERE])
def test_every_alert_fits_one_sms(language, level):
    text = messages.flood_message(level, "Very Long Neighbourhood Name Estate Phase Two", 123, language,
                                  "Reply STOP to opt out")
    assert len(text) <= 160


def test_full_opt_out_text_survives_in_the_longest_real_message():
    text = messages.flood_message(SEVERE, "Very Long Neighbourhood Name", 123, "sw", "Reply STOP to opt out")
    assert text.endswith("Reply STOP to opt out")


def test_message_does_not_quote_rain_it_did_not_forecast():
    assert "mm" not in messages.flood_message(WATCH, "South B", 0.4, "en")


def test_official_headline_is_trimmed_to_one_sms():
    assert len(messages.official_message("x " * 200, "en")) <= 160


@pytest.mark.parametrize("raw,expected", [
    ("0712345678", "+254712345678"), ("712345678", "+254712345678"), ("254712345678", "+254712345678"),
    ("+254 712 345 678", "+254712345678"), ("0112345678", "+254112345678"), ("0812345678", None),
    ("12345", None), ("+14155550100", None), ("", None), ("07123456789", None),
])
def test_phone_normalisation(raw, expected):
    assert normalise_phone(raw) == expected


def test_point_in_ring():
    square = [(0.0, 0.0), (0.0, 2.0), (2.0, 2.0), (2.0, 0.0)]
    assert point_in_ring(1, 1, square) and not point_in_ring(3, 1, square)


def test_cap_parsing_reads_lat_lon_order():
    alert = parse_cap(cap_xml(), "u")
    assert covers(alert.polygons, -1.29, 36.82) and not covers(alert.polygons, 36.82, -1.29)


@pytest.mark.parametrize("kwargs", [{"status": "Test"}, {"status": "Exercise"}, {"msg_type": "Cancel"},
                                    {"scope": "Restricted"}, {"event": "Large waves", "headline": "High seas"}])
def test_cap_ignores_tests_cancellations_and_unrelated_hazards(kwargs):
    assert parse_cap(cap_xml(**kwargs), "u") is None
