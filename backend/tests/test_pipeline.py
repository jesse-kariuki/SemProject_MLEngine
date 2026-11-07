"""End-to-end cycles against fake external APIs: the failure paths matter as much as the happy path."""
from datetime import timedelta

from sqlalchemy import select

from app.models import Delivery, OfficialAlert, PipelineRun, Subscriber
from app.pipeline import latest_assessments, latest_run, run_cycle
from tests.conftest import NOW, cap_xml, point_with_tier, subscribe


def cycle(db, world, sms, now=NOW):
    world.now = now
    with world.client() as client:
        return run_cycle(db, client, sms, now=now)


def high_subscriber(api, otp_sink, phone="0712345678", **extra):
    return subscribe(api, otp_sink, phone, point_with_tier("High"), **extra)


def test_dry_day_stores_all_zones_and_sends_nothing(api, db, world, sms, otp_sink):
    high_subscriber(api, otp_sink)
    run = cycle(db, world, sms)
    assert (run.status, run.zones_available, run.zones_total) == ("ok", 31, 31)
    assert sms.sent == []


def test_heavy_rain_alerts_a_high_location(api, db, world, sms, otp_sink):
    high_subscriber(api, otp_sink, place_label="South B")
    world.rain_per_hour = 2.5       # 60 mm/24 h: heavy band
    run = cycle(db, world, sms)
    assert run.sms_sent == 1
    phone, text = sms.sent[0]
    assert phone == "+254712345678" and "SEVERE WARNING" in text and "South B" in text and "60mm" in text


def test_moderate_rain_alerts_by_default_even_at_a_low_location(api, db, world, sms, otp_sink):
    subscribe(api, otp_sink, "0712345678", point_with_tier("Low"))
    world.rain_per_hour = 1.0       # 24 mm: the 20 mm band caught most past floods, so the default includes it
    cycle(db, world, sms)
    assert len(sms.sent) == 1 and "WATCH" in sms.sent[0][1]


def test_subscriber_can_choose_warnings_only(api, db, world, sms, otp_sink):
    subscribe(api, otp_sink, "0712345678", point_with_tier("Low"), min_level=2)
    world.rain_per_hour = 1.0
    cycle(db, world, sms)
    assert sms.sent == []


def test_heavy_rain_still_alerts_a_low_location(api, db, world, sms, otp_sink):
    subscribe(api, otp_sink, "0712345678", point_with_tier("Low"))
    world.rain_per_hour = 2.5
    cycle(db, world, sms)
    assert len(sms.sent) == 1 and "WARNING" in sms.sent[0][1]


def test_kiswahili_subscriber_gets_kiswahili(api, db, world, sms, otp_sink):
    high_subscriber(api, otp_sink, language="sw")
    world.rain_per_hour = 2.5
    cycle(db, world, sms)
    assert "HATARI KUBWA" in sms.sent[0][1]


def test_hourly_cycles_do_not_repeat_the_alert(api, db, world, sms, otp_sink):
    high_subscriber(api, otp_sink)
    world.rain_per_hour = 2.5
    for hour in range(6):
        cycle(db, world, sms, NOW + timedelta(hours=hour))
    assert len(sms.sent) == 1


def test_escalation_is_sent_immediately(api, db, world, sms, otp_sink):
    high_subscriber(api, otp_sink)
    world.rain_per_hour = 1.0       # 24 mm: warning
    cycle(db, world, sms)
    world.rain_per_hour = 2.5       # 60 mm: severe
    cycle(db, world, sms, NOW + timedelta(hours=1))
    assert ["SEVERE" in text for _, text in sms.sent] == [False, True]


def test_all_clear_after_the_risk_stays_low(api, db, world, sms, otp_sink):
    high_subscriber(api, otp_sink)
    world.rain_per_hour = 2.5
    cycle(db, world, sms)
    world.rain_per_hour = 0
    cycle(db, world, sms, NOW + timedelta(hours=1))
    assert len(sms.sent) == 1       # too soon
    cycle(db, world, sms, NOW + timedelta(hours=7))
    assert len(sms.sent) == 2 and "eased" in sms.sent[1][1]
    assert db.scalar(select(Subscriber)).last_alert_level == 0


def test_forecast_outage_sends_nothing_and_never_an_all_clear(api, db, world, sms, otp_sink):
    high_subscriber(api, otp_sink)
    world.rain_per_hour = 2.5
    cycle(db, world, sms)
    world.forecast_status = 503
    run = cycle(db, world, sms, NOW + timedelta(hours=8))
    assert run.status == "failed" and "open_meteo_forecast" in run.source_errors
    assert len(sms.sent) == 1
    assert db.scalar(select(Subscriber)).last_alert_level > 0   # still on record as alerted


def test_failed_run_keeps_the_last_good_assessment(db, world, sms):
    good = cycle(db, world, sms)
    world.forecast_status = 500
    cycle(db, world, sms, NOW + timedelta(hours=1))
    assert latest_run(db).id == good.id and len(latest_assessments(db, good)) == 31


def test_river_feed_outage_degrades_but_rain_alerts_continue(api, db, world, sms, otp_sink):
    high_subscriber(api, otp_sink)
    world.flood_status = 500
    world.rain_per_hour = 2.5
    run = cycle(db, world, sms)
    assert run.status == "ok" and "open_meteo_flood" in run.source_errors and len(sms.sent) == 1


def test_kmd_alert_goes_out_even_when_our_forecast_is_down(api, db, world, sms, otp_sink):
    high_subscriber(api, otp_sink)
    world.forecast_status = 503
    world.kmd_items = [cap_xml()]
    cycle(db, world, sms)
    assert len(sms.sent) == 1 and sms.sent[0][1].startswith("Kenya Met alert:")


def test_kmd_alert_is_sent_once(api, db, world, sms, otp_sink):
    high_subscriber(api, otp_sink)
    world.kmd_items = [cap_xml()]
    for hour in (0, 1, 26):
        cycle(db, world, sms, NOW + timedelta(hours=hour))
    assert len(sms.sent) == 1


def test_reissued_kmd_alerts_on_one_day_do_not_stack(api, db, world, sms, otp_sink):
    high_subscriber(api, otp_sink)
    world.kmd_items = [cap_xml("urn:a"), cap_xml("urn:b")]
    cycle(db, world, sms)
    assert len(sms.sent) == 1


def test_kmd_alert_outside_nairobi_is_stored_but_not_sent(api, db, world, sms, otp_sink):
    high_subscriber(api, otp_sink)
    world.kmd_items = [cap_xml(polygon="-4.5,39 -4.5,40 -3,40 -3,39 -4.5,39")]
    cycle(db, world, sms)
    assert sms.sent == []
    assert db.scalar(select(OfficialAlert)).zones_covered == []


def test_expired_kmd_alert_is_not_sent(api, db, world, sms, otp_sink):
    high_subscriber(api, otp_sink)
    world.kmd_items = [cap_xml(expires="2026-04-01T09:00:00+03:00")]
    cycle(db, world, sms)
    assert sms.sent == []


def test_sms_outage_is_recorded_and_retried(api, db, world, sms, otp_sink):
    high_subscriber(api, otp_sink)
    world.rain_per_hour = 2.5
    sms.fail = True
    run = cycle(db, world, sms)
    assert run.sms_failed == 1 and db.scalar(select(Delivery)).status == "failed"
    assert db.scalar(select(Subscriber)).last_alert_level == 0
    sms.fail = False
    cycle(db, world, sms, NOW + timedelta(hours=1))
    assert len(sms.sent) == 1


def test_sms_cap_sends_the_most_severe_first(api, db, world, sms, otp_sink, monkeypatch):
    subscribe(api, otp_sink, "0711111111", point_with_tier("Medium"))
    high_subscriber(api, otp_sink)
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "max_sms_per_cycle", 1)
    world.rain_per_hour = 2.5       # High = severe, Medium = warning
    run = cycle(db, world, sms)
    assert run.sms_held_back == 1 and sms.sent[0][0] == "+254712345678"


def test_unsubscribed_number_gets_nothing(api, db, world, sms, otp_sink):
    high_subscriber(api, otp_sink)
    assert api.post("/webhooks/sms/inbound?secret=test-secret",
                    json={"from": "+254712345678", "text": "stop"}).status_code == 200
    world.rain_per_hour = 2.5
    world.kmd_items = [cap_xml()]
    cycle(db, world, sms)
    assert sms.sent == []


def test_old_runs_are_pruned(db, world, sms):
    cycle(db, world, sms, NOW - timedelta(days=40))
    cycle(db, world, sms)
    assert len(db.scalars(select(PipelineRun)).all()) == 1
