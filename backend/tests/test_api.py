from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.models import Delivery, PipelineRun, Subscriber
from app.pipeline import run_cycle
from tests.conftest import cap_xml, point_with_tier, subscribe

KISUMU = (-0.09, 34.77)
ADMIN = {"X-Admin-Key": "test-admin"}


def live_cycle(db, world, sms, rain=0.0):
    world.rain_per_hour = rain
    now = datetime.now(timezone.utc)
    world.now = now
    with world.client() as client:
        return run_cycle(db, client, sms, now=now)


def body(point=None, **extra):
    point = point or point_with_tier("Low")
    return {"phone": "0712345678", "latitude": point[0], "longitude": point[1], "consent": True, **extra}


def code_from(sink):
    return sink[-1][1].split(": ")[1][:6]


def test_liveness_and_openapi(api):
    assert api.get("/health/live").json() == {"status": "up"}
    assert api.get("/docs").status_code == 200


def test_health_is_degraded_before_any_data(api):
    assert api.get("/health").json()["status"] == "degraded"


def test_risk_before_any_data_is_unknown_not_safe(api):
    lat, lon = point_with_tier("High")
    data = api.get("/api/v1/risk", params={"latitude": lat, "longitude": lon}).json()
    assert data["level"] is None and data["level_name"] == "unknown" and data["data"]["state"] == "no_data"


def test_risk_reflects_rain_and_location(api, db, world, sms):
    live_cycle(db, world, sms, rain=2.5)
    lat, lon = point_with_tier("High")
    data = api.get("/api/v1/risk", params={"latitude": lat, "longitude": lon}).json()
    assert data["level_name"] == "severe" and data["data"]["state"] == "fresh" and data["disclaimer"]
    lat, lon = point_with_tier("Low")
    assert api.get("/api/v1/risk", params={"latitude": lat, "longitude": lon}).json()["level_name"] == "warning"


def test_stale_data_is_reported_as_unknown(api, db, world, sms):
    live_cycle(db, world, sms, rain=2.5)
    run = db.scalar(select(PipelineRun))
    run.started_at = datetime.now(timezone.utc) - timedelta(hours=5)
    db.commit()
    lat, lon = point_with_tier("High")
    data = api.get("/api/v1/risk", params={"latitude": lat, "longitude": lon}).json()
    assert data["level"] is None and data["data"]["state"] == "stale"
    assert api.get("/api/v1/roads/at-risk").json()["roads"] == []


def test_outside_nairobi_is_rejected_clearly(api):
    response = api.get("/api/v1/risk", params={"latitude": KISUMU[0], "longitude": KISUMU[1]})
    assert response.status_code == 422 and "coverage" in response.json()["detail"]


def test_invalid_coordinates_are_rejected(api):
    assert api.get("/api/v1/risk", params={"latitude": 95, "longitude": 36}).status_code == 422


def test_road_search(api):
    results = api.get("/api/v1/roads/search", params={"q": "ngong"}).json()["results"]
    assert results and results[0]["road_name"].lower().startswith("ngong")
    assert api.get("/api/v1/roads/search", params={"q": "zzzzqq"}).json()["results"] == []


def test_roads_at_risk_only_when_it_rains(api, db, world, sms):
    live_cycle(db, world, sms, rain=0)
    assert api.get("/api/v1/roads/at-risk").json()["roads"] == []
    live_cycle(db, world, sms, rain=2.5)
    roads = api.get("/api/v1/roads/at-risk", params={"min_level": 3}).json()["roads"]
    assert roads and all(r["level"] == 3 for r in roads)


def test_official_alerts_endpoint(api, db, world, sms):
    expires = datetime.now(timezone.utc) + timedelta(days=1)
    world.kmd_items = [cap_xml(expires=expires.isoformat())]
    live_cycle(db, world, sms)
    alerts = api.get("/api/v1/alerts/official", params={"nairobi_only": True}).json()["kmd_alerts"]
    assert len(alerts) == 1 and alerts[0]["covers_nairobi"] and alerts[0]["instruction"] == "Stay safe."


def test_subscribe_verify_and_manage(api, otp_sink):
    assert api.post("/api/v1/subscribers", json=body(place_label="South B")).status_code == 202
    verified = api.post("/api/v1/subscribers/verify", json={"phone": "+254712345678", "code": code_from(otp_sink)})
    assert verified.status_code == 200
    headers = {"Authorization": f"Bearer {verified.json()['token']}"}
    me = api.get("/api/v1/subscribers/me", headers=headers).json()
    assert me["status"] == "active" and me["place_label"] == "South B" and me["min_level"] == 1

    high = point_with_tier("High")
    updated = api.patch("/api/v1/subscribers/me", headers=headers,
                        json={"latitude": high[0], "longitude": high[1], "language": "sw", "min_level": 1}).json()
    assert updated["local_susceptibility_tier"] == "High" and updated["language"] == "sw"
    assert api.get("/api/v1/subscribers/me/alerts", headers=headers).json() == {"alerts": []}


def test_consent_is_mandatory(api, otp_sink):
    assert api.post("/api/v1/subscribers", json=body(consent=False)).status_code == 422
    assert otp_sink == []


def test_bad_phone_and_location_send_no_sms(api, otp_sink):
    assert api.post("/api/v1/subscribers", json=body(phone="12345")).status_code == 422
    assert api.post("/api/v1/subscribers", json=body(point=KISUMU)).status_code == 422
    assert otp_sink == []


def test_unverified_number_receives_no_alerts(api, db, world, sms, otp_sink):
    api.post("/api/v1/subscribers", json=body(point=point_with_tier("High")))
    live_cycle(db, world, sms, rain=2.5)
    assert sms.sent == []


def test_wrong_codes_lock_out_after_five_attempts(api, otp_sink):
    api.post("/api/v1/subscribers", json=body())
    right = code_from(otp_sink)
    wrong = "000000" if right != "000000" else "111111"
    for _ in range(5):
        assert api.post("/api/v1/subscribers/verify", json={"phone": "0712345678", "code": wrong}).status_code == 400
    assert api.post("/api/v1/subscribers/verify", json={"phone": "0712345678", "code": right}).status_code == 400


def test_expired_code_is_rejected(api, db, otp_sink):
    api.post("/api/v1/subscribers", json=body())
    sub = db.scalar(select(Subscriber))
    sub.otp_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    db.commit()
    assert api.post("/api/v1/subscribers/verify", json={"phone": "0712345678", "code": code_from(otp_sink)}).status_code == 400


def test_code_is_single_use(api, otp_sink):
    api.post("/api/v1/subscribers", json=body())
    payload = {"phone": "0712345678", "code": code_from(otp_sink)}
    assert api.post("/api/v1/subscribers/verify", json=payload).status_code == 200
    assert api.post("/api/v1/subscribers/verify", json=payload).status_code == 400


def test_resend_is_throttled(api, otp_sink):
    assert api.post("/api/v1/subscribers", json=body()).status_code == 202
    assert api.post("/api/v1/subscribers", json=body()).status_code == 429
    assert len(otp_sink) == 1


def test_a_stranger_cannot_move_someone_elses_subscription(api, db, otp_sink):
    low, high = point_with_tier("Low"), point_with_tier("High")
    subscribe(api, otp_sink, "0712345678", low)
    sub = db.scalar(select(Subscriber))
    sub.otp_sent_at = None
    db.commit()
    assert api.post("/api/v1/subscribers", json=body(point=high)).status_code == 202   # never verified
    db.expire_all()
    assert db.scalar(select(Subscriber)).latitude == low[0]


def test_sms_failure_during_signup_is_reported(api, monkeypatch):
    class Down:
        name = "down"

        def send(self, to, message):
            from app.sms import SmsResult
            return SmsResult(False, "down", error="x")
    monkeypatch.setattr("app.api.subscribers.build_provider", lambda: Down())
    assert api.post("/api/v1/subscribers", json=body()).status_code == 502
    assert api.post("/api/v1/subscribers", json=body()).status_code == 502      # no cooldown was consumed


def test_me_needs_a_valid_token(api):
    assert api.get("/api/v1/subscribers/me").status_code == 401
    assert api.get("/api/v1/subscribers/me", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_delete_removes_all_personal_data(api, db, world, sms, otp_sink):
    token = subscribe(api, otp_sink, "0712345678", point_with_tier("High"))
    live_cycle(db, world, sms, rain=2.5)
    assert api.delete("/api/v1/subscribers/me", headers={"Authorization": f"Bearer {token}"}).status_code == 204
    db.expire_all()
    assert db.scalar(select(Subscriber)) is None and db.scalar(select(Delivery)) is None


def test_stop_opts_out_and_revokes_the_token(api, otp_sink):
    token = subscribe(api, otp_sink, "0712345678", point_with_tier("Low"))
    assert api.post("/webhooks/sms/inbound?secret=test-secret", json={"from": "254712345678", "text": "STOP"}).status_code == 200
    assert api.get("/api/v1/subscribers/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_inbound_webhook_needs_the_secret(api):
    assert api.post("/webhooks/sms/inbound?secret=wrong", json={"from": "0712345678", "text": "STOP"}).status_code == 403


def test_admin_routes_need_the_key(api):
    assert api.post("/admin/pipeline/run").status_code == 403
    assert api.get("/admin/stats", headers={"X-Admin-Key": "wrong"}).status_code == 403
    assert api.get("/admin/stats", headers=ADMIN).status_code == 200


def test_drill_previews_without_changing_state(api, db, world, sms, otp_sink):
    subscribe(api, otp_sink, "0712345678", point_with_tier("High"))
    sends_before = len(otp_sink)
    data = api.post("/admin/drill", headers=ADMIN, json={"rain_next_24h_mm": 60}).json()
    assert data["subscribers_alerted"] == 1 and data["by_level"]["severe"] == 1
    db.expire_all()
    assert db.scalar(select(Subscriber)).last_alert_level == 0
    assert db.scalar(select(Delivery)) is None and len(otp_sink) == sends_before


def test_drill_sms_is_labelled_and_goes_only_to_the_named_subscriber(api, otp_sink):
    subscribe(api, otp_sink, "0712345678", point_with_tier("High"))
    subscribe(api, otp_sink, "0711111111", point_with_tier("High"))
    before = len(otp_sink)
    data = api.post("/admin/drill", headers=ADMIN, json={"rain_next_24h_mm": 60, "send_to_phone": "0711111111"}).json()
    assert data["sent_to"]["ok"] and len(otp_sink) == before + 1
    assert otp_sink[-1][0] == "+254711111111" and otp_sink[-1][1].startswith("DRILL, NOT REAL")
    assert len(otp_sink[-1][1]) <= 160


def test_drill_refuses_unverified_numbers(api, otp_sink):
    response = api.post("/admin/drill", headers=ADMIN, json={"rain_next_24h_mm": 60, "send_to_phone": "0799999999"})
    assert response.status_code == 422 and otp_sink == []


def test_neighbour_radius_links_nearby_zones_only():
    from app.geo import get_store
    from app.risk import NEIGHBOUR_RADIUS_KM
    links = get_store().neighbours(NEIGHBOUR_RADIUS_KM)
    assert all(zone in near for zone, near in links.items())
    assert all(len(near) >= 1 for near in links.values())
    assert any(len(near) < len(links) for near in links.values())


def test_hosted_postgres_urls_use_the_installed_driver():
    from app.db import normalise_database_url as norm
    assert norm("postgres://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert norm("postgresql://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert norm("postgresql+psycopg://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert norm("sqlite:///x.db").startswith("sqlite")
