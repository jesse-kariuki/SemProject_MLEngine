"""Test setup: a throwaway database, no scheduler, and no real network or SMS."""
from __future__ import annotations

import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

_tmp = tempfile.mkdtemp(prefix="floodguard-test-")
os.environ.update({"DATABASE_URL": f"sqlite:///{Path(_tmp) / 'test.db'}", "SCHEDULER_ENABLED": "false",
                   "SMS_PROVIDER": "console", "ADMIN_API_KEY": "test-admin", "INBOUND_WEBHOOK_SECRET": "test-secret",
                   "HTTP_RETRIES": "1"})

import httpx  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db import Base, get_engine, init_db, session_factory  # noqa: E402
from app.geo import get_store  # noqa: E402
from app.main import app  # noqa: E402
from app.sms import SmsResult  # noqa: E402

NOW = datetime(2026, 4, 28, 9, 30, tzinfo=timezone.utc)


class FakeSms:
    """Records messages. Set `fail=True` to simulate a provider outage."""
    name = "fake"

    def __init__(self):
        self.sent: list[tuple[str, str]] = []
        self.fail = False

    def send(self, to: str, message: str) -> SmsResult:
        if self.fail:
            return SmsResult(False, self.name, error="provider down")
        self.sent.append((to, message))
        return SmsResult(True, self.name, "ref")


def forecast_payload(n: int, rain_per_hour: float, past_rain_per_hour: float = 0.0, now: datetime = NOW) -> list[dict]:
    start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=3)
    times = [start + timedelta(hours=h) for h in range(24 * 6)]
    issue = now.replace(minute=0, second=0, microsecond=0)
    rain = [past_rain_per_hour if t < issue else rain_per_hour for t in times]
    hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times], "precipitation": rain,
              "precipitation_probability": [70] * len(times)}
    return [{"hourly": hourly} for _ in range(n)]


class FakeWorld:
    """Stands in for every external API through httpx.MockTransport."""

    def __init__(self):
        self.rain_per_hour = 0.0
        self.past_rain_per_hour = 0.0
        self.now = NOW
        self.forecast_status = 200
        self.flood_status = 200
        self.kmd_status = 200
        self.kmd_items: list[str] = []
        self.river_q = 0.1

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        count = len(request.url.params.get("latitude", "0").split(","))
        if "api.open-meteo.com/v1/forecast" in url and "flood" not in url:
            if self.forecast_status != 200:
                return httpx.Response(self.forecast_status)
            return httpx.Response(200, json=forecast_payload(count, self.rain_per_hour, self.past_rain_per_hour, self.now))
        if "flood-api" in url:
            if self.flood_status != 200:
                return httpx.Response(self.flood_status)
            day = self.now.date().isoformat()
            return httpx.Response(200, json=[{"daily": {"time": [day], "river_discharge": [self.river_q]}}] * count)
        if url.endswith("rss.xml"):
            if self.kmd_status != 200:
                return httpx.Response(self.kmd_status)
            items = "".join(f"<item><link>https://meteo.go.ke/api/cap/{i}.xml</link></item>" for i in range(len(self.kmd_items)))
            return httpx.Response(200, text=f"<rss><channel>{items}</channel></rss>")
        match = re.search(r"/api/cap/(\d+)\.xml", url)
        if match:
            return httpx.Response(200, text=self.kmd_items[int(match.group(1))])
        if "gdacs" in url:
            return httpx.Response(200, json={"features": []})
        return httpx.Response(404)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))


@pytest.fixture(autouse=True)
def clean_db():
    Base.metadata.drop_all(get_engine())
    init_db()
    yield


@pytest.fixture
def db():
    with session_factory()() as session:
        yield session


@pytest.fixture
def api():
    with TestClient(app) as client:
        yield client


@pytest.fixture
def world():
    return FakeWorld()


@pytest.fixture
def sms():
    return FakeSms()


@pytest.fixture
def otp_sink(monkeypatch):
    """Capture verification codes that the API sends through the console provider."""
    sent: list[tuple[str, str]] = []

    class Sink:
        name = "sink"

        def send(self, to, message):
            sent.append((to, message))
            return SmsResult(True, "sink", "ref")

    monkeypatch.setattr("app.api.subscribers.build_provider", lambda: Sink())
    monkeypatch.setattr("app.api.admin.build_provider", lambda: Sink())
    return sent


def point_with_tier(tier: str) -> tuple[float, float]:
    """A real road location whose local tier is `tier`."""
    store = get_store()
    for i in range(0, len(store.lat), 37):
        place = store.locate(float(store.lat[i]), float(store.lon[i]), 300)
        if place and place.tier == tier:
            return float(store.lat[i]), float(store.lon[i])
    raise AssertionError(f"no point with tier {tier}")


def subscribe(api: TestClient, sink: list, phone: str, point: tuple[float, float], **extra) -> str:
    response = api.post("/api/v1/subscribers", json={"phone": phone, "latitude": point[0], "longitude": point[1],
                                                     "consent": True, **extra})
    assert response.status_code == 202, response.text
    code = re.search(r"\d{6}", sink[-1][1]).group()
    verified = api.post("/api/v1/subscribers/verify", json={"phone": phone, "code": code})
    assert verified.status_code == 200, verified.text
    return verified.json()["token"]


def cap_xml(identifier="urn:test:1", event="Heavy rainfall", status="Actual", msg_type="Alert", scope="Public",
            polygon="-2,36 -2,37.5 -0.5,37.5 -0.5,36 -2,36", expires="2026-04-30T09:00:00+03:00",
            headline="Heavy rainfall exceeding 50 mm expected over Nairobi"):
    poly = f"<cap:polygon>{polygon}</cap:polygon>" if polygon else ""
    return (f'<cap:alert xmlns:cap="urn:oasis:names:tc:emergency:cap:1.2"><cap:identifier>{identifier}</cap:identifier>'
            f"<cap:sent>2026-04-27T18:00:00+03:00</cap:sent><cap:status>{status}</cap:status>"
            f"<cap:msgType>{msg_type}</cap:msgType><cap:scope>{scope}</cap:scope><cap:info><cap:event>{event}</cap:event>"
            f"<cap:severity>Moderate</cap:severity><cap:expires>{expires}</cap:expires><cap:headline>{headline}</cap:headline>"
            f"<cap:description>Details.</cap:description><cap:instruction>Stay safe.</cap:instruction>"
            f"<cap:area><cap:areaDesc>Central Kenya</cap:areaDesc>{poly}</cap:area></cap:info></cap:alert>")



