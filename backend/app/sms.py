"""SMS delivery behind one small interface, so changing provider touches only this file."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import httpx

from app.config import Settings, get_settings

log = logging.getLogger("floodguard.sms")


@dataclass
class SmsResult:
    ok: bool
    provider: str
    reference: str | None = None
    error: str | None = None


class SmsProvider(Protocol):
    name: str

    def send(self, to: str, message: str) -> SmsResult: ...


class ConsoleProvider:
    """Logs instead of sending. The default until real credentials are set."""
    name = "console"

    def send(self, to: str, message: str) -> SmsResult:
        log.info("SMS (not sent) to %s***: %s", to[:7], message)
        return SmsResult(True, self.name, "console")


class ThriveWaveProvider:
    """ThriveWave adapter. UNVERIFIED: its API documentation is not public, so the request below is a
    placeholder. Match `send` to the docs from the hackathon organisers before relying on it."""
    name = "thrivewave"

    def __init__(self, settings: Settings):
        if not (settings.thrivewave_api_url and settings.thrivewave_api_key):
            raise ValueError("SMS_PROVIDER=thrivewave needs THRIVEWAVE_API_URL and THRIVEWAVE_API_KEY")
        self.url = settings.thrivewave_api_url
        self.key = settings.thrivewave_api_key
        self.sender = settings.thrivewave_sender_id

    def send(self, to: str, message: str) -> SmsResult:
        try:
            response = httpx.post(self.url, headers={"Authorization": f"Bearer {self.key}"}, timeout=20,
                                  json={"to": to, "message": message, "sender_id": self.sender})
            response.raise_for_status()
        except httpx.HTTPError as err:
            return SmsResult(False, self.name, error=str(err)[:300])
        return SmsResult(True, self.name, response.text[:120])


def build_provider(settings: Settings | None = None) -> SmsProvider:
    settings = settings or get_settings()
    if settings.sms_provider == "thrivewave":
        return ThriveWaveProvider(settings)
    if settings.sms_provider == "console":
        return ConsoleProvider()
    raise ValueError(f"Unknown SMS_PROVIDER: {settings.sms_provider}")
