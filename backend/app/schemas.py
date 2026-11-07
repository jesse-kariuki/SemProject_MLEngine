"""Request bodies. Field limits here are the first line of input validation."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Language = Literal["en", "sw"]
MinLevel = Field(default=1, ge=1, le=3, description="Lowest alert level to receive: 1 watch, 2 warning, 3 severe")


class SubscribeRequest(BaseModel):
    phone: str = Field(max_length=20, examples=["0712345678"])
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    place_label: str | None = Field(default=None, max_length=40, examples=["South B"])
    language: Language = "en"
    min_level: int = MinLevel
    consent: bool = Field(description="The user agreed to receive flood SMS at this number")


class VerifyRequest(BaseModel):
    phone: str = Field(max_length=20)
    code: str = Field(min_length=6, max_length=6, pattern=r"^\d{6}$")


class SubscriberUpdate(BaseModel):
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    place_label: str | None = Field(default=None, max_length=40)
    language: Language | None = None
    min_level: int | None = Field(default=None, ge=1, le=3)


class InboundSms(BaseModel):
    from_: str = Field(alias="from", max_length=20)
    text: str = Field(max_length=500)


class DrillRequest(BaseModel):
    rain_next_24h_mm: float = Field(ge=0, le=500, examples=[60])
    rain_prev_72h_mm: float = Field(default=0, ge=0, le=1000)
    zone_ids: list[str] | None = Field(default=None, description="Zones to apply it to. Omit for all zones")
    send_to_phone: str | None = Field(default=None, max_length=20, description="One verified subscriber to text")
