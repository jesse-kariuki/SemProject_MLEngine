"""Database tables."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base, UTCDateTime


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Subscriber(Base):
    __tablename__ = "subscribers"

    id: Mapped[int] = mapped_column(primary_key=True)
    phone: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending")   # pending | active | unsubscribed
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    place_label: Mapped[str | None] = mapped_column(String(40))
    zone_id: Mapped[str | None] = mapped_column(String(16), index=True)
    local_tier: Mapped[str | None] = mapped_column(String(16))
    language: Mapped[str] = mapped_column(String(2), default="en")
    min_level: Mapped[int] = mapped_column(Integer, default=1)
    consent_at: Mapped[datetime | None] = mapped_column(UTCDateTime)

    # Requested details wait here until the code is verified, so nobody can alter another person's
    # subscription just by knowing their number.
    pending: Mapped[dict | None] = mapped_column(JSON)
    otp_hash: Mapped[str | None] = mapped_column(String(64))
    otp_expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    otp_sent_at: Mapped[datetime | None] = mapped_column(UTCDateTime, index=True)
    otp_attempts: Mapped[int] = mapped_column(Integer, default=0)
    token_hash: Mapped[str | None] = mapped_column(String(64), index=True)

    last_alert_level: Mapped[int] = mapped_column(Integer, default=0)
    last_alert_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    status: Mapped[str] = mapped_column(String(16), default="running")   # running | ok | partial | failed
    issue_time: Mapped[datetime | None] = mapped_column(UTCDateTime)
    zones_available: Mapped[int] = mapped_column(Integer, default=0)
    zones_total: Mapped[int] = mapped_column(Integer, default=0)
    sms_sent: Mapped[int] = mapped_column(Integer, default=0)
    sms_failed: Mapped[int] = mapped_column(Integer, default=0)
    sms_held_back: Mapped[int] = mapped_column(Integer, default=0)
    source_errors: Mapped[dict] = mapped_column(JSON, default=dict)
    gdacs_events: Mapped[list] = mapped_column(JSON, default=list)


class ZoneAssessment(Base):
    __tablename__ = "zone_assessments"
    __table_args__ = (Index("ix_assessment_run_zone", "run_id", "zone_id", unique=True),)

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("pipeline_runs.id", ondelete="CASCADE"))
    zone_id: Mapped[str] = mapped_column(String(16))
    available: Mapped[bool] = mapped_column(Boolean)
    rain_band: Mapped[int] = mapped_column(Integer, default=0)
    rain_next_6h_mm: Mapped[float | None] = mapped_column(Float)
    rain_next_24h_mm: Mapped[float | None] = mapped_column(Float)
    rain_area_24h_mm: Mapped[float | None] = mapped_column(Float)
    rain_prev_72h_mm: Mapped[float | None] = mapped_column(Float)
    max_hourly_mm: Mapped[float | None] = mapped_column(Float)
    precip_probability_max: Mapped[float | None] = mapped_column(Float)
    river_discharge_m3s: Mapped[float | None] = mapped_column(Float)
    river_p99_m3s: Mapped[float | None] = mapped_column(Float)
    reasons: Mapped[list] = mapped_column(JSON, default=list)


class OfficialAlert(Base):
    __tablename__ = "official_alerts"

    id: Mapped[int] = mapped_column(primary_key=True)
    identifier: Mapped[str] = mapped_column(String(128), unique=True)
    source: Mapped[str] = mapped_column(String(16), default="kmd")
    event: Mapped[str] = mapped_column(String(120))
    headline: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text)
    instruction: Mapped[str] = mapped_column(Text)
    severity: Mapped[str] = mapped_column(String(24))
    area: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(Text)
    sent_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    zones_covered: Mapped[list] = mapped_column(JSON, default=list)


class Delivery(Base):
    __tablename__ = "deliveries"

    id: Mapped[int] = mapped_column(primary_key=True)
    subscriber_id: Mapped[int] = mapped_column(ForeignKey("subscribers.id", ondelete="CASCADE"), index=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("pipeline_runs.id", ondelete="SET NULL"))
    official_alert_id: Mapped[int | None] = mapped_column(ForeignKey("official_alerts.id", ondelete="SET NULL"))
    kind: Mapped[str] = mapped_column(String(16))                        # flood | all_clear | official | drill
    level: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16))                      # sent | failed
    provider: Mapped[str] = mapped_column(String(16))
    provider_reference: Mapped[str | None] = mapped_column(String(128))
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
