"""Who gets which SMS, and when. Designed so a noisy hourly forecast does not become a noisy inbox."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import messages
from app.config import get_settings
from app.models import Delivery, OfficialAlert, PipelineRun, Subscriber, ZoneAssessment
from app.risk import NONE, local_level
from app.sms import SmsProvider

log = logging.getLogger("floodguard.alerts")

ALERT, ALL_CLEAR = "flood", "all_clear"
ALL_CLEAR_MIN_HOURS = 6             # a forecast dipping under a threshold for one run is not an all-clear
OFFICIAL_MIN_GAP_HOURS = 24         # KMD can reissue the same advisory several times in a day
OFFICIAL_DEFAULT_LIFETIME_HOURS = 48


def decide(level: int, min_level: int, last_level: int, last_at: datetime | None, now: datetime,
           cooldown_hours: int) -> str | None:
    """Return ALERT, ALL_CLEAR or None for one subscriber."""
    since_last = (now - last_at) if last_at else None
    if level >= min_level:
        if level > last_level or since_last is None or since_last >= timedelta(hours=cooldown_hours):
            return ALERT
        return None
    if last_level > NONE and since_last is not None and since_last >= timedelta(hours=ALL_CLEAR_MIN_HOURS):
        return ALL_CLEAR
    return None


@dataclass
class SendCounts:
    sent: int = 0
    failed: int = 0
    held_back: int = 0


def _send(db: Session, sms: SmsProvider, sub: Subscriber, kind: str, level: int, message: str,
          run_id: int | None = None, official_id: int | None = None) -> bool:
    result = sms.send(sub.phone, message)
    db.add(Delivery(subscriber_id=sub.id, run_id=run_id, official_alert_id=official_id, kind=kind, level=level,
                    message=message, status="sent" if result.ok else "failed", provider=result.provider,
                    provider_reference=result.reference, error=result.error))
    return result.ok


def dispatch_flood_alerts(db: Session, run: PipelineRun, sms: SmsProvider, now: datetime, counts: SendCounts) -> None:
    """Send level changes for one finished run. Zones without a usable forecast are left alone: missing
    data must never produce an alert or an all-clear."""
    settings = get_settings()
    zones = {z.zone_id: z for z in db.scalars(select(ZoneAssessment).where(ZoneAssessment.run_id == run.id))
             if z.available}
    subscribers = db.scalars(select(Subscriber).where(Subscriber.status == "active")).all()
    due: list[tuple[int, Subscriber, str]] = []
    for sub in subscribers:
        zone = zones.get(sub.zone_id)
        if zone is None:
            continue
        level = local_level(zone.rain_band, sub.local_tier)
        action = decide(level, sub.min_level, sub.last_alert_level, sub.last_alert_at, now,
                        settings.alert_cooldown_hours)
        if action:
            due.append((level, sub, action))

    due.sort(key=lambda item: -item[0])     # credits may run out: the most severe go first
    for level, sub, action in due:
        if counts.sent + counts.failed >= settings.max_sms_per_cycle:
            counts.held_back += 1
            continue
        zone = zones[sub.zone_id]
        if action == ALERT:
            text = messages.flood_message(level, sub.place_label, zone.rain_area_24h_mm, sub.language,
                                          settings.sms_opt_out_hint)
        else:
            text = messages.all_clear_message(sub.place_label, sub.language)
        ok = _send(db, sms, sub, action, level, text, run_id=run.id)
        counts.sent += ok
        counts.failed += not ok
        if ok:                              # a failed send stays due and is retried next cycle
            sub.last_alert_level = level if action == ALERT else NONE
            sub.last_alert_at = now
        db.commit()
    if counts.held_back:
        log.warning("SMS cap reached: %d messages held back until the next cycle", counts.held_back)


def active_official_alerts(db: Session, now: datetime) -> list[OfficialAlert]:
    def active(alert: OfficialAlert) -> bool:
        if alert.expires_at:
            return alert.expires_at > now
        return alert.sent_at is not None and now - alert.sent_at < timedelta(hours=OFFICIAL_DEFAULT_LIFETIME_HOURS)
    return [a for a in db.scalars(select(OfficialAlert).order_by(OfficialAlert.sent_at.desc())) if active(a)]


def dispatch_official_alerts(db: Session, run: PipelineRun, sms: SmsProvider, now: datetime, counts: SendCounts) -> None:
    """Forward KMD alerts covering a subscriber's zone. These do not depend on our own forecast being up."""
    settings = get_settings()
    subscribers = db.scalars(select(Subscriber).where(Subscriber.status == "active")).all()
    gap_start = now - timedelta(hours=OFFICIAL_MIN_GAP_HOURS)
    for alert in active_official_alerts(db, now):
        covered = set(alert.zones_covered or [])
        for sub in subscribers:
            if sub.zone_id not in covered:
                continue
            sent = select(Delivery.id).where(Delivery.subscriber_id == sub.id, Delivery.kind == "official",
                                             Delivery.status == "sent", Delivery.created_at >= gap_start)
            already = select(Delivery.id).where(Delivery.subscriber_id == sub.id, Delivery.status == "sent",
                                                Delivery.official_alert_id == alert.id)
            if db.scalar(sent.limit(1)) or db.scalar(already.limit(1)):
                continue
            if counts.sent + counts.failed >= settings.max_sms_per_cycle:
                counts.held_back += 1
                continue
            ok = _send(db, sms, sub, "official", 0, messages.official_message(alert.headline, sub.language),
                       run_id=run.id, official_id=alert.id)
            counts.sent += ok
            counts.failed += not ok
            db.commit()


def drill_preview(db: Session, band_by_zone: dict[str, int], rain_mm: float) -> list[dict]:
    """What a scenario would send. Reads subscribers but changes nothing."""
    settings = get_settings()
    rows = []
    for sub in db.scalars(select(Subscriber).where(Subscriber.status == "active")):
        level = local_level(band_by_zone.get(sub.zone_id, 0), sub.local_tier)
        if level >= sub.min_level:
            rows.append({"subscriber_id": sub.id, "zone_id": sub.zone_id, "level": level,
                         "message": messages.flood_message(level, sub.place_label, rain_mm, sub.language,
                                                           settings.sms_opt_out_hint)})
    return rows
