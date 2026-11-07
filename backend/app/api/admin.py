"""Operator endpoints. Disabled unless ADMIN_API_KEY is set."""
from __future__ import annotations

import hmac
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import alerts, messages
from app.config import get_settings
from app.db import get_db
from app.geo import get_store
from app.models import Delivery, PipelineRun, Subscriber
from app.phone import normalise_phone
from app.pipeline import run_once
from app.risk import LEVEL_NAMES, apply_modifiers, local_level, rain_band
from app.schemas import DrillRequest
from app.sms import build_provider


def require_admin(x_admin_key: str = Header(default="")) -> None:
    expected = get_settings().admin_api_key
    if not expected or not hmac.compare_digest(x_admin_key, expected):
        raise HTTPException(403, "Forbidden")


router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


@router.post("/pipeline/run", summary="Run a live monitoring cycle now")
def run_pipeline() -> dict:
    run = run_once()
    if run is None:
        raise HTTPException(409, "A cycle is already running.")
    return {"run_id": run.id, "status": run.status, "zones_available": run.zones_available,
            "zones_total": run.zones_total, "sms_sent": run.sms_sent, "sms_failed": run.sms_failed,
            "source_errors": run.source_errors}


@router.post("/drill", summary="Preview a rainfall scenario without touching live state")
def drill(body: DrillRequest, db: Session = Depends(get_db)) -> dict:
    """Shows who a scenario would alert. Nothing reaches subscribers except the one named in send_to_phone,
    and that message is marked DRILL."""
    store = get_store()
    unknown = set(body.zone_ids or []) - set(store.zones)
    if unknown:
        raise HTTPException(422, f"Unknown zones: {sorted(unknown)}")
    band, reasons = apply_modifiers(rain_band(body.rain_next_24h_mm), body.rain_next_24h_mm, body.rain_prev_72h_mm,
                                    None, None)
    bands = {zone_id: band for zone_id in (body.zone_ids or store.zones)}
    preview = alerts.drill_preview(db, bands, body.rain_next_24h_mm)

    sent_to = None
    if body.send_to_phone:
        phone = normalise_phone(body.send_to_phone)
        sub = db.scalar(select(Subscriber).where(Subscriber.phone == phone, Subscriber.status == "active"))
        if sub is None:
            raise HTTPException(422, "send_to_phone must be an active, verified subscriber.")
        level = max(local_level(bands.get(sub.zone_id, 0), sub.local_tier), 1)
        text = messages.flood_message(level, sub.place_label, body.rain_next_24h_mm, sub.language)
        text = ("DRILL, NOT REAL: " + text)[:messages.SMS_LIMIT]
        result = build_provider().send(sub.phone, text)
        db.add(Delivery(subscriber_id=sub.id, kind="drill", level=level, message=text,
                        status="sent" if result.ok else "failed", provider=result.provider,
                        provider_reference=result.reference, error=result.error))
        db.commit()
        sent_to = {"phone": sub.phone, "ok": result.ok, "message": text, "error": result.error}
    return {"rain_band": band, "reasons": reasons, "subscribers_alerted": len(preview),
            "by_level": {LEVEL_NAMES[lv]: sum(p["level"] == lv for p in preview) for lv in (1, 2, 3)},
            "sample": preview[:5], "sent_to": sent_to}


@router.get("/stats", summary="Subscribers, deliveries and recent cycles")
def stats(db: Session = Depends(get_db)) -> dict:
    by_status = dict(db.execute(select(Subscriber.status, func.count()).group_by(Subscriber.status)).all())
    by_delivery = dict(db.execute(select(Delivery.kind, func.count()).where(Delivery.status == "sent")
                                  .group_by(Delivery.kind)).all())
    failed = db.scalar(select(func.count()).select_from(Delivery).where(Delivery.status == "failed"))
    day_ago = datetime.now(timezone.utc) - timedelta(hours=24)
    runs = db.scalars(select(PipelineRun).where(PipelineRun.started_at >= day_ago)).all()
    return {"subscribers": by_status, "deliveries_sent": by_delivery, "deliveries_failed": failed,
            "cycles_24h": {"total": len(runs), "ok": sum(r.status == "ok" for r in runs),
                           "partial": sum(r.status == "partial" for r in runs),
                           "failed": sum(r.status == "failed" for r in runs)}}
