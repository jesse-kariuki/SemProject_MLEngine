"""Opt-in subscriptions. A number only receives alerts after its owner confirms a code sent to it."""
from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import messages
from app.config import get_settings
from app.db import get_db
from app.geo import Place, get_store
from app.models import Delivery, Subscriber
from app.phone import normalise_phone
from app.schemas import InboundSms, SubscribeRequest, SubscriberUpdate, VerifyRequest
from app.sms import build_provider

router = APIRouter(tags=["subscribers"])
MAX_CODES_PER_HOUR = 200            # stops a script from draining SMS credit across many numbers
BAD_CODE = "That code is wrong or has expired. Request a new one."
STOP_WORDS = {"STOP", "SIMAMA", "ACHA"}


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _code_hash(phone: str, code: str) -> str:
    return _hash(f"{phone}:{code}")


def _locate(latitude: float, longitude: float) -> Place:
    place = get_store().locate(latitude, longitude, get_settings().default_radius_m)
    if place is None:
        raise HTTPException(422, "This location is outside FloodGuard coverage (Nairobi County roads).")
    return place


def _phone_or_422(raw: str) -> str:
    phone = normalise_phone(raw)
    if phone is None:
        raise HTTPException(422, "Enter a Kenyan mobile number, for example 0712345678.")
    return phone


def current_subscriber(authorization: str = Header(default=""), db: Session = Depends(get_db)) -> Subscriber:
    scheme, _, token = authorization.partition(" ")
    sub = None
    if scheme.lower() == "bearer" and token:
        sub = db.scalar(select(Subscriber).where(Subscriber.token_hash == _hash(token)))
    if sub is None or sub.status != "active":
        raise HTTPException(401, "Sign in again by verifying your number.")
    return sub


def _public(sub: Subscriber) -> dict:
    return {"phone": sub.phone, "status": sub.status, "latitude": sub.latitude, "longitude": sub.longitude,
            "place_label": sub.place_label, "zone_id": sub.zone_id, "local_susceptibility_tier": sub.local_tier,
            "language": sub.language, "min_level": sub.min_level, "last_alert_level": sub.last_alert_level,
            "last_alert_at": sub.last_alert_at}


@router.post("/api/v1/subscribers", status_code=202, summary="Start a subscription: sends a code by SMS")
def subscribe(body: SubscribeRequest, response: Response, db: Session = Depends(get_db)) -> dict:
    settings = get_settings()
    if not body.consent:
        raise HTTPException(422, "Consent is required to receive SMS alerts.")
    phone = _phone_or_422(body.phone)
    _locate(body.latitude, body.longitude)
    now = datetime.now(timezone.utc)

    sub = db.scalar(select(Subscriber).where(Subscriber.phone == phone))
    if sub and sub.otp_sent_at and now - sub.otp_sent_at < timedelta(seconds=settings.otp_resend_seconds):
        response.headers["Retry-After"] = str(settings.otp_resend_seconds)
        raise HTTPException(429, "A code was just sent. Wait a minute before requesting another.")
    hour_ago = now - timedelta(hours=1)
    if (db.scalar(select(func.count()).select_from(Subscriber).where(Subscriber.otp_sent_at >= hour_ago)) or 0) \
            >= MAX_CODES_PER_HOUR:
        raise HTTPException(429, "Too many sign-ups right now. Try again later.")

    if sub is None:
        sub = Subscriber(phone=phone)
        db.add(sub)
    code = f"{secrets.randbelow(10 ** 6):06d}"
    result = build_provider().send(phone, messages.otp_message(code, settings.otp_ttl_minutes, body.language))
    if not result.ok:
        db.rollback()
        raise HTTPException(502, "We could not send the SMS. Try again shortly.")
    sub.pending = body.model_dump(include={"latitude", "longitude", "place_label", "language", "min_level"})
    sub.otp_hash = _code_hash(phone, code)
    sub.otp_expires_at = now + timedelta(minutes=settings.otp_ttl_minutes)
    sub.otp_sent_at = now
    sub.otp_attempts = 0
    db.commit()
    return {"status": "verification_sent", "phone": phone, "expires_in_seconds": settings.otp_ttl_minutes * 60}


@router.post("/api/v1/subscribers/verify", summary="Confirm the code. Returns a token for /subscribers/me")
def verify(body: VerifyRequest, db: Session = Depends(get_db)) -> dict:
    settings = get_settings()
    phone = _phone_or_422(body.phone)
    sub = db.scalar(select(Subscriber).where(Subscriber.phone == phone))
    now = datetime.now(timezone.utc)
    if (sub is None or not sub.otp_hash or not sub.pending or sub.otp_expires_at <= now
            or sub.otp_attempts >= settings.otp_max_attempts):
        raise HTTPException(400, BAD_CODE)
    if not hmac.compare_digest(sub.otp_hash, _code_hash(phone, body.code)):
        sub.otp_attempts += 1
        db.commit()
        raise HTTPException(400, BAD_CODE)

    place = _locate(sub.pending["latitude"], sub.pending["longitude"])
    for field, value in sub.pending.items():
        setattr(sub, field, value)
    sub.zone_id, sub.local_tier = place.zone_id, place.tier
    sub.status, sub.consent_at = "active", now
    sub.pending = sub.otp_hash = sub.otp_expires_at = None
    token = secrets.token_urlsafe(32)
    sub.token_hash = _hash(token)
    db.commit()
    return {"token": token, "subscriber": _public(sub)}


@router.get("/api/v1/subscribers/me", summary="My subscription")
def me(sub: Subscriber = Depends(current_subscriber)) -> dict:
    return _public(sub)


@router.patch("/api/v1/subscribers/me", summary="Change my location, language or alert level")
def update_me(body: SubscriberUpdate, sub: Subscriber = Depends(current_subscriber),
              db: Session = Depends(get_db)) -> dict:
    changes = body.model_dump(exclude_unset=True)
    if ("latitude" in changes) != ("longitude" in changes):
        raise HTTPException(422, "Send latitude and longitude together.")
    if "latitude" in changes:
        place = _locate(changes["latitude"], changes["longitude"])
        sub.zone_id, sub.local_tier = place.zone_id, place.tier
    for field, value in changes.items():
        setattr(sub, field, value)
    db.commit()
    return _public(sub)


@router.get("/api/v1/subscribers/me/alerts", summary="My alert history")
def my_alerts(sub: Subscriber = Depends(current_subscriber), db: Session = Depends(get_db)) -> dict:
    rows = db.scalars(select(Delivery).where(Delivery.subscriber_id == sub.id, Delivery.status == "sent")
                      .order_by(Delivery.id.desc()).limit(50))
    return {"alerts": [{"kind": d.kind, "level": d.level, "message": d.message, "sent_at": d.created_at} for d in rows]}


@router.delete("/api/v1/subscribers/me", status_code=204, response_class=Response,
               summary="Delete my subscription and all my data")
def delete_me(sub: Subscriber = Depends(current_subscriber), db: Session = Depends(get_db)):
    for delivery in db.scalars(select(Delivery).where(Delivery.subscriber_id == sub.id)):
        db.delete(delivery)
    db.delete(sub)
    db.commit()


@router.post("/webhooks/sms/inbound", summary="Provider callback for replies: handles STOP")
def inbound_sms(body: InboundSms, secret: str = "", db: Session = Depends(get_db)) -> dict:
    expected = get_settings().inbound_webhook_secret
    if not expected or not hmac.compare_digest(secret, expected):
        raise HTTPException(403, "Forbidden")
    phone = normalise_phone(body.from_)
    words = body.text.strip().upper().split()
    if phone and words and words[0] in STOP_WORDS:
        sub = db.scalar(select(Subscriber).where(Subscriber.phone == phone))
        if sub:                     # the row stays, so this number cannot be alerted until it opts in again
            sub.status, sub.token_hash = "unsubscribed", None
            db.commit()
    return {"ok": True}
