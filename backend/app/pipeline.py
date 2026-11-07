"""One monitoring cycle: fetch live data, assess every zone, store it, send alerts."""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app import alerts
from app.db import session_factory
from app.geo import GeoStore, get_store
from app.models import OfficialAlert, PipelineRun, ZoneAssessment
from app.risk import NEIGHBOUR_RADIUS_KM, assess_zone, spread_to_neighbours
from app.sms import SmsProvider, build_provider
from app.sources import gdacs, kmd_cap, openmeteo
from app.sources.http import SourceError

log = logging.getLogger("floodguard.pipeline")
RETENTION_DAYS = 30
_cycle_lock = threading.Lock()


def _store_official(db: Session, store: GeoStore, items: list[kmd_cap.CapAlert]) -> None:
    for item in items:
        if not item.identifier or db.scalar(select(OfficialAlert.id).where(OfficialAlert.identifier == item.identifier)):
            continue
        covered = [z.zone_id for z in store.zones.values() if kmd_cap.covers(item.polygons, z.lat, z.lon)]
        if not item.polygons and "nairobi" in item.area.lower():    # text-only alerts name the area instead
            covered = list(store.zones)
        db.add(OfficialAlert(identifier=item.identifier, event=item.event[:120], headline=item.headline,
                             description=item.description, instruction=item.instruction, severity=item.severity[:24],
                             area=item.area, url=item.url, sent_at=item.sent, expires_at=item.expires,
                             zones_covered=covered))
    db.commit()


def _assess_zones(db: Session, store: GeoStore, run: PipelineRun, client: httpx.Client, issue: datetime,
                  errors: dict) -> None:
    zones = list(store.zones.values())
    series = openmeteo.fetch_forecast(client, zones)
    river: dict[str, float] = {}
    try:
        river = openmeteo.fetch_river(client, zones, issue.date())
    except (SourceError, ValueError) as err:    # rain alone can still drive alerts
        errors["open_meteo_flood"] = str(err)[:300]
    risks = {}
    for zone in zones:
        s = series[zone.zone_id]
        risks[zone.zone_id] = assess_zone(zone.zone_id, s.times, s.precipitation, issue, s.probability,
                                          river.get(zone.zone_id), zone.river_p99)
    spread_to_neighbours(risks, store.neighbours(NEIGHBOUR_RADIUS_KM))
    for risk in risks.values():
        db.add(ZoneAssessment(run_id=run.id, zone_id=risk.zone_id, available=risk.available, rain_band=risk.rain_band,
                              rain_next_6h_mm=risk.rain_next_6h_mm, rain_next_24h_mm=risk.rain_next_24h_mm,
                              rain_area_24h_mm=risk.rain_area_24h_mm,
                              rain_prev_72h_mm=risk.rain_prev_72h_mm, max_hourly_mm=risk.max_hourly_mm,
                              precip_probability_max=risk.precip_probability_max,
                              river_discharge_m3s=risk.river_discharge_m3s, river_p99_m3s=risk.river_p99_m3s,
                              reasons=risk.reasons))
        run.zones_available += risk.available
    run.zones_total = len(zones)


def run_cycle(db: Session, client: httpx.Client, sms: SmsProvider, store: GeoStore | None = None,
              now: datetime | None = None) -> PipelineRun:
    """Run one cycle. Each source fails on its own: a dead feed is recorded, not fatal to the others."""
    store = store or get_store()
    now = now or datetime.now(timezone.utc)
    issue = now.replace(minute=0, second=0, microsecond=0)
    run = PipelineRun(started_at=now, issue_time=issue)
    db.add(run)
    db.commit()
    errors: dict[str, str] = {}

    try:
        _assess_zones(db, store, run, client, issue, errors)
        run.status = "ok" if run.zones_available == run.zones_total else "partial"
    except (SourceError, ValueError, KeyError) as err:
        db.rollback()
        errors["open_meteo_forecast"] = str(err)[:300]
        run.status = "failed"
    db.commit()

    try:
        _store_official(db, store, kmd_cap.fetch_alerts(client))
    except (SourceError, ValueError) as err:
        errors["kmd_cap"] = str(err)[:300]
    try:
        run.gdacs_events = gdacs.fetch_events(client, issue.date())
    except (SourceError, ValueError) as err:
        errors["gdacs"] = str(err)[:300]

    counts = alerts.SendCounts()
    if run.status != "failed":
        alerts.dispatch_flood_alerts(db, run, sms, now, counts)
    alerts.dispatch_official_alerts(db, run, sms, now, counts)

    run.sms_sent, run.sms_failed, run.sms_held_back = counts.sent, counts.failed, counts.held_back
    run.source_errors = errors
    run.finished_at = datetime.now(timezone.utc)
    db.commit()
    db.execute(delete(PipelineRun).where(PipelineRun.started_at < now - timedelta(days=RETENTION_DAYS)))
    db.commit()
    log.info("Cycle %d: %s, %d/%d zones, %d SMS sent, %d failed, errors: %s", run.id, run.status,
             run.zones_available, run.zones_total, counts.sent, counts.failed, list(errors))
    return run


def run_once() -> PipelineRun | None:
    """Run a real cycle, or return None if one is already in progress."""
    if not _cycle_lock.acquire(blocking=False):
        return None
    try:
        with session_factory()() as db, httpx.Client(headers={"User-Agent": "FloodGuard/0.1"}) as client:
            return run_cycle(db, client, build_provider())
    finally:
        _cycle_lock.release()


def latest_run(db: Session) -> PipelineRun | None:
    """The newest run with usable assessments."""
    query = select(PipelineRun).where(PipelineRun.status.in_(("ok", "partial"))).order_by(PipelineRun.id.desc())
    return db.scalar(query.limit(1))


def latest_assessments(db: Session, run: PipelineRun) -> dict[str, ZoneAssessment]:
    return {z.zone_id: z for z in db.scalars(select(ZoneAssessment).where(ZoneAssessment.run_id == run.id))}
