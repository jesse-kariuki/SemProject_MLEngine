"""Read-only endpoints for the mobile app and public use."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.alerts import active_official_alerts
from app.config import get_settings
from app.db import get_db
from app.geo import get_store
from app.models import PipelineRun, ZoneAssessment
from app.pipeline import latest_assessments, latest_run
from app.risk import BAND_NAMES, LEVEL_NAMES, MATRIX, local_level

router = APIRouter(prefix="/api/v1", tags=["public"])

DISCLAIMER = ("Decision support only, not an official warning. Road susceptibility comes from a model trained on "
              "simulated flooding, and alert levels are rule-based and not yet validated against road-level "
              "observations. Follow Kenya Met and county instructions, and never cross moving water.")


def data_status(run: PipelineRun | None) -> dict:
    """How fresh the data is. Clients must not read a stale or missing level as 'safe'."""
    if run is None:
        return {"state": "no_data", "issue_time": None, "age_minutes": None}
    age = datetime.now(timezone.utc) - run.started_at
    stale = age > timedelta(minutes=get_settings().max_data_age_minutes)
    return {"state": "stale" if stale else "fresh", "issue_time": run.issue_time,
            "age_minutes": round(age.total_seconds() / 60, 1), "run_status": run.status}


def _zone_json(z: ZoneAssessment) -> dict:
    store = get_store()
    zone = store.zones.get(z.zone_id)
    return {"zone_id": z.zone_id, "latitude": zone.lat if zone else None, "longitude": zone.lon if zone else None,
            "available": z.available, "rain_band": z.rain_band, "rain_band_name": BAND_NAMES[z.rain_band],
            "rain_next_6h_mm": z.rain_next_6h_mm, "rain_next_24h_mm": z.rain_next_24h_mm,
            "rain_area_24h_mm": z.rain_area_24h_mm,
            "rain_prev_72h_mm": z.rain_prev_72h_mm, "max_hourly_mm": z.max_hourly_mm,
            "precip_probability_max": z.precip_probability_max, "river_discharge_m3s": z.river_discharge_m3s,
            "river_p99_m3s": z.river_p99_m3s, "reasons": z.reasons}


@router.get("/status", summary="Data freshness, sources and model provenance")
def status(db: Session = Depends(get_db)) -> dict:
    last = db.scalar(select(PipelineRun).order_by(PipelineRun.id.desc()).limit(1))
    settings = get_settings()
    return {"data": data_status(latest_run(db)),
            "last_cycle": None if last is None else {
                "status": last.status, "started_at": last.started_at, "zones_available": last.zones_available,
                "zones_total": last.zones_total, "sms_sent": last.sms_sent, "sms_failed": last.sms_failed,
                "source_errors": last.source_errors},
            "sms_provider": settings.sms_provider, "susceptibility_model": get_store().model_version,
            "coverage": "Nairobi County road network", "disclaimer": DISCLAIMER}


@router.get("/zones", summary="Current assessment for every weather zone")
def zones(db: Session = Depends(get_db)) -> dict:
    run = latest_run(db)
    rows = [] if run is None else [_zone_json(z) for z in latest_assessments(db, run).values()]
    return {"data": data_status(run), "zones": rows, "disclaimer": DISCLAIMER}


@router.get("/risk", summary="Flood risk at a location")
def risk(latitude: float = Query(ge=-90, le=90), longitude: float = Query(ge=-180, le=180),
         radius_m: int = Query(default=None, ge=50, le=2000), db: Session = Depends(get_db)) -> dict:
    radius = radius_m or get_settings().default_radius_m
    place = get_store().locate(latitude, longitude, radius)
    if place is None:
        raise HTTPException(422, "This location is outside FloodGuard coverage (Nairobi County roads).")
    run = latest_run(db)
    status_info = data_status(run)
    zone = latest_assessments(db, run).get(place.zone_id) if run else None
    usable = zone is not None and zone.available and status_info["state"] == "fresh"
    level = local_level(zone.rain_band, place.tier) if usable else None
    return {"data": status_info, "zone_id": place.zone_id, "level": level,
            "level_name": LEVEL_NAMES[level] if usable else "unknown",
            "local_susceptibility": place.susceptibility, "local_susceptibility_tier": place.tier,
            "distance_to_road_m": place.distance_to_road_m, "nearby_roads": place.roads,
            "weather": _zone_json(zone) if zone else None, "disclaimer": DISCLAIMER}


@router.get("/roads/search", summary="Find a named road, for example to pick a subscription location")
def road_search(q: str = Query(min_length=2, max_length=60), limit: int = Query(default=10, ge=1, le=50)) -> dict:
    return {"results": get_store().search_roads(q, limit)}


@router.get("/roads/at-risk", summary="Named roads at or above an alert level right now")
def roads_at_risk(min_level: int = Query(default=2, ge=1, le=3), limit: int = Query(default=50, ge=1, le=200),
                  db: Session = Depends(get_db)) -> dict:
    run = latest_run(db)
    status_info = data_status(run)
    if run is None or status_info["state"] != "fresh":
        return {"data": status_info, "roads": []}
    levels = {zone_id: {tier: row[z.rain_band] for tier, row in MATRIX.items()}
              for zone_id, z in latest_assessments(db, run).items() if z.available}
    return {"data": status_info, "roads": get_store().roads_at_level(levels, min_level, limit)}


@router.get("/alerts/official", summary="Active Kenya Met alerts, and recent GDACS flood events in Kenya")
def official_alerts(nairobi_only: bool = False, db: Session = Depends(get_db)) -> dict:
    rows = active_official_alerts(db, datetime.now(timezone.utc))
    if nairobi_only:
        rows = [a for a in rows if a.zones_covered]
    last = db.scalar(select(PipelineRun).order_by(PipelineRun.id.desc()).limit(1))
    return {"kmd_alerts": [{"identifier": a.identifier, "event": a.event, "headline": a.headline,
                            "description": a.description, "instruction": a.instruction, "severity": a.severity,
                            "area": a.area, "sent_at": a.sent_at, "expires_at": a.expires_at, "url": a.url,
                            "covers_nairobi": bool(a.zones_covered)} for a in rows],
            "gdacs_events": last.gdacs_events if last else []}
