"""FloodGuard API: live flood alerts for Nairobi roads."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.api import admin, public, subscribers
from app.config import get_settings
from app.db import get_db, init_db
from app.geo import get_store
from app.pipeline import latest_run, run_once
from app.sms import build_provider

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("floodguard")


async def _monitor_loop(interval_minutes: int) -> None:
    while True:
        try:
            await asyncio.to_thread(run_once)
        except Exception:               # one bad cycle must not end monitoring
            log.exception("Monitoring cycle crashed")
        await asyncio.sleep(interval_minutes * 60)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    build_provider(settings)            # misconfigured SMS should stop the deploy, not surface mid-flood
    init_db()
    get_store()
    task = None
    if settings.scheduler_enabled:
        task = asyncio.create_task(_monitor_loop(settings.pipeline_interval_minutes))
    yield
    if task:
        task.cancel()


app = FastAPI(title="FloodGuard API", version="0.1.0", lifespan=lifespan,
              description="Live flood risk and SMS alerts for Nairobi roads.")
app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_origins.split(","),
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(public.router)
app.include_router(subscribers.router)
app.include_router(admin.router)


@app.get("/health/live", tags=["health"], summary="Is the process up")
def live() -> dict:
    return {"status": "up"}


@app.get("/health", tags=["health"], summary="Is the data fresh enough to trust")
def health(db: Session = Depends(get_db)) -> dict:
    data = public.data_status(latest_run(db))
    return {"status": "ok" if data["state"] == "fresh" else "degraded", "data": data}
