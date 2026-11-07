"""Runtime settings, read from the environment or backend/.env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    database_url: str = "sqlite:///./floodguard.db"
    assets_dir: Path = BACKEND_DIR / "assets"
    cors_origins: str = "*"

    scheduler_enabled: bool = True
    pipeline_interval_minutes: int = 60
    max_data_age_minutes: int = 180     # older assessments are reported as stale, never as "no risk"

    admin_api_key: str = ""
    inbound_webhook_secret: str = ""

    sms_provider: str = "console"       # console | thrivewave
    thrivewave_api_url: str = ""
    thrivewave_api_key: str = ""
    thrivewave_sender_id: str = ""
    max_sms_per_cycle: int = 500
    sms_opt_out_hint: str = "Reply STOP to opt out"

    forecast_url: str = "https://api.open-meteo.com/v1/forecast"
    flood_url: str = "https://flood-api.open-meteo.com/v1/flood"
    kmd_cap_rss_url: str = "https://meteo.go.ke/api/cap/rss.xml"
    gdacs_url: str = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"
    http_timeout_seconds: float = 30.0
    http_retries: int = 3

    otp_ttl_minutes: int = 10
    otp_resend_seconds: int = 60
    otp_max_attempts: int = 5
    alert_cooldown_hours: int = 24      # repeat an unchanged alert at most this often
    default_radius_m: int = 300
    coverage_radius_m: int = 2000       # further than this from any mapped road = outside coverage


@lru_cache
def get_settings() -> Settings:
    return Settings()
