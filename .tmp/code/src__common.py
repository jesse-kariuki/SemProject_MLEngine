"""Shared helpers: settings, paths, logging, time utilities."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pandas as pd
import shapely
import yaml

PROJECT_ROOT = Path(os.environ.get("FLOOD_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "settings.yaml"


def load_settings(path: str | os.PathLike | None = None) -> dict:
    """Load settings.yaml (path defaults to config/settings.yaml)."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(rel_path: str | os.PathLike) -> Path:
    """Resolve a path from settings relative to the project root."""
    p = Path(rel_path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(name: str, cfg: dict | None = None) -> logging.Logger:
    """Console + file logger writing to logs/<name>.log."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    log_dir = resolve((cfg or {}).get("paths", {}).get("logs", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def to_display_tz(series: pd.Series, cfg: dict) -> pd.Series:
    """Convert a UTC datetime series to the display timezone (EAT)."""
    return series.dt.tz_convert(cfg["project"]["display_timezone"])


def utc(value) -> pd.Timestamp:
    """Parse any timestamp-like value to a tz-aware UTC Timestamp."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def boundary_sanity(geom, cfg: dict, log) -> dict:
    """Log area, complexity and rectangle-likeness of the study boundary (geom in the working CRS, metres).

    A rectangle, or an area much larger than the real county, makes every coverage figure misleading.
    """
    area_km2 = geom.area / 1e6
    env_km2 = geom.envelope.area / 1e6
    fill = area_km2 / env_km2 if env_km2 else float("nan")
    n_vertices = int(shapely.get_num_coordinates(geom))
    expected = cfg.get("project", {}).get("expected_county_area_km2")
    log.info("Boundary: area %.1f km2, %d vertices, fills %.1f%% of its bounding rectangle", area_km2, n_vertices, 100 * fill)
    if fill > 0.97 or n_vertices < 20:
        log.warning("BOUNDARY CHECK: the boundary looks like a rectangle/bounding box (fill %.2f, %d vertices). "
                    "Use the real county polygon.", fill, n_vertices)
    if expected and area_km2 > 1.15 * expected:
        log.warning("BOUNDARY CHECK: boundary area %.0f km2 is %.0f%% above the expected county area (~%.0f km2). "
                    "It probably includes land outside the county; check inputs.boundary_file.",
                    area_km2, 100 * (area_km2 / expected - 1), expected)
    return {"boundary_area_km2": area_km2, "boundary_vertices": n_vertices, "boundary_bbox_fill": fill}
