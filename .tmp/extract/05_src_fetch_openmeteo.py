Builds the weather point grid, fetches hourly forecasts in batches with retries, saves raw JSON and the normalised table. Commands: grid, fetch.
"""Open-Meteo ingestion: build a weather grid, fetch hourly forecasts, store raw + normalised data.

Usage:
    python src/fetch_openmeteo.py grid
    python src/fetch_openmeteo.py fetch --mode live --limit 8                 # prototype test
    python src/fetch_openmeteo.py fetch --mode live                           # full grid, current forecast
    python src/fetch_openmeteo.py fetch --mode previous_runs --start-date 2025-03-01 --end-date 2025-05-31
    python src/fetch_openmeteo.py fetch --mode historical --start-date 2025-03-01 --end-date 2025-05-31

Modes
    live           Forecast API. Issue time is NOT exposed by the API, so the retrieval time floored to the
                   hour is stored as a proxy (issue_time_source = 'retrieval_proxy'). Run it on a schedule.
    previous_runs  Previous Runs API (<var>_previous_dayN). Forecast made ~N days before the valid time, so
                   it is safe against leakage for a 24 h window (issue_time_source = 'nominal_previous_day').
    historical     Historical Forecast API (stitched archive). The run time is unknown
                   (issue_time_source = 'unknown_archive'); NOT suitable for strict 24 h evaluation.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point

from common import ensure_parent, load_settings, resolve, setup_logging

KEY_COLS = ["location_id", "valid_time_utc", "variable", "forecast_model", "source_endpoint",
            "forecast_issue_time_utc"]


# --------------------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------------------
def build_grid(cfg: dict, log) -> gpd.GeoDataFrame:
    study = gpd.read_file(resolve(cfg["paths"]["study_area"]))
    crs = cfg["project"]["working_crs"]
    poly = study.to_crs(crs).geometry.union_all()
    spacing = cfg["openmeteo"]["grid_spacing_km"] * 1000.0
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(minx + spacing / 2, maxx, spacing)
    ys = np.arange(miny + spacing / 2, maxy, spacing)
    pts = [Point(x, y) for x in xs for y in ys]
    pts = [p for p in pts if poly.contains(p)]
    if not pts:
        pts = [poly.representative_point()]
        log.warning("Grid spacing larger than study area; using a single representative point.")
    grid = gpd.GeoDataFrame(geometry=pts, crs=crs).to_crs(cfg["project"]["geographic_crs"])
    grid["location_id"] = [f"WX_{i + 1:03d}" for i in range(len(grid))]
    grid["latitude"] = grid.geometry.y.round(5)
    grid["longitude"] = grid.geometry.x.round(5)
    out = ensure_parent(resolve(cfg["paths"]["grid_points"]))
    grid.to_file(out, layer="grid", driver="GPKG")
    log.info("Wrote %d grid points (%.1f km spacing) to %s", len(grid), spacing / 1000, out)
    log.info("Note: more points do not give finer weather resolution; many may share one model cell.")
    return grid


# --------------------------------------------------------------------------------------
# API access
# --------------------------------------------------------------------------------------
def endpoint_for(mode: str, cfg: dict) -> str:
    om = cfg["openmeteo"]
    return {"live": om["forecast_url"], "historical": om["historical_forecast_url"],
            "previous_runs": om["previous_runs_url"]}[mode]


def hourly_variable_sets(mode: str, cfg: dict) -> list[list[str]]:
    """Variable lists to try, richest first. Used for graceful fallback when something is unavailable."""
    om = cfg["openmeteo"]
    full, core = list(om["hourly"]), list(om["core_hourly"])
    if mode == "previous_runs":
        days = om["previous_days"]
        f = lambda names: [f"{v}_previous_day{d}" for v in names for d in days]  # noqa: E731
        return [f(full), f(core), f(["precipitation"])]
    return [full, core, ["precipitation"]]


def request_json(url: str, params: dict, cfg: dict, log):
    om = cfg["openmeteo"]
    last_err = None
    for attempt in range(1, om["max_retries"] + 1):
        try:
            r = requests.get(url, params=params, timeout=om["timeout_seconds"])
            if r.status_code == 400:
                return r.status_code, r.json() if r.content else {}
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            return r.status_code, r.json()
        except (requests.RequestException, ValueError) as err:
            last_err = err
            wait = min(60, 2 ** attempt)
            log.warning("Request failed (attempt %d/%d): %s - retrying in %ds", attempt, om["max_retries"], err, wait)
            time.sleep(wait)
    raise RuntimeError(f"Open-Meteo request failed after retries: {last_err}")


def fetch_batch(batch: pd.DataFrame, mode: str, args, cfg: dict, log):
    """Fetch one batch of coordinates. Returns (list_of_location_dicts, hourly_variables_used)."""
    om = cfg["openmeteo"]
    url = endpoint_for(mode, cfg)
    base = {
        "latitude": ",".join(str(v) for v in batch["latitude"]),
        "longitude": ",".join(str(v) for v in batch["longitude"]),
        "timezone": "UTC",
    }
    if om.get("model"):
        base["models"] = om["model"]
    if mode == "live":
        base.update({"forecast_days": om["forecast_days"], "past_days": om["past_days"]})
    else:
        if not (args.start_date and args.end_date):
            raise SystemExit("--start-date and --end-date are required for this mode.")
        base.update({"start_date": args.start_date, "end_date": args.end_date})

    for variables in hourly_variable_sets(mode, cfg):
        params = {**base, "hourly": ",".join(variables)}
        status, payload = request_json(url, params, cfg, log)
        if status == 400:
            log.warning("API rejected variable set (%s): %s", variables[:3], payload.get("reason", payload))
            continue
        locs = payload if isinstance(payload, list) else [payload]
        return locs, variables
    raise RuntimeError("API rejected every variable set; check endpoint, dates and variable names.")


# --------------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------------
PREV_RE = re.compile(r"^(?P<base>.+)_previous_day(?P<day>\d+)$")


def normalise_response(loc_json: dict, location_id: str, req_lat: float, req_lon: float, mode: str,
                       retrieved_at: pd.Timestamp, forecast_model: str, source_endpoint: str,
                       raw_file: str) -> pd.DataFrame:
    """Turn one location's JSON into the long table defined in the plan (section 3.5)."""
    hourly = loc_json.get("hourly")
    if not hourly or "time" not in hourly:
        return pd.DataFrame()
    units = loc_json.get("hourly_units", {})
    times = pd.to_datetime(hourly["time"], utc=True)  # request used timezone=UTC
    frames = []
    for name, values in hourly.items():
        if name == "time":
            continue
        base, day = name, None
        m = PREV_RE.match(name)
        if m:
            base, day = m.group("base"), int(m.group("day"))
        df = pd.DataFrame({"valid_time_utc": times, "value": pd.to_numeric(pd.Series(values), errors="coerce")})
        df = df.dropna(subset=["value"])
        df["variable"] = base
        df["unit"] = units.get(name)
        if mode == "previous_runs":
            df["forecast_issue_time_utc"] = df["valid_time_utc"] - pd.Timedelta(days=day)
            df["issue_time_source"] = "nominal_previous_day"
        elif mode == "live":
            df["forecast_issue_time_utc"] = retrieved_at.floor("h")
            df["issue_time_source"] = "retrieval_proxy"
        else:
            df["forecast_issue_time_utc"] = pd.NaT
            df["issue_time_source"] = "unknown_archive"
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["location_id"] = location_id
    out["latitude"] = req_lat
    out["longitude"] = req_lon
    out["returned_latitude"] = loc_json.get("latitude")
    out["returned_longitude"] = loc_json.get("longitude")
    out["retrieved_at_utc"] = retrieved_at
    out["forecast_model"] = forecast_model
    out["source_endpoint"] = source_endpoint
    out["raw_file"] = raw_file
    out["lead_hours"] = (out["valid_time_utc"] - out["forecast_issue_time_utc"]).dt.total_seconds() / 3600.0
    cols = ["location_id", "latitude", "longitude", "returned_latitude", "returned_longitude", "retrieved_at_utc",
            "forecast_issue_time_utc", "issue_time_source", "valid_time_utc", "forecast_model", "variable", "value",
            "unit", "lead_hours", "source_endpoint", "raw_file"]
    return out[cols]


def append_weather(new: pd.DataFrame, path, log) -> pd.DataFrame:
    ensure_parent(path)
    if path.exists():
        old = pd.read_parquet(path)
        new = pd.concat([old, new], ignore_index=True)
    # NaT in issue time must still dedupe correctly, so use a filled helper key column
    new["_issue_key"] = new["forecast_issue_time_utc"].fillna(pd.Timestamp("1970-01-01", tz="UTC"))
    key = [c if c != "forecast_issue_time_utc" else "_issue_key" for c in KEY_COLS]
    before = len(new)
    new = new.drop_duplicates(subset=key, keep="last").drop(columns="_issue_key").reset_index(drop=True)
    new.to_parquet(path, index=False)
    log.info("weather_hourly.parquet: %d rows (%d duplicates removed)", len(new), before - len(new))
    return new


def run_fetch(args, cfg: dict, log) -> None:
    grid_path = resolve(cfg["paths"]["grid_points"])
    if not grid_path.exists():
        raise SystemExit("Grid not found - run 'python src/fetch_openmeteo.py grid' first.")
    grid = gpd.read_file(grid_path)[["location_id", "latitude", "longitude"]]
    if args.limit:
        grid = grid.head(args.limit)
    om = cfg["openmeteo"]
    raw_dir = resolve(cfg["paths"]["raw_openmeteo"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    source_endpoint = endpoint_for(args.mode, cfg)
    model = om.get("model") or "best_match"
    frames, n_batches = [], int(np.ceil(len(grid) / om["batch_size"]))

    for b in range(n_batches):
        batch = grid.iloc[b * om["batch_size"]:(b + 1) * om["batch_size"]].reset_index(drop=True)
        locs, used = fetch_batch(batch, args.mode, args, cfg, log)
        retrieved_at = pd.Timestamp(datetime.now(timezone.utc))
        raw_file = raw_dir / f"{args.mode}_{stamp}_batch{b + 1:03d}.json"
        with open(raw_file, "w", encoding="utf-8") as fh:
            json.dump({"retrieved_at_utc": retrieved_at.isoformat(), "mode": args.mode, "endpoint": source_endpoint,
                       "variables": used, "location_ids": batch["location_id"].tolist(), "response": locs}, fh)
        if len(locs) != len(batch):
            raise RuntimeError(f"Expected {len(batch)} locations, API returned {len(locs)}.")
        for row, loc in zip(batch.itertuples(index=False), locs):
            df = normalise_response(loc, row.location_id, row.latitude, row.longitude, args.mode, retrieved_at,
                                    model, source_endpoint, str(raw_file.relative_to(resolve("."))))
            if df.empty:
                log.warning("No hourly data returned for %s", row.location_id)
            else:
                frames.append(df)
        log.info("Batch %d/%d done (%d locations, %d variables)", b + 1, n_batches, len(batch), len(used))
        time.sleep(om["sleep_seconds"])

    if not frames:
        raise RuntimeError("No weather data collected.")
    new = pd.concat(frames, ignore_index=True)
    # Gate G1 quick checks
    log.info("Variables collected: %s", sorted(new["variable"].unique()))
    log.info("Units: %s", new.groupby("variable")["unit"].first().to_dict())
    log.info("Valid time range (UTC): %s -> %s", new["valid_time_utc"].min(), new["valid_time_utc"].max())
    miss = [v for v in om["hourly"] if v not in set(new["variable"])]
    if miss:
        log.warning("Variables not returned (optional features will be NaN): %s", miss)
    append_weather(new, resolve(cfg["paths"]["processed"]) / "weather_hourly.parquet", log)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["grid", "fetch"])
    ap.add_argument("--mode", choices=["live", "previous_runs", "historical"], default="live")
    ap.add_argument("--start-date", default=None)
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only first N grid points (prototype test)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_settings(args.config)
    log = setup_logging("fetch_openmeteo", cfg)
    if args.command == "grid":
        build_grid(cfg, log)
    else:
        run_fetch(args, cfg, log)


if __name__ == "__main__":
    main()
