"""Export the trained model's output into the small assets the API serves at runtime.

The API never loads models/H_rf.joblib (418 MB): H_rf scores depend only on terrain and land cover, so
they do not change with the weather and can be computed once. Rerun this after retraining.

Run with the pipeline environment (it needs pandas and pyarrow), from the repo root:
    .venv/Scripts/python backend/scripts/export_assets.py
    .venv/Scripts/python backend/scripts/export_assets.py --skip-river   # keep existing river thresholds
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "backend" / "assets"
FLOOD_URL = "https://flood-api.open-meteo.com/v1/flood"
RIVER_START, RIVER_END = "2003-01-01", "2022-07-31"  # GloFAS v4 reanalysis ends July 2022


def river_thresholds(grid: pd.DataFrame) -> dict[str, dict]:
    """95th/99th percentile of daily GloFAS discharge per zone, so 'high' means high for that river."""
    out: dict[str, dict] = {}
    for start in range(0, len(grid), 5):
        batch = grid.iloc[start:start + 5]
        for attempt in range(5):
            r = requests.get(FLOOD_URL, timeout=120, params={
                "latitude": ",".join(map(str, batch["latitude"])),
                "longitude": ",".join(map(str, batch["longitude"])),
                "daily": "river_discharge", "start_date": RIVER_START, "end_date": RIVER_END})
            if r.status_code != 429:
                break
            time.sleep(30 * (attempt + 1))
        r.raise_for_status()
        payload = r.json()
        for row, loc in zip(batch.itertuples(), payload if isinstance(payload, list) else [payload]):
            q = np.array([v for v in loc["daily"]["river_discharge"] if v is not None], dtype=float)
            out[row.location_id] = {"river_p95": round(float(np.percentile(q, 95)), 3),
                                    "river_p99": round(float(np.percentile(q, 99)), 3)} if len(q) else {}
        time.sleep(3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-river", action="store_true")
    args = ap.parse_args()
    ASSETS.mkdir(exist_ok=True)

    pred = pd.read_csv(ROOT / "outputs" / "predictions_H_rf_static.csv")
    feats = pd.read_parquet(ROOT / "data" / "processed" / "static_segment_features.parquet")[
        ["segment_id", "centroid_lat", "centroid_lon", "segment_length_m"]]
    wmap = pd.read_parquet(ROOT / "data" / "processed" / "segment_weather_map.parquet")[
        ["segment_id", "weather_location_id"]]
    seg = pred.merge(feats, on="segment_id", validate="1:1").merge(wmap, on="segment_id", validate="1:1")
    seg = seg.rename(columns={"susceptibility_probability": "p", "susceptibility_tier": "tier",
                              "centroid_lat": "lat", "centroid_lon": "lon", "weather_location_id": "zone_id"})
    seg = seg[["segment_id", "road_name", "road_class", "lat", "lon", "p", "tier", "zone_id"]]
    seg["lat"], seg["lon"] = seg["lat"].round(5), seg["lon"].round(5)
    with gzip.open(ASSETS / "segments.csv.gz", "wt", encoding="utf-8", newline="") as fh:
        seg.to_csv(fh, index=False)

    import geopandas as gpd
    grid = gpd.read_file(ROOT / "config" / "grid_points.gpkg")[["location_id", "latitude", "longitude"]]
    grid = grid[grid["location_id"].isin(seg["zone_id"].unique())].reset_index(drop=True)
    zones_path = ASSETS / "zones.json"
    previous = {z["zone_id"]: z for z in json.loads(zones_path.read_text())["zones"]} if zones_path.exists() else {}
    thresholds = {k: {f: v.get(f) for f in ("river_p95", "river_p99")} for k, v in previous.items()}
    if not args.skip_river:
        thresholds = river_thresholds(grid)
    meta = json.loads((ROOT / "models" / "H_rf.json").read_text(encoding="utf-8"))
    zones = [{"zone_id": r.location_id, "lat": r.latitude, "lon": r.longitude,
              "segments": int((seg["zone_id"] == r.location_id).sum()), **thresholds.get(r.location_id, {})}
             for r in grid.itertuples()]
    zones_path.write_text(json.dumps({"model_version": meta["model_version"], "model_trained_on": meta["trained_on"],
                                      "river_baseline": [RIVER_START, RIVER_END], "zones": zones}, indent=1))
    print(f"segments: {len(seg)}  zones: {len(zones)}  tiers: {seg['tier'].value_counts().to_dict()}")
    print(f"zones with river thresholds: {sum(1 for z in zones if z.get('river_p99') is not None)}")


if __name__ == "__main__":
    main()
