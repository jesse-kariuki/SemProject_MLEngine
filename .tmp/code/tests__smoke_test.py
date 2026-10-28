"""End-to-end smoke test on SYNTHETIC data (no internet, no real Nairobi data needed; includes a TAB-separated flood-hazard file).

Run:   python tests/smoke_test.py          or          pytest -q tests/smoke_test.py
It copies src/ and config/ into a temporary project, fabricates a boundary, roads, rivers, DEM, land cover,
weather responses and a flood event inventory, then runs every pipeline step and checks the outputs.
The numbers it produces are meaningless - it only proves the code runs and the plumbing is correct.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.transform import from_origin
from shapely.geometry import LineString, Polygon, box

REPO = Path(__file__).resolve().parents[1]
W, S, E, N = 36.78, -1.32, 36.86, -1.25            # synthetic study area (lon/lat)


def run(root: Path, *cmd: str) -> None:
    env = {**os.environ, "FLOOD_PROJECT_ROOT": str(root)}
    res = subprocess.run([sys.executable, *cmd], cwd=root, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        raise AssertionError(f"Command failed: {' '.join(cmd)}\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}")


def make_inputs(root: Path) -> None:
    rng = np.random.default_rng(1)
    raw = root / "data" / "raw"
    for d in ("boundary", "osm_shapefiles", "dem", "landcover"):
        (raw / d).mkdir(parents=True, exist_ok=True)

    gpd.GeoDataFrame({"name": ["Nairobi"]}, geometry=[box(W, S, E, N)], crs="EPSG:4326").to_file(
        raw / "boundary" / "nairobi_boundary.gpkg", driver="GPKG")

    lines, classes = [], []
    for lon in np.arange(W - 0.01, E + 0.01, 0.004):
        lines.append(LineString([(lon, S - 0.01), (lon, N + 0.01)]))
    for lat in np.arange(S - 0.01, N + 0.01, 0.004):
        lines.append(LineString([(W - 0.01, lat), (E + 0.01, lat)]))
    classes = rng.choice(["residential", "tertiary", "primary", "footway"], size=len(lines), p=[.5, .25, .15, .1])
    roads = gpd.GeoDataFrame({"osm_id": [str(1000 + i) for i in range(len(lines))], "fclass": classes,
                              "name": [f"Road {i}" for i in range(len(lines))], "bridge": "F", "tunnel": "F"},
                             geometry=lines, crs="EPSG:4326")
    roads.to_file(raw / "osm_shapefiles" / "gis_osm_roads_free_1.shp")
    gpd.GeoDataFrame({"fclass": ["river"]}, geometry=[LineString([(36.78, -1.25), (36.86, -1.32)])],
                     crs="EPSG:4326").to_file(raw / "osm_shapefiles" / "gis_osm_waterways_free_1.shp")

    west, north, south = 36.76, -1.23, -1.34
    res = 0.0003
    cols, rows = int((0.12) / res), int((north - south) / res)
    yy, xx = np.mgrid[0:rows, 0:cols]
    dem = (1700 + 0.4 * yy + 15 * np.sin(xx / 25.0) + rng.normal(0, 1, (rows, cols))).astype("float32")
    with rasterio.open(raw / "dem" / "dem.tif", "w", driver="GTiff", height=rows, width=cols, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(west, north, res, res), nodata=-9999) as dst:
        dst.write(dem, 1)

    res = 0.0001
    cols, rows = int(0.12 / res), int((north - south) / res)
    block = 100
    codes = rng.choice([10, 30, 40, 50, 50, 50, 80], size=(rows // block + 1, cols // block + 1))
    lc = np.kron(codes, np.ones((block, block), dtype="uint8"))[:rows, :cols].astype("uint8")
    with rasterio.open(raw / "landcover" / "landcover.tif", "w", driver="GTiff", height=rows, width=cols, count=1,
                       dtype="uint8", crs="EPSG:4326", transform=from_origin(west, north, res, res), nodata=0) as dst:
        dst.write(lc, 1)

    # synthetic flood-hazard cells in the same layout as the real dataset (TAB separated); hazard follows low DEM
    (raw / "flood_hazard").mkdir(parents=True, exist_ok=True)
    dres, dwest, dnorth = 0.0003, 0.0 + 36.76, -1.23
    rows_h = []
    for lat in np.arange(S + 0.00045, N, 0.0009):
        for lon in np.arange(W + 0.0005, E, 0.001):
            if lon < W + 0.012:      # leave the western strip without hazard cells (partial coverage like the real file)
                continue
            r_i, c_i = int((dnorth - lat) / dres), int((lon - dwest) / dres)
            elev = float(dem[min(r_i, dem.shape[0] - 1), min(c_i, dem.shape[1] - 1)])
            z = -(elev - 1745.0) / 8.0
            rows_h.append({"latitude": round(lat, 8), "longitude": round(lon, 8), "lat_min": round(lat - 0.00045, 8),
                           "lat_max": round(lat + 0.00045, 8), "lon_min": round(lon - 0.0005 - (0.000004 if len(rows_h) % 2 else 0), 8),
                           "lon_max": round(lon + 0.0005, 8), "flood_hazard": int(rng.random() < 1 / (1 + np.exp(-z))) + int(rng.random() < 0.4 / (1 + np.exp(-z)))})
    pd.DataFrame(rows_h).to_csv(raw / "flood_hazard" / "flood_hazard.csv", sep="\t", index=False)


def make_weather_and_events(root: Path) -> None:
    sys.path.insert(0, str(root / "src"))
    os.environ["FLOOD_PROJECT_ROOT"] = str(root)
    import fetch_openmeteo as fo
    from common import load_settings
    cfg = load_settings()
    rng = np.random.default_rng(7)
    grid = gpd.read_file(root / cfg["paths"]["grid_points"])
    static = pd.read_parquet(root / "data" / "processed" / "static_segment_features.parquet")

    starts = pd.date_range("2026-03-05 14:00", periods=10, freq="9D")       # local event starts (EAT)
    event_rain = rng.uniform(5, 70, len(starts))
    times = pd.date_range("2026-03-01", "2026-06-10", freq="h", tz="UTC")
    frames = []
    retrieved = pd.Timestamp("2026-06-11", tz="UTC")
    for row in grid.itertuples():
        precip = rng.exponential(0.05, len(times))
        for st, rain in zip(starts, event_rain):
            issue = (st.tz_localize("Africa/Nairobi") - pd.Timedelta(hours=6)).tz_convert("UTC").floor("h")
            heavy = (times >= issue) & (times < issue + pd.Timedelta(hours=6))
            precip[heavy] += rain / 6.0
        loc_json = {"latitude": row.latitude, "longitude": row.longitude,
                    "hourly_units": {"precipitation_previous_day1": "mm", "temperature_2m_previous_day1": "C"},
                    "hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
                               "precipitation_previous_day1": precip.round(3).tolist(),
                               "temperature_2m_previous_day1": (18 + rng.normal(0, 1, len(times))).round(2).tolist()}}
        frames.append(fo.normalise_response(loc_json, row.location_id, row.latitude, row.longitude, "previous_runs",
                                            retrieved, "best_match", "synthetic", "synthetic.json"))
    weather = pd.concat(frames, ignore_index=True)
    assert set(weather["variable"]) == {"precipitation", "temperature_2m"}
    assert (weather["lead_hours"] == 24).all()
    import logging
    fo.append_weather(weather, root / "data" / "processed" / "weather_hourly.parquet", logging.getLogger("smoke"))

    # synthetic events: flooding likelier on low ground in heavy-rain events
    med_elev = static["elevation_m"].median()
    recs = []
    for i, (st, rain) in enumerate(zip(starts, event_rain)):
        pick = static.sample(40, random_state=i)
        for j, seg in enumerate(pick.itertuples()):
            z = -(seg.elevation_m - med_elev) / 15.0 + (rain - 35.0) / 12.0
            flooded = rng.random() < 1 / (1 + np.exp(-z))
            recs.append({"event_id": f"EVT_{i + 1:02d}", "observation_id": f"OBS_{i + 1:02d}_{j + 1:03d}",
                         "observed_status": "flooded" if flooded else "not_flooded",
                         "lat": seg.centroid_lat, "lon": seg.centroid_lon,
                         "event_start_local": st.strftime("%Y-%m-%d %H:%M"), "event_end_local": "",
                         "date_only": "false", "spatial_precision_m": 20, "evidence_type": "synthetic",
                         "source_id": "SYN", "source_url": "", "published_time_local": "",
                         "confidence": "high", "review_status": "reviewed" if j % 10 else "pending", "notes": ""})
    pd.DataFrame(recs).to_csv(root / "data" / "interim" / "event_inventory.csv", index=False)


def run_smoke(root: Path) -> None:
    shutil.copytree(REPO / "src", root / "src")
    shutil.copytree(REPO / "config", root / "config")
    cfg = yaml.safe_load(open(root / "config" / "settings.yaml"))
    cfg["openmeteo"]["grid_spacing_km"] = 2
    cfg["labels"]["min_events_for_modelling"] = 5
    yaml.safe_dump(cfg, open(root / "config" / "settings.yaml", "w"))

    make_inputs(root)
    run(root, "src/prepare_gis.py", "all")
    run(root, "src/fetch_openmeteo.py", "grid")
    run(root, "src/build_features.py", "static")
    run(root, "src/build_hazard_labels.py")
    run(root, "src/check_coverage.py")
    run(root, "src/train_static.py")
    run(root, "src/evaluate.py", "--static")
    make_weather_and_events(root)
    run(root, "src/build_labels.py", "labels")
    run(root, "src/build_features.py", "forecast", "--from-labels")
    run(root, "src/build_labels.py", "assemble")
    run(root, "src/train.py")
    run(root, "src/evaluate.py")
    run(root, "src/predict.py", "--model", "B_rf", "--issue-time", "2026-04-10T05:00:00Z")
    run(root, "src/predict.py")                                             # static map, default H_rf
    run(root, "src/predict.py", "--issue-time", "2026-04-10T05:00:00Z")      # static + rain context

    proc = root / "data" / "processed"
    seg = gpd.read_file(root / "data" / "interim" / "road_segments.gpkg")
    assert len(seg) > 500 and seg["segment_id"].is_unique
    assert "footway" not in set(seg["fclass"]), "fclass filter failed"
    static = pd.read_parquet(proc / "static_segment_features.parquet")
    assert static["elevation_m"].notna().mean() > 0.95
    assert static["builtup_fraction"].between(0, 1).all()
    table = pd.read_parquet(proc / "modelling_table.parquet")
    assert table["flood_label"].isin([0, 1]).all() and table["flood_label"].nunique() == 2
    assert (table["max_issue_time_used_utc"] <= table["forecast_issue_time_utc"]).all()
    assert (table["data_quality_flag"] == "OK").mean() > 0.9
    metrics = pd.read_csv(root / "outputs" / "metrics" / "metrics.csv")
    assert {"rule_rain24h", "rule_fastflood", "A_rf", "B_rf", "B_rf_haz"} <= set(metrics["model"])
    assert {"max_hazard_value", "hazard_pos_fraction"} <= set(table.columns)
    hl = pd.read_parquet(proc / "hazard_labels.parquet")
    assert {"positive", "negative"} <= set(hl["label_status"]), "hazard labels need both classes"
    assert hl.loc[hl["label_status"] == "unknown", "hazard_label"].isna().all(), "Unknown must stay unlabelled"
    cov = pd.read_csv(root / "outputs" / "tables" / "coverage_report.csv", index_col=0)["value"]
    assert cov["segments_outside_footprint"] > 0 and cov["boundary_without_cells_km2"] > 0
    assert abs(cov["labels_minus_diagnostic"]) <= 0.005 * cov["segments_total"], "label script vs footprint test disagree"
    assert (hl["label_source"].isin(["fastflood_original", "none"])).all() and hl["susceptibility_score"].between(0, 1).sum() > 0
    ms = pd.read_csv(root / "outputs" / "metrics" / "static" / "metrics.csv")
    assert {"rule_low_elevation", "H_logreg", "H_rf"} <= set(ms["model"])
    h = ms.set_index("model")
    assert h.loc["H_rf", "pr_auc"] > h.loc["H_rf", "prevalence"], "static model should beat prevalence on held-out blocks"
    out = root / "outputs"
    p1 = pd.read_csv(out / "predictions_B_rf_20260410T0500Z.csv")
    assert p1["flood_probability"].between(0, 1).sum() > 0 and "risk_tier" in p1.columns
    p2 = pd.read_csv(out / "predictions_H_rf_static.csv")
    assert p2["susceptibility_probability"].between(0, 1).sum() > 0
    p3 = pd.read_csv(out / "predictions_H_rf_20260410T0500Z.csv")
    assert set(p2["susceptibility_source"]) == {"fastflood_covered", "model_predicted_no_fastflood"}
    assert {"rain_above_threshold", "susceptibility_tier"} <= set(p3.columns) and "rain_escalated_tier_heuristic" not in p3.columns
    print(ms[["model", "n_rows", "n_positive", "prevalence", "precision", "recall", "pr_auc"]].round(3).to_string(index=False))
    print("SMOKE TEST PASSED")
    print(metrics[["model", "n_rows", "n_positive", "precision", "recall", "pr_auc", "brier"]].round(3).to_string(index=False))


def test_pipeline_end_to_end(tmp_path):
    run_smoke(tmp_path / "proj")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        run_smoke(Path(d) / "proj")
