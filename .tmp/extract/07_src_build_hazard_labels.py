PRIMARY. Turns the FastFlood hazard cells (levels 0-2) into Positive / Negative / Unknown road-segment labels, a continuous susceptibility_score and exposure columns, with overlap and coverage checks, label provenance and spatial blocks (gate G4h).
"""Primary label source: turn the flood-hazard cell dataset into road-segment labels.

Input columns (names configurable in settings.yaml -> hazard.columns):
    latitude, longitude, lat_min, lat_max, lon_min, lon_max, flood_hazard
Each row is one 100 x 100 m cell with a modelled flood-hazard category (flood depth). The data are FastFlood
MODEL OUTPUT (SPACE4ALL; Copernicus DEM, ESA WorldCover, SoilGrids, TAHMO rainfall; no drainage network, no
calibration), not observed flooding, and the file has NO date or time. The labels therefore describe modelled
susceptibility, and they are also exposure features (max_hazard_value, hazard_pos_fraction) for every segment
that can be tested against observed events.

Usage:
    python src/build_hazard_labels.py

Rules (all thresholds in settings.yaml):
    - cells are overlaid on the road segments and the length of each segment inside positive / negative cells
      is measured;
    - Unknown: segment not covered by mapped cells (covered share < min_coverage), conflicting cell values, or
      0 treated as "no information" (hazard.treat_zero_as_negative: false). Unknown is never turned into 0;
    - Positive: positive share of covered length >= positive_fraction_min;
    - Negative: positive share == 0 (when zero is a confirmed negative);
    - Mixed: in between, handled by mixed_policy (default: excluded).
Outputs: data/processed/hazard_labels.parquet, data/interim/hazard_cells.gpkg (for QGIS checks),
         outputs/tables/hazard_label_summary.csv
"""
from __future__ import annotations

import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box

from common import ensure_parent, load_settings, resolve, setup_logging


def read_table(path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    with open(path, "r", encoding="utf-8-sig") as fh:
        header = fh.readline()
    sep = "\t" if header.count("\t") >= header.count(",") and "\t" in header else ","
    return pd.read_csv(path, sep=sep, encoding="utf-8-sig")


def overlap_report(cells: gpd.GeoDataFrame, log, tol_m2: float = 1.0) -> None:
    """Measure real overlap between cells. Coordinate rounding makes neighbours overlap by centimetres, which is harmless."""
    a, b = cells.sindex.query(cells.geometry.values, predicate="intersects")
    keep = a < b
    a, b = a[keep], b[keep]
    if len(a) == 0:
        log.info("Cell overlap: none.")
        return
    ov = shapely.area(shapely.intersection(cells.geometry.values[a], cells.geometry.values[b]))
    big = ov > tol_m2
    total = float(cells.geometry.area.sum())
    log.info("Cell overlap: %d touching pairs, %d overlap by more than %.0f m2; overlapping area %.1f m2 (%.5f%% of all cell area).",
             len(a), int(big.sum()), tol_m2, float(ov.sum()), 100 * float(ov.sum()) / total)
    if big.sum() > 0.001 * len(cells):
        log.warning("Substantial cell overlap: lengths may be double counted. Inspect hazard_cells.gpkg.")
    else:
        log.info("Overlaps are negligible (typically coordinate rounding); no action needed.")


def load_cells(cfg: dict, log, path=None, clip: bool = True) -> gpd.GeoDataFrame:
    hz = cfg["hazard"]
    path = path or resolve(cfg["inputs"]["hazard_dataset"])
    if not path.exists():
        raise FileNotFoundError(f"Hazard dataset not found: {path}")
    col = hz["columns"]
    raw = read_table(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    need = [col["latitude"], col["longitude"], col["value"]]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise KeyError(f"Hazard dataset is missing columns {missing}; found {list(raw.columns)}. "
                       "Adjust hazard.columns in settings.yaml.")
    df = pd.DataFrame({"latitude": pd.to_numeric(raw[col["latitude"]], errors="coerce"),
                       "longitude": pd.to_numeric(raw[col["longitude"]], errors="coerce"),
                       "value": pd.to_numeric(raw[col["value"]], errors="coerce")})
    bbox_cols = ["lat_min", "lat_max", "lon_min", "lon_max"]
    has_bbox = all(col[c] in raw.columns for c in bbox_cols)
    for c in bbox_cols:
        df[c] = pd.to_numeric(raw[col[c]], errors="coerce") if has_bbox else np.nan
    log.info("Hazard dataset: %d rows, bbox columns present: %s", len(df), has_bbox)

    bad = df[["latitude", "longitude", "value"]].isna().any(axis=1) | ~df["latitude"].between(-90, 90) \
        | ~df["longitude"].between(-180, 180)
    if bad.any():
        log.warning("Dropping %d rows with missing/invalid latitude, longitude or hazard value.", int(bad.sum()))
    df = df[~bad].copy()

    vals = df["value"].round(6).value_counts().sort_index()
    log.info("flood_hazard value counts:\n%s", vals.head(15).to_string())
    levels = {float(k): v for k, v in hz.get("levels", {}).items()}
    for v, n in vals.items():
        log.info("  level %s (%s): %d cells", v, levels.get(float(v), "NOT IN LEGEND"), n)
    unknown_levels = sorted(set(vals.index) - set(levels))
    if levels and unknown_levels:
        log.warning("flood_hazard values not in the settings legend: %s - check hazard.levels.", unknown_levels)
    log.info("Hazard cells = flood_hazard >= %s; deep cells = flood_hazard >= %s", hz["positive_threshold"],
             hz.get("high_threshold", "n/a"))

    # same cell listed twice: identical rows collapse, conflicting values are dropped (Unknown)
    key = ["latitude", "longitude"]
    nuniq = df.groupby(key)["value"].transform("nunique")
    conflict = nuniq > 1
    if conflict.any():
        log.warning("%d rows belong to cells with conflicting hazard values; they are dropped.", int(conflict.sum()))
    df = df[~conflict].drop_duplicates(subset=key).reset_index(drop=True)
    df["cell_id"] = [f"CELL_{i + 1:07d}" for i in range(len(df))]
    df["hazard_pos"] = (df["value"] >= hz["positive_threshold"]).astype(int)
    df["hazard_high"] = (df["value"] >= hz.get("high_threshold", np.inf)).astype(int)

    crs_w = cfg["project"]["working_crs"]
    valid_box = has_bbox & (df["lat_min"] < df["lat_max"]) & (df["lon_min"] < df["lon_max"]) & \
        (df["lat_min"] <= df["latitude"]) & (df["latitude"] <= df["lat_max"]) & \
        (df["lon_min"] <= df["longitude"]) & (df["longitude"] <= df["lon_max"])
    if has_bbox and (~valid_box).any():
        log.warning("%d rows have an inconsistent bounding box; a square of +/-%d m around the point is used.",
                    int((~valid_box).sum()), hz["fallback_half_size_m"])
    parts = []
    if valid_box.any():
        v = df[valid_box]
        g = gpd.GeoDataFrame(v, geometry=[box(a, b, c, d) for a, b, c, d in
                                          zip(v["lon_min"], v["lat_min"], v["lon_max"], v["lat_max"])],
                             crs=cfg["project"]["geographic_crs"]).to_crs(crs_w)
        parts.append(g)
    if (~valid_box).any():
        f = df[~valid_box]
        pts = gpd.GeoSeries(gpd.points_from_xy(f["longitude"], f["latitude"]), index=f.index,
                            crs=cfg["project"]["geographic_crs"]).to_crs(crs_w)
        parts.append(gpd.GeoDataFrame(f, geometry=pts.buffer(hz["fallback_half_size_m"], cap_style=3), crs=crs_w))
    cells = pd.concat(parts, ignore_index=True)
    cells = gpd.GeoDataFrame(cells, geometry="geometry", crs=crs_w)

    if clip:
        boundary = gpd.read_file(resolve(cfg["paths"]["interim"]) / "nairobi_boundary.gpkg", layer="boundary")
        inside = cells.intersects(boundary.geometry.iloc[0])
        log.info("%d of %d cells intersect the study area; %d are outside and ignored.", int(inside.sum()),
                 len(cells), int((~inside).sum()))
        cells = cells[inside].reset_index(drop=True)
    if cells.empty:
        raise ValueError("No hazard cells fall inside the study area - check lat/lon order and CRS.")
    area = cells.geometry.area
    log.info("Cell area (m2): median %.0f, min %.0f, max %.0f; total %.1f km2", area.median(), area.min(), area.max(),
             area.sum() / 1e6)
    overlap_report(cells, log)
    log.info("Positive cells: %d (%.1f%%)", int(cells["hazard_pos"].sum()), 100 * cells["hazard_pos"].mean())
    return cells


def label_segments(cfg: dict, cells: gpd.GeoDataFrame, log) -> pd.DataFrame:
    hz = cfg["hazard"]
    seg = gpd.read_file(resolve(cfg["paths"]["interim"]) / "road_segments.gpkg", layer="segments")
    seg = seg[["segment_id", "segment_length_m", "geometry"]]
    inter = gpd.overlay(seg[["segment_id", "geometry"]], cells[["cell_id", "hazard_pos", "hazard_high", "value", "geometry"]],
                        how="intersection", keep_geom_type=True)
    inter["len"] = inter.geometry.length
    inter = inter[inter["len"] > hz.get("min_overlap_m", 0.5)]
    g = inter.groupby("segment_id")
    stats = pd.DataFrame({
        "n_cells": g["cell_id"].nunique(),
        "covered_length_m": g["len"].sum(),
        "pos_length_m": inter[inter["hazard_pos"] == 1].groupby("segment_id")["len"].sum(),
        "high_length_m": inter[inter["hazard_high"] == 1].groupby("segment_id")["len"].sum(),
        "max_hazard_value": g["value"].max(),
        "mean_hazard_value": g.apply(lambda d: np.average(d["value"], weights=d["len"]), include_groups=False),
    }).reset_index()
    out = seg.drop(columns="geometry").merge(stats, on="segment_id", how="left")
    out["n_cells"] = out["n_cells"].fillna(0).astype(int)
    out["covered_length_m"] = out["covered_length_m"].fillna(0.0)
    out["pos_length_m"] = out["pos_length_m"].fillna(0.0)
    out["high_length_m"] = out["high_length_m"].fillna(0.0)
    out["covered_fraction"] = (out["covered_length_m"] / out["segment_length_m"]).clip(upper=1.0)
    out["hazard_pos_fraction"] = np.where(out["covered_length_m"] > 0, out["pos_length_m"] / out["covered_length_m"].replace(0, np.nan), np.nan)

    out["hazard_high_fraction"] = np.where(out["covered_length_m"] > 0, out["high_length_m"] / out["covered_length_m"].replace(0, np.nan), np.nan)

    status = pd.Series("unknown", index=out.index)
    reason = pd.Series("", index=out.index)
    covered = out["covered_fraction"] >= hz["min_coverage"]
    # segments that merely touch a cell edge (overlap below min_overlap_m) intersect the footprint but have no coverage
    touched = set(gpd.sjoin(seg[["segment_id", "geometry"]], cells[["geometry"]], predicate="intersects")["segment_id"])
    reason[~covered] = np.where(out.loc[~covered, "n_cells"] == 0,
                                np.where(out.loc[~covered, "segment_id"].isin(touched), "touch_only_overlap_below_min",
                                         "no_hazard_cells"), "insufficient_coverage")
    frac = out["hazard_pos_fraction"]
    pos = covered & (frac >= hz["positive_fraction_min"])
    zero = covered & (frac == 0)
    mixed = covered & (frac > 0) & (frac < hz["positive_fraction_min"])
    status[pos] = "positive"
    if hz["treat_zero_as_negative"]:
        status[zero] = "negative"
    else:
        reason[zero] = "zero_means_no_information"
    policy = hz["mixed_policy"]
    if policy == "positive":
        status[mixed] = "positive"
    elif policy == "negative":
        status[mixed] = "negative"
    else:
        reason[mixed] = "mixed_cells"
    max_level = max([float(k) for k in hz.get("levels", {1: 0})] + [1.0]) if hz.get("levels") else float(cells["value"].max())
    # continuous score (length-weighted mean level / highest level): defined for every covered segment, incl. mixed
    out["susceptibility_score"] = (out["mean_hazard_value"] / max_level).where(covered)
    out["label_source"] = np.where(covered, "fastflood_original", "none")
    out["label_status"] = status
    out["unknown_reason"] = np.where(status == "unknown", reason, "")
    n_mixed = int(mixed.sum())
    if n_mixed:
        log.info("'mixed_cells' = covered segments with 0 < hazard share < %.2f (%d). They are mostly low-hazard with a "
                 "minority of hazard cells, not conflicting data. Options: mixed_policy negative/positive, or model "
                 "susceptibility_score directly.", hz["positive_fraction_min"], n_mixed)
    out["hazard_label"] = status.map({"positive": 1.0, "negative": 0.0})

    cent = gpd.GeoSeries(seg.geometry.centroid, crs=seg.crs)
    bs = hz["block_size_m"]
    out["spatial_block"] = [f"B_{int(np.floor(x / bs))}_{int(np.floor(y / bs))}" for x, y in zip(cent.x, cent.y)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_settings(args.config)
    log = setup_logging("build_hazard_labels", cfg)
    hz = cfg["hazard"]

    cells = load_cells(cfg, log)
    cells.to_file(ensure_parent(resolve(cfg["paths"]["interim"]) / "hazard_cells.gpkg"), layer="cells", driver="GPKG")
    labels = label_segments(cfg, cells, log)
    labels.to_parquet(ensure_parent(resolve(cfg["paths"]["processed"]) / "hazard_labels.parquet"), index=False)

    summ = labels.groupby("label_status").agg(segments=("segment_id", "size"),
                                              km=("segment_length_m", lambda s: s.sum() / 1000)).reset_index()
    summ.to_csv(ensure_parent(resolve(cfg["paths"]["outputs"]) / "tables" / "hazard_label_summary.csv"), index=False)
    log.info("Segment label summary:\n%s", summ.round(1).to_string(index=False))
    log.info("Unknown reasons:\n%s", labels.loc[labels["label_status"] == "unknown", "unknown_reason"].value_counts().to_string())

    # Gate G4h
    n_pos = int((labels["label_status"] == "positive").sum())
    n_neg = int((labels["label_status"] == "negative").sum())
    pos_blocks = labels.loc[labels["label_status"] == "positive", "spatial_block"].nunique()
    if n_pos < hz["min_positive_segments"] or n_neg < hz["min_negative_segments"]:
        log.warning("GATE G4h WARNING: %d positive and %d negative segments (need >= %d / %d).",
                    n_pos, n_neg, hz["min_positive_segments"], hz["min_negative_segments"])
    if pos_blocks < hz["min_positive_blocks"]:
        log.warning("GATE G4h WARNING: positives occur in only %d spatial blocks (need >= %d); spatial "
                    "validation will be weak.", pos_blocks, hz["min_positive_blocks"])
    log.info("Prevalence among labelled segments: %.1f%%", 100 * n_pos / max(1, n_pos + n_neg))
    log.info("These labels are MODELLED (FastFlood) static hazard labels with no timestamps; they cannot validate 24 h forecasts "
             "and are not proof that roads flooded. Test them against observed events when you have them.")


if __name__ == "__main__":
    main()
