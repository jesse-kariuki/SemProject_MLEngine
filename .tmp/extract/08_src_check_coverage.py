NEW. Checks the study boundary and the hazard-cell footprint: boundary area and shape, area and roads without cells, QGIS layers, a map, and a reconciliation with the label script.
"""Diagnose how well the hazard cells cover the study boundary and the road network.

Answers: is the boundary the real county, how much of it has hazard cells, which road segments lie outside the
cell footprint, and does that match the 'no_hazard_cells' count from build_hazard_labels.py?

Usage (project layout from settings.yaml):
    python src/check_coverage.py
Any input can be overridden, e.g. for a different folder layout:
    python src/check_coverage.py --segments D:\\floodguard\\data\\nairobi_roads.gpkg --segments-layer segments ^
        --boundary D:\\floodguard\\data\\nairobi_boundary.gpkg --boundary-layer boundary --cells D:\\floodguard\\data\\hazard.csv
Outputs: outputs/tables/coverage_report.csv, outputs/maps/coverage.png,
         data/interim/uncovered_area.gpkg and data/interim/uncovered_segments.gpkg (open both in QGIS)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402

from build_hazard_labels import load_cells  # noqa: E402
from common import boundary_sanity, ensure_parent, load_settings, resolve, setup_logging  # noqa: E402


def read_layer(path, layer, crs):
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"{path} has no CRS.")
    return gdf.to_crs(crs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--segments", default=None)
    ap.add_argument("--segments-layer", default="segments")
    ap.add_argument("--boundary", default=None)
    ap.add_argument("--boundary-layer", default="boundary")
    ap.add_argument("--cells", default=None)
    ap.add_argument("--close-gaps-m", type=float, default=1.0, help="close sub-metre gaps between cells (rounding)")
    args = ap.parse_args()
    cfg = load_settings(args.config)
    log = setup_logging("check_coverage", cfg)
    crs = cfg["project"]["working_crs"]
    interim = resolve(cfg["paths"]["interim"])
    out_tab = ensure_parent(resolve(cfg["paths"]["outputs"]) / "tables" / "coverage_report.csv")

    bnd = read_layer(args.boundary or interim / "nairobi_boundary.gpkg", args.boundary_layer, crs)
    boundary = shapely.make_valid(bnd.geometry.union_all())
    report = dict(boundary_sanity(boundary, cfg, log))

    cells = load_cells(cfg, log, path=Path(args.cells) if args.cells else None, clip=False)
    log.info("Cells: %d, total area %.1f km2", len(cells), cells.geometry.area.sum() / 1e6)
    g = args.close_gaps_m
    footprint = shapely.union_all(cells.geometry.values)
    if g > 0:
        footprint = footprint.buffer(g).buffer(-g)
    footprint = shapely.make_valid(footprint)

    b_area = boundary.area / 1e6
    inside = footprint.intersection(boundary)
    in_km2 = inside.area / 1e6
    uncovered = boundary.difference(footprint)
    report.update({
        "cells_total": len(cells),
        "cells_total_area_km2": cells.geometry.area.sum() / 1e6,
        "footprint_area_km2": footprint.area / 1e6,
        "footprint_inside_boundary_km2": in_km2,
        "boundary_without_cells_km2": uncovered.area / 1e6,
        "boundary_covered_pct": 100 * in_km2 / b_area,
        "footprint_outside_boundary_km2": (footprint.area - inside.area) / 1e6,
    })
    log.info("Boundary %.1f km2: %.1f km2 (%.1f%%) covered by cells, %.1f km2 without cells; cells outside boundary: %.1f km2",
             b_area, in_km2, report["boundary_covered_pct"], report["boundary_without_cells_km2"],
             report["footprint_outside_boundary_km2"])

    parts = gpd.GeoDataFrame(geometry=[uncovered], crs=crs).explode(index_parts=False)
    parts["area_km2"] = parts.geometry.area / 1e6
    parts = parts.sort_values("area_km2", ascending=False).reset_index(drop=True)
    parts.to_file(ensure_parent(interim / "uncovered_area.gpkg"), layer="uncovered", driver="GPKG")
    log.info("Largest uncovered parts (km2): %s", parts["area_km2"].head(5).round(1).tolist())

    seg = read_layer(args.segments or interim / "road_segments.gpkg", args.segments_layer, crs)
    if "segment_id" not in seg.columns:
        seg["segment_id"] = [f"SEG_{i:07d}" for i in range(len(seg))]
    seg["length_m"] = seg.geometry.length
    seg_inside_b = seg.intersects(boundary)
    hit = seg.intersects(footprint)
    # length-based coverage: share of each segment's length inside the footprint
    cov_len = shapely.length(shapely.intersection(seg.geometry.values, footprint)) / seg["length_m"].replace(0, np.nan)
    seg["footprint_cover"] = cov_len
    total_km = seg["length_m"].sum() / 1000
    out_mask = ~hit
    low_mask = (seg["footprint_cover"] < cfg["hazard"]["min_coverage"]) & hit
    report.update({
        "segments_total": len(seg), "road_km_total": total_km,
        "segments_outside_boundary": int((~seg_inside_b).sum()),
        "segments_intersecting_footprint": int(hit.sum()),
        "segments_outside_footprint": int(out_mask.sum()),
        "segments_outside_footprint_pct": 100 * out_mask.mean(),
        "road_km_outside_footprint": seg.loc[out_mask, "length_m"].sum() / 1000,
        "segments_below_min_coverage_but_touching": int(low_mask.sum()),
    })
    log.info("Segments: %d (%.0f km). Outside boundary: %d. Intersecting footprint: %d. Outside footprint: %d (%.2f%%, %.0f km). "
             "Touching but below min_coverage: %d",
             len(seg), total_km, report["segments_outside_boundary"], report["segments_intersecting_footprint"],
             report["segments_outside_footprint"], report["segments_outside_footprint_pct"],
             report["road_km_outside_footprint"], report["segments_below_min_coverage_but_touching"])
    if "highway" in seg.columns or "fclass" in seg.columns:
        col = "fclass" if "fclass" in seg.columns else "highway"
        by = seg.assign(outside=out_mask).groupby(col)["outside"].agg(["size", "mean"]).sort_values("size", ascending=False)
        log.info("Share of segments outside the footprint by road class (top 8):\n%s", by.head(8).round(3).to_string())
    seg.loc[out_mask, ["segment_id", "geometry"]].to_file(ensure_parent(interim / "uncovered_segments.gpkg"),
                                                          layer="uncovered_segments", driver="GPKG")

    # Reconcile with the production labels
    hl_path = resolve(cfg["paths"]["processed"]) / "hazard_labels.parquet"
    if hl_path.exists():
        hl = pd.read_parquet(hl_path)
        counts = hl["unknown_reason"].value_counts()
        nh, to = int(counts.get("no_hazard_cells", 0)), int(counts.get("touch_only_overlap_below_min", 0))
        log.info("hazard_labels.parquet: no_hazard_cells=%d, touch_only_overlap_below_min=%d; diagnostic outside footprint=%d",
                 nh, to, report["segments_outside_footprint"])
        diff = nh - report["segments_outside_footprint"]
        report.update({"labels_no_hazard_cells": nh, "labels_touch_only": to, "labels_minus_diagnostic": diff})
        if abs(diff) > 0.005 * len(seg):
            log.warning("Label script and diagnostic disagree by %d segments: compare segment sets "
                        "(uncovered_segments.gpkg) and check the segment inputs are the same file.", diff)
        else:
            log.info("The label script and the footprint test agree (difference %d).", diff)
    else:
        log.info("Run build_hazard_labels.py first to reconcile with the label counts.")

    pd.Series(report).rename("value").to_csv(out_tab)
    fig, ax = plt.subplots(figsize=(9, 9))
    gpd.GeoSeries([boundary], crs=crs).boundary.plot(ax=ax, color="black", linewidth=1)
    gpd.GeoSeries([footprint.simplify(30)], crs=crs).plot(ax=ax, color="#9ecae1", alpha=0.6)
    seg.loc[out_mask].plot(ax=ax, color="#c0392b", linewidth=0.4)
    ax.set_title("Hazard-cell footprint (blue) vs roads outside it (red); black = study boundary", fontsize=10)
    ax.set_axis_off()
    fig.savefig(ensure_parent(resolve(cfg["paths"]["outputs"]) / "maps" / "coverage.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s, coverage.png, uncovered_area.gpkg and uncovered_segments.gpkg", out_tab.name)


if __name__ == "__main__":
    main()
