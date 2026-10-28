Verifies assets, clips and cleans roads, splits them into stable segments, and clips/warps the DEM and land cover to a working grid. Commands: inventory, all.
"""Phase 1-2: verify assets, clip roads, build road segments, clip/warp DEM and land cover.

Usage:
    python src/prepare_gis.py inventory      # metadata + overlap QA only (gate G2/G3)
    python src/prepare_gis.py all            # boundary, roads, segments, rasters
Original downloads are never modified.
"""
from __future__ import annotations

import argparse
import math

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from shapely.geometry import LineString, MultiLineString, box
from shapely.ops import substring

from common import boundary_sanity, ensure_parent, load_settings, resolve, setup_logging

ROAD_KEEP_COLS = ["osm_id", "fclass", "name", "ref", "oneway", "maxspeed", "layer", "bridge", "tunnel", "surface"]


# --------------------------------------------------------------------------------------
# Boundary
# --------------------------------------------------------------------------------------
def load_boundary(cfg: dict, log) -> gpd.GeoDataFrame:
    inp = cfg["inputs"]
    path = resolve(inp["boundary_file"])
    if not path.exists():
        raise FileNotFoundError(f"Boundary file not found: {path}")
    kwargs = {"layer": inp["boundary_layer"]} if inp.get("boundary_layer") else {}
    gdf = gpd.read_file(path, **kwargs)
    if inp.get("boundary_name_field") and inp.get("boundary_name_value"):
        field, value = inp["boundary_name_field"], inp["boundary_name_value"]
        if field not in gdf.columns:
            raise KeyError(f"Boundary field '{field}' not in columns {list(gdf.columns)}")
        gdf = gdf[gdf[field].astype(str).str.strip().str.lower() == str(value).strip().lower()]
        if gdf.empty:
            raise ValueError(f"No boundary polygon where {field} == {value}")
    if gdf.crs is None:
        raise ValueError("Boundary has no CRS defined; assign the correct CRS before use.")
    gdf = gdf.to_crs(cfg["project"]["working_crs"])
    geom = shapely.make_valid(gdf.geometry.union_all())
    out = gpd.GeoDataFrame({"name": ["study_area"]}, geometry=[geom], crs=cfg["project"]["working_crs"])
    boundary_sanity(geom, cfg, log)
    return out


# --------------------------------------------------------------------------------------
# Rasters
# --------------------------------------------------------------------------------------
def raster_info(path, boundary_wgs: gpd.GeoDataFrame | None, label: str) -> dict:
    with rasterio.open(path) as src:
        info = {
            "asset": label,
            "path": str(path),
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "pixel_x": src.res[0],
            "pixel_y": src.res[1],
            "bounds": tuple(round(v, 5) for v in src.bounds),
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
            "units": (src.units[0] if src.units and src.units[0] else None),
        }
        if boundary_wgs is not None and src.crs is not None:
            b = boundary_wgs.to_crs(src.crs).union_all()
            rb = box(*src.bounds)
            info["boundary_fraction_covered"] = round(b.intersection(rb).area / b.area, 4)
    return info


def warp_to_grid(src_path, dst_path, bounds, res, resampling, nodata, crs, dtype, log):
    """Warp src raster onto a regular grid in `crs` covering `bounds` (minx, miny, maxx, maxy)."""
    minx, miny, maxx, maxy = bounds
    minx = math.floor(minx / res) * res
    miny = math.floor(miny / res) * res
    maxx = math.ceil(maxx / res) * res
    maxy = math.ceil(maxy / res) * res
    width = int(round((maxx - minx) / res))
    height = int(round((maxy - miny) / res))
    transform = from_origin(minx, maxy, res, res)
    with rasterio.open(src_path) as src:
        dst = np.full((height, width), nodata, dtype=dtype)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=nodata,
            resampling=resampling,
        )
    profile = {
        "driver": "GTiff", "height": height, "width": width, "count": 1, "dtype": dtype,
        "crs": crs, "transform": transform, "nodata": nodata, "compress": "deflate",
    }
    ensure_parent(dst_path)
    with rasterio.open(dst_path, "w", **profile) as out:
        out.write(dst, 1)
    valid = np.mean(dst != nodata)
    log.info("Wrote %s (%d x %d, %.1f%% valid)", dst_path.name, width, height, 100 * valid)
    if valid < 0.5:
        log.warning("%s has under 50%% valid pixels - check CRS/coverage of the source raster.", dst_path.name)


def prepare_rasters(cfg: dict, boundary: gpd.GeoDataFrame, log) -> None:
    crs = cfg["project"]["working_crs"]
    interim = resolve(cfg["paths"]["interim"])

    dem_cfg = cfg["dem"]
    bounds = boundary.buffer(dem_cfg["margin_m"]).total_bounds
    warp_to_grid(resolve(cfg["inputs"]["dem"]), interim / "dem_clip.tif", bounds, dem_cfg["resolution_m"],
                 Resampling.bilinear, float(dem_cfg["nodata"]), crs, "float32", log)

    lc_cfg = cfg["landcover"]
    warp_to_grid(resolve(cfg["inputs"]["landcover"]), interim / "landcover_clip.tif", bounds, lc_cfg["resolution_m"],
                 Resampling.nearest, int(lc_cfg["nodata"]), crs, "uint8", log)  # nearest: categorical


# --------------------------------------------------------------------------------------
# Roads
# --------------------------------------------------------------------------------------
def explode_lines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.explode(index_parts=False, ignore_index=True)
    return gdf[gdf.geometry.geom_type == "LineString"].reset_index(drop=True)


def clip_roads(cfg: dict, boundary: gpd.GeoDataFrame, log) -> gpd.GeoDataFrame:
    path = resolve(cfg["inputs"]["roads_shapefile"])
    if not path.exists():
        raise FileNotFoundError(
            f"Roads layer not found: {path}. The OSM download has several layers - look for a "
            "gis_osm_roads_* LINE layer, not adminareas/buildings/landuse/natural.")
    roads = gpd.read_file(path)
    if roads.crs is None:
        raise ValueError("Roads layer has no CRS (.prj missing?). Keep .shp/.shx/.dbf/.prj/.cpg together.")
    log.info("Roads read: %d features, geometry types %s", len(roads), sorted(roads.geom_type.unique()))
    if not roads.geom_type.isin(["LineString", "MultiLineString"]).all():
        raise ValueError("Roads layer contains non-line geometries - wrong layer loaded (gate G3).")

    roads = roads.to_crs(cfg["project"]["working_crs"])
    roads = roads[~roads.geometry.is_empty & roads.geometry.notna()].copy()
    roads["geometry"] = shapely.make_valid(roads.geometry.values)
    keep = cfg["roads"]["keep_fclass"]
    if "fclass" in roads.columns:
        before = len(roads)
        roads = roads[roads["fclass"].isin(keep)]
        log.info("fclass filter kept %d of %d features", len(roads), before)
        log.info("fclass counts kept:\n%s", roads["fclass"].value_counts().to_string())
    else:
        log.warning("No 'fclass' column - road class filter skipped. Columns: %s", list(roads.columns))

    cols = [c for c in ROAD_KEEP_COLS if c in roads.columns]
    roads = roads[cols + ["geometry"]]
    clipped = gpd.clip(roads, boundary.geometry.iloc[0])
    clipped = explode_lines(clipped)
    clipped = clipped[clipped.geometry.length > 0]
    n0 = len(clipped)
    clipped = clipped.drop_duplicates(subset=["geometry"] if "osm_id" not in clipped.columns else ["osm_id", "geometry"])
    log.info("Dropped %d duplicate road records; %d lines remain", n0 - len(clipped), len(clipped))
    if clipped.empty:
        raise ValueError("No roads left after clipping - check CRS of boundary and roads (gate G2).")
    return clipped.reset_index(drop=True)


def split_line(line: LineString, target_m: float) -> list[LineString]:
    length = line.length
    n = max(1, int(round(length / target_m)))
    if n == 1:
        return [line]
    step = length / n
    return [substring(line, i * step, (i + 1) * step) for i in range(n)]


def build_segments(roads: gpd.GeoDataFrame, cfg: dict, log) -> gpd.GeoDataFrame:
    target = cfg["roads"]["segment_length_m"]
    min_len = cfg["roads"]["min_segment_length_m"]
    sort_cols = [c for c in ["osm_id"] if c in roads.columns]
    roads = roads.assign(wkt_tmp=roads.geometry.to_wkt(rounding_precision=1))
    roads = roads.sort_values(sort_cols + ["wkt_tmp"]).reset_index(drop=True)  # deterministic order -> stable IDs
    rows = []
    for rec in roads.itertuples(index=False):
        attrs = rec._asdict()
        geom = attrs.pop("geometry")
        attrs.pop("wkt_tmp")
        for k, piece in enumerate(split_line(geom, target)):
            if piece.length >= min_len and not piece.is_empty:
                rows.append({**attrs, "piece_in_road": k, "geometry": piece})
    seg = gpd.GeoDataFrame(rows, crs=roads.crs)
    seg.insert(0, "segment_id", [f"NRD_{i + 1:06d}" for i in range(len(seg))])
    seg["segment_length_m"] = seg.geometry.length.round(2)
    log.info("Built %d segments (target %d m); length quantiles (m): %s", len(seg), target,
             seg["segment_length_m"].quantile([0, .05, .5, .95, 1]).round(1).to_dict())
    return seg


# --------------------------------------------------------------------------------------
def run_inventory(cfg: dict, log) -> pd.DataFrame:
    boundary = load_boundary(cfg, log)
    bwgs = boundary.to_crs(cfg["project"]["geographic_crs"])
    rows = [raster_info(resolve(cfg["inputs"]["dem"]), bwgs, "dem"),
            raster_info(resolve(cfg["inputs"]["landcover"]), bwgs, "landcover")]
    df = pd.DataFrame(rows)
    out = ensure_parent(resolve(cfg["paths"]["outputs"]) / "tables" / "asset_inventory.csv")
    df.to_csv(out, index=False)
    log.info("Asset inventory:\n%s", df.to_string())
    for r in rows:
        if r.get("boundary_fraction_covered", 1) < 0.99:
            log.warning("%s covers only %.1f%% of the boundary (gate G2).", r["asset"],
                        100 * r["boundary_fraction_covered"])
    roads_path = resolve(cfg["inputs"]["roads_shapefile"])
    if roads_path.exists():
        sidecars = [roads_path.with_suffix(s).exists() for s in (".shx", ".dbf", ".prj")]
        log.info("Roads sidecars (.shx, .dbf, .prj) present: %s", sidecars)
        if not all(sidecars):
            log.warning("Missing Shapefile sidecar files - keep all components together (gate G3).")
    else:
        log.warning("Roads shapefile not found at %s", roads_path)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["inventory", "all"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_settings(args.config)
    log = setup_logging("prepare_gis", cfg)

    if args.step == "inventory":
        run_inventory(cfg, log)
        return

    run_inventory(cfg, log)
    interim = resolve(cfg["paths"]["interim"])
    boundary = load_boundary(cfg, log)
    ensure_parent(interim / "nairobi_boundary.gpkg")
    boundary.to_file(interim / "nairobi_boundary.gpkg", layer="boundary", driver="GPKG")
    study = ensure_parent(resolve(cfg["paths"]["study_area"]))
    boundary.to_crs(cfg["project"]["geographic_crs"]).to_file(study, driver="GeoJSON")

    roads = clip_roads(cfg, boundary, log)
    roads.to_file(interim / "roads_clipped.gpkg", layer="roads", driver="GPKG")
    segments = build_segments(roads, cfg, log)
    segments.to_file(interim / "road_segments.gpkg", layer="segments", driver="GPKG")
    prepare_rasters(cfg, boundary, log)
    log.info("prepare_gis complete.")


if __name__ == "__main__":
    main()
