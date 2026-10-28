"""Build static segment features and forecast features.

Usage:
    python src/build_features.py static                       # terrain, land cover, river, road context
    python src/build_features.py forecast --from-labels       # forecast features at every label issue time
    python src/build_features.py forecast --issue-time 2026-10-24T03:00:00Z      # one explicit issue time (UTC)
    python src/build_features.py forecast --latest            # latest issue time present in weather_hourly.parquet

Leakage rule: for an issue time T only forecast rows with forecast_issue_time <= T are used, and for each
(location, variable, valid time) the latest such forecast wins. Rows with unknown issue time are ignored.
"""
from __future__ import annotations

import argparse
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from scipy import ndimage
from scipy.spatial import cKDTree

from common import ensure_parent, load_settings, resolve, setup_logging, utc

SAMPLE_FRACTIONS = np.linspace(0.0, 1.0, 5)


# --------------------------------------------------------------------------------------
# Raster sampling helpers
# --------------------------------------------------------------------------------------
def sample_points(geoms: gpd.GeoSeries, fractions=SAMPLE_FRACTIONS):
    """Return x, y arrays (n_segments, n_fractions) of points interpolated along each line."""
    g = np.asarray(geoms.values)
    xs, ys = [], []
    for f in fractions:
        pts = shapely.line_interpolate_point(g, f, normalized=True)
        xs.append(shapely.get_x(pts))
        ys.append(shapely.get_y(pts))
    return np.column_stack(xs), np.column_stack(ys)


def sample_raster(arr: np.ndarray, transform, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Nearest-pixel sampling of a float array (NaN = invalid). Out-of-bounds -> NaN."""
    col = np.floor((x - transform.c) / transform.a).astype(int)
    row = np.floor((y - transform.f) / transform.e).astype(int)
    ok = (row >= 0) & (row < arr.shape[0]) & (col >= 0) & (col < arr.shape[1])
    out = np.full(x.shape, np.nan, dtype="float64")
    out[ok] = arr[row[ok], col[ok]]
    return out


def nanmean_rows(a: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=1)


def disc_kernel(radius_px: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    return (xx ** 2 + yy ** 2 <= radius_px ** 2).astype("float32")


# --------------------------------------------------------------------------------------
# Static features
# --------------------------------------------------------------------------------------
def terrain_features(cfg: dict, seg: gpd.GeoDataFrame, x, y, log) -> pd.DataFrame:
    path = resolve(cfg["paths"]["interim"]) / "dem_clip.tif"
    with rasterio.open(path) as src:
        dem = src.read(1).astype("float32")
        nodata = src.nodata
        transform = src.transform
    if nodata is not None:
        dem[dem == nodata] = np.nan
    res = abs(transform.a)
    gy, gx = np.gradient(dem, res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    half = max(1, int(round(cfg["dem"]["relief_window_m"] / res)))
    size = 2 * half + 1
    hi = np.where(np.isnan(dem), -np.inf, dem)
    lo = np.where(np.isnan(dem), np.inf, dem)
    relief = ndimage.maximum_filter(hi, size=size) - ndimage.minimum_filter(lo, size=size)
    relief[~np.isfinite(relief)] = np.nan
    log.info("DEM: %.0f-%.0f m, mean slope %.1f deg", np.nanmin(dem), np.nanmax(dem), np.nanmean(slope))
    return pd.DataFrame({
        "elevation_m": nanmean_rows(sample_raster(dem, transform, x, y)),
        "slope_deg": nanmean_rows(sample_raster(slope, transform, x, y)),
        "local_relief_m": nanmean_rows(sample_raster(relief, transform, x, y)),
    })


def landcover_features(cfg: dict, x, y, log) -> pd.DataFrame:
    lc_cfg = cfg["landcover"]
    path = resolve(cfg["paths"]["interim"]) / "landcover_clip.tif"
    with rasterio.open(path) as src:
        lc = src.read(1)
        transform = src.transform
    nodata = int(lc_cfg["nodata"])
    valid = (lc != nodata).astype("float32")
    res = abs(transform.a)
    kernel = disc_kernel(max(1, int(round(lc_cfg["buffer_m"] / res))))
    denom = ndimage.convolve(valid, kernel, mode="constant", cval=0.0)
    codes = [int(c) for c in np.unique(lc) if int(c) != nodata]
    unknown = [c for c in codes if c not in lc_cfg["class_names"]]
    if unknown:
        log.warning("Land-cover codes not in settings legend (verify the product legend!): %s", unknown)
    log.info("Land-cover classes present: %s", codes)
    out = {}
    for code in codes:
        num = ndimage.convolve((lc == code).astype("float32"), kernel, mode="constant", cval=0.0)
        frac = np.where(denom > 0, num / np.maximum(denom, 1e-6), np.nan)
        out[f"lc_frac_{code}"] = nanmean_rows(sample_raster(frac, transform, x, y))
    df = pd.DataFrame(out)

    def total(codes_):
        cols = [f"lc_frac_{c}" for c in codes_ if f"lc_frac_{c}" in df.columns]
        return df[cols].sum(axis=1, min_count=1) if cols else pd.Series(0.0, index=df.index)

    df["builtup_fraction"] = total(lc_cfg["builtup_codes"])
    df["vegetation_fraction"] = total(lc_cfg["vegetation_codes"])
    df["water_fraction"] = total(lc_cfg["water_codes"])
    df["other_fraction"] = (1.0 - df[["builtup_fraction", "vegetation_fraction", "water_fraction"]].sum(axis=1)).clip(lower=0)
    return df


def river_distance(cfg: dict, seg: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame, log) -> pd.Series:
    path = resolve(cfg["inputs"]["rivers_shapefile"])
    if not path.exists():
        log.warning("Rivers layer not found (%s): distance_to_river_m will be NaN.", path)
        return pd.Series(np.nan, index=seg.index)
    rivers = gpd.read_file(path)
    if "fclass" in rivers.columns:
        rivers = rivers[rivers["fclass"].isin(cfg["rivers"]["keep_fclass"])]
    rivers = rivers.to_crs(seg.crs)
    rivers = rivers[rivers.intersects(boundary.buffer(5000).iloc[0])]
    if rivers.empty:
        log.warning("No river features near the study area: distance_to_river_m will be NaN.")
        return pd.Series(np.nan, index=seg.index)
    joined = gpd.sjoin_nearest(seg[["geometry"]], rivers[["geometry"]], how="left", distance_col="d")
    joined = joined[~joined.index.duplicated(keep="first")]
    log.info("River distance (m): median %.0f, max %.0f", joined["d"].median(), joined["d"].max())
    return joined["d"].reindex(seg.index)


def build_static(cfg: dict, log) -> pd.DataFrame:
    interim = resolve(cfg["paths"]["interim"])
    seg = gpd.read_file(interim / "road_segments.gpkg", layer="segments")
    boundary = gpd.read_file(interim / "nairobi_boundary.gpkg", layer="boundary")
    x, y = sample_points(seg.geometry)
    terr = terrain_features(cfg, seg, x, y, log)
    lcf = landcover_features(cfg, x, y, log)
    cent = seg.geometry.centroid
    cent_wgs = gpd.GeoSeries(cent, crs=seg.crs).to_crs(cfg["project"]["geographic_crs"])

    df = pd.DataFrame({"segment_id": seg["segment_id"].values})
    if "osm_id" in seg.columns:
        df["osm_id"] = seg["osm_id"].values
    if "name" in seg.columns:
        df["road_name"] = seg["name"].values
    df["road_class"] = seg["fclass"].values if "fclass" in seg.columns else "unknown"
    df["segment_length_m"] = seg["segment_length_m"].values
    df["centroid_x"] = cent.x.values
    df["centroid_y"] = cent.y.values
    df["centroid_lon"] = cent_wgs.x.values
    df["centroid_lat"] = cent_wgs.y.values
    df = pd.concat([df, terr, lcf], axis=1)
    df["distance_to_river_m"] = river_distance(cfg, seg, boundary, log).values
    # Missing tags are NOT proof of absence: keep NaN when the column does not exist
    for tag, col in (("bridge", "is_bridge"), ("tunnel", "is_tunnel")):
        df[col] = (seg[tag].astype(str).str.upper().eq("T").astype(float).values if tag in seg.columns else np.nan)
    df["surface"] = seg["surface"].values if "surface" in seg.columns else None

    out = ensure_parent(resolve(cfg["paths"]["processed"]) / "static_segment_features.parquet")
    df.to_parquet(out, index=False)
    log.info("Static features: %d segments x %d columns -> %s", len(df), df.shape[1], out)
    log.info("Missing share per feature:\n%s", df.isna().mean().round(3).to_string())
    return df


# --------------------------------------------------------------------------------------
# Weather location assignment
# --------------------------------------------------------------------------------------
def assign_weather_points(cfg: dict, static: pd.DataFrame, log) -> pd.DataFrame:
    grid = gpd.read_file(resolve(cfg["paths"]["grid_points"])).to_crs(cfg["project"]["working_crs"])
    tree = cKDTree(np.column_stack([grid.geometry.x, grid.geometry.y]))
    dist, idx = tree.query(np.column_stack([static["centroid_x"], static["centroid_y"]]))
    mapping = pd.DataFrame({
        "segment_id": static["segment_id"].values,
        "weather_location_id": grid["location_id"].values[idx],
        "weather_distance_m": dist.round(1),
    })
    out = ensure_parent(resolve(cfg["paths"]["processed"]) / "segment_weather_map.parquet")
    mapping.to_parquet(out, index=False)
    log.info("Assigned %d segments to %d weather points (max distance %.0f m)",
             len(mapping), mapping["weather_location_id"].nunique(), dist.max())
    return mapping


# --------------------------------------------------------------------------------------
# Forecast features
# --------------------------------------------------------------------------------------
def compute_forecast_features(weather: pd.DataFrame, issue_times, cfg: dict, log) -> pd.DataFrame:
    fc = cfg["forecast_features"]
    W, S, A = fc["window_hours"], fc["short_window_hours"], fc["antecedent_hours"]
    thr, min_cov = fc["wet_hour_threshold_mm"], fc["min_hour_coverage"]
    w = weather.dropna(subset=["forecast_issue_time_utc"]).copy()
    if len(w) < len(weather):
        log.warning("Ignored %d weather rows with unknown issue time (archive mode) - not valid for strict "
                    "24 h forecast evaluation.", len(weather) - len(w))
    rows = []
    for T in sorted({utc(t) for t in issue_times}):
        sub = w[(w["valid_time_utc"] >= T - pd.Timedelta(hours=A)) &
                (w["valid_time_utc"] < T + pd.Timedelta(hours=W)) &
                (w["forecast_issue_time_utc"] <= T)]
        if sub.empty:
            log.warning("No usable forecast rows for issue time %s", T)
            continue
        sub = sub.sort_values("forecast_issue_time_utc").drop_duplicates(
            ["location_id", "variable", "valid_time_utc"], keep="last")
        piv = sub.pivot_table(index=["location_id", "valid_time_utc"], columns="variable", values="value",
                              aggfunc="last").reset_index()
        issue_max = sub.groupby("location_id")["forecast_issue_time_utc"].max()
        piv["h"] = (piv["valid_time_utc"] - T).dt.total_seconds() / 3600.0
        fut = piv[(piv["h"] >= 0) & (piv["h"] < W)]
        ante = piv[piv["h"] < 0]
        locs = pd.Index(sorted(piv["location_id"].unique()), name="location_id")
        feat = pd.DataFrame(index=locs)

        if "precipitation" in fut.columns:
            p = fut.dropna(subset=["precipitation"])
            g = p.groupby("location_id")["precipitation"]
            feat["n_hours_24h"] = g.size()
            feat["rain_fcst_next_24h_mm"] = g.sum()
            feat["max_hourly_precip_mm"] = g.max()
            feat["wet_hours_count"] = p[p["precipitation"] >= thr].groupby("location_id").size()
            p6 = p[p["h"] < S]
            feat["n_hours_6h"] = p6.groupby("location_id").size()
            feat["rain_fcst_next_6h_mm"] = p6.groupby("location_id")["precipitation"].sum()
            if "precipitation" in ante.columns:
                pa = ante.dropna(subset=["precipitation"])
                ga = pa.groupby("location_id")["precipitation"]
                feat["n_hours_ante"] = ga.size()
                feat["antecedent_fcst_prev_24h_mm"] = ga.sum()
        if "precipitation_probability" in fut.columns:
            feat["precipitation_probability_max"] = fut.groupby("location_id")["precipitation_probability"].max()

        for col in ["n_hours_24h", "n_hours_6h", "n_hours_ante", "rain_fcst_next_24h_mm", "max_hourly_precip_mm",
                    "wet_hours_count", "rain_fcst_next_6h_mm", "antecedent_fcst_prev_24h_mm",
                    "precipitation_probability_max"]:
            if col not in feat.columns:
                feat[col] = np.nan
        feat["wet_hours_count"] = feat["wet_hours_count"].where(feat["n_hours_24h"].notna(), np.nan)
        feat.loc[feat["n_hours_24h"].notna() & feat["wet_hours_count"].isna(), "wet_hours_count"] = 0

        # coverage rules: incomplete windows give NaN features and a quality flag
        bad24 = feat["n_hours_24h"].fillna(0) < min_cov * W
        bad6 = feat["n_hours_6h"].fillna(0) < min_cov * S
        badA = feat["n_hours_ante"].fillna(0) < min_cov * A
        for c in ["rain_fcst_next_24h_mm", "max_hourly_precip_mm", "wet_hours_count"]:
            feat.loc[bad24, c] = np.nan
        feat.loc[bad6, "rain_fcst_next_6h_mm"] = np.nan
        feat.loc[badA, "antecedent_fcst_prev_24h_mm"] = np.nan
        feat["data_quality_flag"] = np.where(bad24, "INCOMPLETE_WEATHER", "OK")
        feat["max_issue_time_used_utc"] = issue_max.reindex(locs)
        feat["forecast_issue_time_utc"] = T
        rows.append(feat.reset_index())
    if not rows:
        raise RuntimeError("No forecast features could be computed - check weather_hourly.parquet.")
    out = pd.concat(rows, ignore_index=True)
    out = out.rename(columns={"location_id": "weather_location_id"})
    # Gate G5 leakage assertion
    assert (out["max_issue_time_used_utc"] <= out["forecast_issue_time_utc"]).all(), "Leakage: future issue time used"
    return out


def run_forecast(args, cfg: dict, log) -> pd.DataFrame:
    processed = resolve(cfg["paths"]["processed"])
    static = pd.read_parquet(processed / "static_segment_features.parquet")
    assign_weather_points(cfg, static, log)
    weather = pd.read_parquet(processed / "weather_hourly.parquet")
    if args.from_labels:
        labels = pd.read_parquet(processed / "labels.parquet")
        issue_times = labels["forecast_issue_time_utc"].dropna().unique()
    elif args.issue_time:
        issue_times = [utc(args.issue_time)]
    elif args.latest:
        wl = weather.dropna(subset=["forecast_issue_time_utc"])
        issue_times = [wl["forecast_issue_time_utc"].max()]
    else:
        raise SystemExit("Choose one of --from-labels, --issue-time or --latest.")
    feats = compute_forecast_features(weather, issue_times, cfg, log)
    out = ensure_parent(processed / ("forecast_features.parquet" if args.from_labels else "forecast_features_live.parquet"))
    feats.to_parquet(out, index=False)
    log.info("Forecast features: %d rows (%d issue times) -> %s", len(feats), feats["forecast_issue_time_utc"].nunique(), out)
    log.info("Share flagged INCOMPLETE_WEATHER: %.3f", (feats["data_quality_flag"] != "OK").mean())
    return feats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["static", "forecast"])
    ap.add_argument("--from-labels", action="store_true")
    ap.add_argument("--issue-time", default=None)
    ap.add_argument("--latest", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_settings(args.config)
    log = setup_logging("build_features", cfg)
    if args.command == "static":
        static = build_static(cfg, log)
        if resolve(cfg["paths"]["grid_points"]).exists():
            assign_weather_points(cfg, static, log)
    else:
        run_forecast(args, cfg, log)


if __name__ == "__main__":
    main()
