"""Per-road flood SUSCEPTIBILITY (research prototype, not a warning service).

Two kinds of model:
    H_* (trained on the FastFlood hazard dataset) -> STATIC susceptibility: how prone a road segment is to
        flooding, valid for any time. If you also pass --latest or --issue-time, the Open-Meteo 24 h rainfall is
        added as context columns only (it does not change the susceptibility score or tier).
    A_* / B_* (trained on time-stamped events, optional) -> B_* are forecast-conditioned and need an issue time.

Usage:
    python src/predict.py                                  # static susceptibility map (default model H_rf)
    python src/predict.py --latest                         # + latest Open-Meteo rainfall context
    python src/predict.py --issue-time 2026-10-24T03:00:00Z --model H_logreg
    python src/predict.py --latest --model B_rf            # event-trained forecast model (if you built one)
Outputs (outputs/): predictions_<model>_<stamp>.csv / .gpkg and maps/risk_<model>_<stamp>.png
"""
from __future__ import annotations

import argparse
import json

import geopandas as gpd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from build_features import compute_forecast_features  # noqa: E402
from common import ensure_parent, load_settings, resolve, setup_logging, utc  # noqa: E402

def risk_tier(p: float, tiers: dict) -> str:
    if np.isnan(p):
        return "Unavailable"
    label = "Low"
    for name, lower in sorted(tiers.items(), key=lambda kv: kv[1]):
        if p >= lower:
            label = name
    return label


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--issue-time", default=None)
    ap.add_argument("--latest", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_settings(args.config)
    log = setup_logging("predict", cfg)
    proc = resolve(cfg["paths"]["processed"])
    tz = cfg["project"]["display_timezone"]
    name = args.model or cfg["prediction"]["default_model"]

    pipe = joblib.load(resolve(cfg["paths"]["models"]) / f"{name}.joblib")
    meta = json.load(open(resolve(cfg["paths"]["models"]) / f"{name}.json", encoding="utf-8"))
    feats = meta["numeric_features"] + meta["categorical_features"]
    uses_forecast = any(f in set(cfg["training"]["forecast_features"]) for f in feats)

    df = pd.read_parquet(proc / "static_segment_features.parquet")
    map_path = proc / "segment_weather_map.parquet"
    if map_path.exists():
        df = df.merge(pd.read_parquet(map_path), on="segment_id", how="left")
    elif args.issue_time or args.latest:
        raise SystemExit("segment_weather_map.parquet missing: run fetch_openmeteo.py grid, then build_features.py static.")
    else:
        df["weather_location_id"] = None

    hz_path = proc / "hazard_labels.parquet"
    if hz_path.exists():  # direct modelled exposure, shown next to every prediction
        hz = pd.read_parquet(hz_path)[["segment_id", "max_hazard_value", "hazard_pos_fraction", "covered_fraction"]].rename(
            columns={"max_hazard_value": "fastflood_max_hazard", "hazard_pos_fraction": "fastflood_hazard_share",
                     "covered_fraction": "fastflood_coverage"})
        df = df.merge(hz, on="segment_id", how="left")
        # provenance: FastFlood-covered segments carry the original modelled hazard; others are model estimates
        df["susceptibility_source"] = np.where(df["fastflood_coverage"].fillna(0) >= cfg["hazard"]["min_coverage"],
                                               "fastflood_covered", "model_predicted_no_fastflood")

    T = None
    if args.issue_time or args.latest:
        weather = pd.read_parquet(proc / "weather_hourly.parquet")
        T = utc(args.issue_time) if args.issue_time else weather.dropna(subset=["forecast_issue_time_utc"])["forecast_issue_time_utc"].max()
        T = T.floor("h")
        log.info("Issue time %s UTC (%s EAT)", T, T.tz_convert(tz))
        fc = compute_forecast_features(weather, [T], cfg, log)
        df = df.merge(fc, on="weather_location_id", how="left", suffixes=("", "_fc"))
        df["data_quality_flag"] = df["data_quality_flag"].fillna("NO_WEATHER")
    elif uses_forecast:
        raise SystemExit(f"Model {name} uses forecast features: pass --latest or --issue-time.")
    else:
        df["data_quality_flag"] = "STATIC_ONLY"

    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise RuntimeError(f"Features missing at prediction time: {missing}")
    ok = (df["data_quality_flag"] == "OK") if uses_forecast else df["elevation_m"].notna()
    prob = np.full(len(df), np.nan)
    if ok.any():
        prob[ok.values] = pipe.predict_proba(df.loc[ok, feats])[:, 1]
    pcol = "flood_probability" if uses_forecast else "susceptibility_probability"
    df[pcol] = prob.round(4)
    tier_col = "risk_tier" if uses_forecast else "susceptibility_tier"
    df[tier_col] = [risk_tier(p, cfg["prediction"]["risk_tiers"]) for p in prob]
    df["model_version"] = meta["model_version"]
    df["trained_on"] = meta.get("trained_on", "event_inventory")
    df["validated_for_operational_use"] = bool(cfg["prediction"]["validated"])

    cols = ["segment_id", "road_name", "road_class", pcol, tier_col, "susceptibility_source", "fastflood_max_hazard",
            "fastflood_hazard_share", "fastflood_coverage"]
    if T is not None:
        df["forecast_issue_time"] = T.tz_convert(tz).isoformat()
        df["target_start"] = T.tz_convert(tz).strftime("%Y-%m-%d %H:%M")
        df["target_end"] = (T + pd.Timedelta(hours=24)).tz_convert(tz).strftime("%Y-%m-%d %H:%M EAT")
        cols += ["forecast_issue_time", "target_start", "target_end", "rain_fcst_next_24h_mm", "weather_location_id"]
        if not uses_forecast:
            trig = cfg["training"]["rule_threshold_mm_24h"]
            df["rain_above_threshold"] = df["rain_fcst_next_24h_mm"] >= trig
            cols += ["rain_above_threshold"]
            log.info("Rain columns are context only (threshold %.0f mm / 24 h). Susceptibility is static and does not "
                     "change with the forecast.", trig)
    cols += ["data_quality_flag", "model_version", "trained_on", "validated_for_operational_use"]
    cols = [c for c in cols if c in df.columns]

    stamp = T.strftime("%Y%m%dT%H%MZ") if T is not None else "static"
    out_dir = resolve(cfg["paths"]["outputs"])
    csv_path = ensure_parent(out_dir / f"predictions_{name}_{stamp}.csv")
    df[cols].to_csv(csv_path, index=False)
    seg = gpd.read_file(resolve(cfg["paths"]["interim"]) / "road_segments.gpkg", layer="segments")[["segment_id", "geometry"]]
    gdf = seg.merge(df[cols], on="segment_id").to_crs(cfg["project"]["geographic_crs"])
    gdf.to_file(out_dir / f"predictions_{name}_{stamp}.gpkg", layer="predictions", driver="GPKG")

    fig, ax = plt.subplots(figsize=(8, 8))
    colours = {"Low": "#7aa95c", "Medium": "#f0a030", "High": "#c0392b", "Unavailable": "#bbbbbb"}
    for tier, g in gdf.groupby(tier_col):
        g.plot(ax=ax, color=colours.get(tier, "#888888"), linewidth=0.8, label=tier)
    ax.legend(title="Tier (not validated)" if not cfg["prediction"]["validated"] else "Tier")
    kind = "forecast-conditioned risk" if uses_forecast else "susceptibility"
    when = f" - rain context issued {T.tz_convert(tz):%Y-%m-%d %H:%M} EAT" if T is not None else ""
    ax.set_title(f"Road flood {kind} ({name}){when}", fontsize=10)
    ax.set_axis_off()
    map_path = ensure_parent(out_dir / "maps" / f"risk_{name}_{stamp}.png")
    fig.savefig(map_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s, its .gpkg and %s", csv_path.name, map_path.name)
    log.info("Tier counts: %s", df[tier_col].value_counts().to_dict())
    if not cfg["prediction"]["validated"]:
        log.warning("Outputs are research-prototype values: probabilities are uncalibrated until validated.")


if __name__ == "__main__":
    main()
