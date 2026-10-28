OPTIONAL dated events. Turns the reviewed event inventory into Positive / Negative / Unknown labels and assembles the modelling table with a leakage audit. Commands: labels, assemble.
"""Build flood labels from the reviewed event inventory and assemble the modelling table.

Usage:
    python src/build_labels.py labels       # event_inventory.csv -> labels.parquet (+ label summary, gate G4)
    python src/build_labels.py assemble     # labels + static + forecast features -> modelling_table.parquet

event_inventory.csv columns (one row per observation of a road/location during an event):
    event_id, observation_id, observed_status (flooded | not_flooded | unclear), lat, lon,
    event_start_local, event_end_local, date_only (true/false), spatial_precision_m,
    evidence_type, source_id, source_url, published_time_local, confidence (high|medium|low),
    review_status (reviewed|pending|rejected), notes

Rules (plan section 6):
    Positive : flooded, reviewed, accepted confidence, precise enough, matched to a segment.
    Negative : not_flooded with the same requirements (explicit evidence that the place was checked).
    Unknown  : everything else. Unknown is never converted to 0, and conflicting labels become Unknown.
Time handling: the forecast issue time is event_start minus labels.issue_lead_hours (floored to the hour);
the target window is the 24 hours after it. Date-only reports use local midnight of that date and are
recorded as 'daily formulation' with confidence capped at 'low' (so they are Unknown under default settings).
"""
from __future__ import annotations

import argparse

import geopandas as gpd
import numpy as np
import pandas as pd

from common import ensure_parent, load_settings, resolve, setup_logging

REQUIRED = ["event_id", "observation_id", "observed_status", "lat", "lon", "event_start_local", "date_only",
            "spatial_precision_m", "evidence_type", "source_id", "confidence", "review_status"]


def read_inventory(cfg: dict, log) -> pd.DataFrame:
    path = resolve(cfg["inputs"]["event_inventory"])
    if not path.exists():
        raise FileNotFoundError(f"Event inventory not found: {path}")
    inv = pd.read_csv(path, dtype={"event_id": str, "observation_id": str, "source_id": str})
    missing = [c for c in REQUIRED if c not in inv.columns]
    if missing:
        raise ValueError(f"event_inventory.csv is missing columns: {missing}")
    for c in ["observed_status", "confidence", "review_status"]:
        inv[c] = inv[c].astype(str).str.strip().str.lower()
    inv["date_only"] = inv["date_only"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    log.info("Inventory: %d observations, %d events", len(inv), inv["event_id"].nunique())
    return inv


def derive_times(inv: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    tz = cfg["labels"]["timezone"]
    lead = pd.Timedelta(hours=cfg["labels"]["issue_lead_hours"])
    start = pd.to_datetime(inv["event_start_local"]).dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    issue_local = (start - lead).dt.floor("h")
    issue_local = issue_local.where(~inv["date_only"], start.dt.normalize())  # date-only: local midnight
    inv = inv.copy()
    inv["forecast_issue_time_utc"] = issue_local.dt.tz_convert("UTC")
    inv["target_start_utc"] = inv["forecast_issue_time_utc"]
    inv["target_end_utc"] = inv["forecast_issue_time_utc"] + pd.Timedelta(hours=24)
    return inv


def classify(inv: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    lab = cfg["labels"]
    inv = inv.copy()
    reason = pd.Series("", index=inv.index)
    ok = pd.Series(True, index=inv.index)

    def fail(mask, text):
        nonlocal ok, reason
        mask = mask & ok
        reason = reason.mask(mask, text)
        ok = ok & ~mask

    fail(inv["forecast_issue_time_utc"].isna(), "bad_time")
    fail(~inv["observed_status"].isin(["flooded", "not_flooded"]), "unclear_status")
    if lab["require_reviewed"]:
        fail(inv["review_status"] != "reviewed", "not_reviewed")
    conf = inv["confidence"].where(~inv["date_only"], "low")  # date-only reports capped at low confidence
    inv["label_confidence"] = conf
    fail(~conf.isin(lab["accepted_confidence"]), "confidence_too_low")
    fail(pd.to_numeric(inv["spatial_precision_m"], errors="coerce").fillna(1e9) > lab["max_spatial_precision_m"],
         "spatial_precision_too_low")
    inv["usable"] = ok
    inv["unknown_reason"] = reason
    return inv


def match_segments(inv: pd.DataFrame, cfg: dict, log) -> pd.DataFrame:
    seg = gpd.read_file(resolve(cfg["paths"]["interim"]) / "road_segments.gpkg", layer="segments")[["segment_id", "geometry"]]
    pts = gpd.GeoDataFrame(inv.copy(), geometry=gpd.points_from_xy(inv["lon"], inv["lat"]), crs=cfg["project"]["geographic_crs"])
    pts = pts.to_crs(seg.crs)
    pts["geometry"] = pts.geometry.buffer(cfg["labels"]["match_radius_m"])
    joined = gpd.sjoin(pts, seg, how="left", predicate="intersects").drop(columns=["index_right"])
    n_no_match = joined.loc[joined["usable"] & joined["segment_id"].isna(), "observation_id"].nunique()
    if n_no_match:
        log.warning("%d usable observations had no road segment within %d m (they become Unknown).",
                    n_no_match, cfg["labels"]["match_radius_m"])
    return pd.DataFrame(joined.drop(columns="geometry"))


def build_labels(cfg: dict, log) -> pd.DataFrame:
    inv = classify(derive_times(read_inventory(cfg, log), cfg), cfg)
    m = match_segments(inv, cfg, log)
    m["label_status"] = np.where(m["usable"] & m["segment_id"].notna(),
                                 np.where(m["observed_status"] == "flooded", "positive", "negative"), "unknown")
    m.loc[(m["label_status"] == "unknown") & (m["unknown_reason"] == ""), "unknown_reason"] = "no_segment_match"

    # conflicts: same event, segment and issue time with both positive and negative evidence
    known = m[m["label_status"] != "unknown"]
    conflict = known.groupby(["event_id", "segment_id", "forecast_issue_time_utc"])["label_status"].nunique()
    bad = conflict[conflict > 1].index
    if len(bad):
        key = pd.MultiIndex.from_frame(m[["event_id", "segment_id", "forecast_issue_time_utc"]])
        is_bad = key.isin(bad) & (m["label_status"] != "unknown")
        m.loc[is_bad, "unknown_reason"] = "conflicting_evidence"
        m.loc[is_bad, "label_status"] = "unknown"
        log.warning("%d conflicting label rows reset to Unknown.", int(is_bad.sum()))

    m["flood_label"] = m["label_status"].map({"positive": 1.0, "negative": 0.0})
    m["daily_formulation"] = m["date_only"]
    cols = ["event_id", "observation_id", "segment_id", "forecast_issue_time_utc", "target_start_utc", "target_end_utc",
            "flood_label", "label_status", "unknown_reason", "label_confidence", "evidence_type", "source_id",
            "daily_formulation", "spatial_precision_m"]
    out = m[cols].drop_duplicates(subset=["event_id", "segment_id", "forecast_issue_time_utc", "label_status", "observation_id"])
    path = ensure_parent(resolve(cfg["paths"]["processed"]) / "labels.parquet")
    out.to_parquet(path, index=False)

    summ = out.groupby(["label_status"]).agg(rows=("observation_id", "size"), events=("event_id", "nunique")).reset_index()
    ensure_parent(resolve(cfg["paths"]["outputs"]) / "tables" / "label_summary.csv")
    summ.to_csv(resolve(cfg["paths"]["outputs"]) / "tables" / "label_summary.csv", index=False)
    log.info("Label summary:\n%s", summ.to_string(index=False))
    log.info("Unknown reasons:\n%s", out.loc[out["label_status"] == "unknown", "unknown_reason"].value_counts().to_string())

    # Gate G4
    events_pos = out.loc[out["label_status"] == "positive", "event_id"].nunique()
    if events_pos < cfg["labels"]["min_events_for_modelling"]:
        log.warning("GATE G4 WARNING: only %d events with positive labels (< %d). Treat results as a case study or "
                    "a static susceptibility study, not a validated 24 h forecast.",
                    events_pos, cfg["labels"]["min_events_for_modelling"])
    if (out["label_status"] == "negative").sum() == 0:
        log.warning("GATE G4 WARNING: no confirmed negatives. A classifier cannot be trained without them.")
    return out


def assemble_table(cfg: dict, log) -> pd.DataFrame:
    proc = resolve(cfg["paths"]["processed"])
    labels = pd.read_parquet(proc / "labels.parquet")
    labels = labels[labels["label_status"].isin(["positive", "negative"])].copy()
    static = pd.read_parquet(proc / "static_segment_features.parquet")
    wmap = pd.read_parquet(proc / "segment_weather_map.parquet")
    fcst = pd.read_parquet(proc / "forecast_features.parquet")

    tbl = labels.merge(static, on="segment_id", how="left", validate="m:1")
    hz_path = proc / "hazard_labels.parquet"
    if hz_path.exists():  # modelled FastFlood exposure per segment (NaN where no hazard cells cover the segment)
        hz = pd.read_parquet(hz_path)[["segment_id", "max_hazard_value", "mean_hazard_value", "hazard_pos_fraction",
                                       "hazard_high_fraction", "covered_fraction"]]
        tbl = tbl.merge(hz, on="segment_id", how="left", validate="m:1")
    else:
        log.warning("hazard_labels.parquet not found: FastFlood exposure features are not available.")
    tbl = tbl.merge(wmap, on="segment_id", how="left", validate="m:1")
    tbl = tbl.merge(fcst, on=["weather_location_id", "forecast_issue_time_utc"], how="left", validate="m:1")
    tbl["data_quality_flag"] = tbl["data_quality_flag"].fillna("NO_WEATHER")

    # Gate G5: leakage audit
    used = tbl["max_issue_time_used_utc"]
    assert (used.isna() | (used <= tbl["forecast_issue_time_utc"])).all(), "Leakage: forecast newer than issue time"
    forbidden = {"flood_label", "label_status", "label_confidence", "evidence_type", "source_id"}
    cfg_feats = set(cfg["training"]["static_features"] + cfg["training"]["forecast_features"] + cfg["training"]["categorical_features"])
    assert not (cfg_feats & forbidden), "Label columns configured as predictors"

    # dictionary check (warn only)
    dd_path = resolve(cfg["paths"]["data_dictionary"])
    if dd_path.exists():
        documented = set(pd.read_csv(dd_path)["column"])
        undocumented = [c for c in tbl.columns if c not in documented]
        if undocumented:
            log.warning("Columns missing from data_dictionary.csv: %s", undocumented)
    out = proc / "modelling_table.parquet"
    tbl.to_parquet(out, index=False)
    log.info("Modelling table: %d rows, %d events, %d positives, %d negatives -> %s", len(tbl),
             tbl["event_id"].nunique(), int((tbl["flood_label"] == 1).sum()), int((tbl["flood_label"] == 0).sum()), out)
    log.info("Rows by data_quality_flag: %s", tbl["data_quality_flag"].value_counts().to_dict())
    return tbl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["labels", "assemble"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_settings(args.config)
    log = setup_logging("build_labels", cfg)
    if args.command == "labels":
        build_labels(cfg, log)
    else:
        assemble_table(cfg, log)


if __name__ == "__main__":
    main()
