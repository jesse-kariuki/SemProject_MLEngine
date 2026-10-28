PRIMARY. Trains the static susceptibility models H_logreg and H_rf (emulators of FastFlood) with leave-spatial-blocks-out validation, against a low-elevation rule baseline.
"""Train and validate the static susceptibility models on the hazard-dataset labels (primary source).

Usage:
    python src/train_static.py
    python src/train_static.py --advanced        # adds gradient-boosted trees

Validation is SPATIAL: segments are grouped into blocks (hazard.block_size_m) and whole blocks are held out,
because neighbouring road segments share the same hazard cells and would otherwise leak between train and test.
Only static predictors are used (never coordinates, never labels). Models: H_logreg, H_rf (and H_hgb).
Baseline: rule_low_elevation (lower elevation than the training median -> predicted hazard).
Outputs: data/processed/hazard_modelling_table.parquet, outputs/tables/oof_static_predictions.parquet,
         outputs/tables/feature_importance_H_rf.csv, models/H_*.joblib + .json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from common import ensure_parent, load_settings, resolve, setup_logging
from train import make_pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--advanced", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_settings(args.config)
    log = setup_logging("train_static", cfg)
    seed = cfg["project"]["random_seed"]
    proc = resolve(cfg["paths"]["processed"])

    labels = pd.read_parquet(proc / "hazard_labels.parquet")
    static = pd.read_parquet(proc / "static_segment_features.parquet")
    labels = labels[labels["label_status"].isin(["positive", "negative"])]
    tbl = labels.merge(static.drop(columns=["segment_length_m"], errors="ignore"), on="segment_id", how="left",
                       validate="1:1").reset_index(drop=True)
    tbl["flood_label"] = tbl["hazard_label"]
    tbl.to_parquet(proc / "hazard_modelling_table.parquet", index=False)
    if tbl["flood_label"].nunique() < 2:
        raise SystemExit("Both positive and negative labelled segments are required (gate G4h).")
    n_blocks = tbl["spatial_block"].nunique()
    log.warning("The hazard labels are FastFlood MODEL output derived from DEM/land cover/soil. H_* models learn to "
                "reproduce that model from similar inputs (an emulator). Good held-out scores show agreement with "
                "FastFlood, NOT skill at predicting real floods. For exposure use the hazard columns directly "
                "(max_hazard_value, hazard_pos_fraction).")
    log.info("Training table: %d segments, %d spatial blocks, %.1f%% positive", len(tbl), n_blocks,
             100 * tbl["flood_label"].mean())
    if n_blocks < 3:
        raise SystemExit("Need at least 3 spatial blocks; reduce hazard.block_size_m.")

    tr = cfg["training"]
    num = [c for c in tr["static_features"] if c in tbl.columns and tbl[c].notna().any()]
    dropped = [c for c in tr["static_features"] if c not in num]
    if dropped:
        log.warning("Static features dropped (absent or entirely missing): %s", dropped)
    cat = [c for c in tr["categorical_features"] if c in tbl.columns]
    specs = {"H_logreg": "logreg", "H_rf": "rf"}
    if args.advanced:
        specs["H_hgb"] = "hgb"

    n_splits = min(tr["n_splits"], n_blocks)
    oof = []
    for fold, (tr_i, te_i) in enumerate(GroupKFold(n_splits=n_splits).split(tbl, tbl["flood_label"], groups=tbl["spatial_block"])):
        train, test = tbl.iloc[tr_i], tbl.iloc[te_i]
        base = pd.DataFrame({"event_id": test["spatial_block"].values, "segment_id": test["segment_id"].values,
                             "flood_label": test["flood_label"].values, "fold": fold})
        ref = np.sort(train["elevation_m"].dropna().values)
        r = base.copy()
        r["model"] = "rule_low_elevation"
        elev = test["elevation_m"].values
        r["score"] = np.where(np.isnan(elev), 0.5, 1.0 - np.searchsorted(ref, np.nan_to_num(elev)) / max(1, len(ref)))
        oof.append(r)
        if train["flood_label"].nunique() < 2:
            log.warning("Fold %d: one class in training data - model rows skipped.", fold)
            continue
        for name, kind in specs.items():
            pipe = make_pipeline(kind, num, cat, seed).fit(train[num + cat], train["flood_label"].astype(int))
            m = base.copy()
            m["model"] = name
            m["score"] = pipe.predict_proba(test[num + cat])[:, 1]
            oof.append(m)
        log.info("Fold %d: %d train / %d test segments, %d test blocks", fold, len(train), len(test), test["spatial_block"].nunique())
    oof = pd.concat(oof, ignore_index=True)
    oof.to_parquet(ensure_parent(resolve(cfg["paths"]["outputs"]) / "tables" / "oof_static_predictions.parquet"), index=False)

    models_dir = resolve(cfg["paths"]["models"])
    models_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    for name, kind in specs.items():
        pipe = make_pipeline(kind, num, cat, seed).fit(tbl[num + cat], tbl["flood_label"].astype(int))
        joblib.dump(pipe, models_dir / f"{name}.joblib")
        meta = {"name": name, "model_version": f"{name.lower()}_{tr['model_version_prefix']}", "kind": kind,
                "trained_on": "hazard_dataset", "target": "static_susceptibility",
                "numeric_features": num, "categorical_features": cat, "trained_utc": stamp,
                "n_rows": int(len(tbl)), "n_blocks": int(n_blocks), "positive_rate": float(tbl["flood_label"].mean()),
                "validated": False}
        with open(models_dir / f"{name}.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        log.info("Saved %s", name)
        if kind == "rf":
            names = pipe["pre"].get_feature_names_out()
            imp = pd.Series(pipe["clf"].feature_importances_, index=names).sort_values(ascending=False)
            imp.rename("importance").to_csv(resolve(cfg["paths"]["outputs"]) / "tables" / f"feature_importance_{name}.csv")
            log.info("Top features (impurity importance, interpret with care):\n%s", imp.head(8).round(3).to_string())
    log.info("Next: python src/evaluate.py --static")


if __name__ == "__main__":
    main()
