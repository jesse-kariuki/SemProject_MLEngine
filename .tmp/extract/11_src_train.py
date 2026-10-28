OPTIONAL dated events. Rain-threshold and FastFlood baselines, static Model A and forecast-conditioned Model B (with and without FastFlood exposure features) using event-based cross-validation. Its make_pipeline function is also used by train_static.py.
"""Train and cross-validate the rule baseline, static model A and forecast-conditioned model B.

Usage:
    python src/train.py                     # leave-events-out CV + final models
    python src/train.py --advanced          # also add gradient-boosted trees (only if labels justify it)
    python src/train.py --split chronological --test-fraction 0.3

Splits are by EVENT (never by random road rows): every row of a storm stays in one fold.
Outputs: outputs/tables/oof_predictions.parquet, models/<name>.joblib, models/<name>.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import ensure_parent, load_settings, resolve, setup_logging


def make_pipeline(kind: str, numeric: list[str], categorical: list[str], seed: int) -> Pipeline:
    num_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if kind == "logreg":
        num_steps.append(("scale", StandardScaler()))
    transformers = [("num", Pipeline(num_steps), numeric)]
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    pre = ColumnTransformer(transformers)
    if kind == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    elif kind == "rf":
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
                                     n_jobs=-1, random_state=seed)
    elif kind == "hgb":
        clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200, random_state=seed)
    else:
        raise ValueError(kind)
    return Pipeline([("pre", pre), ("clf", clf)])


def model_specs(cfg: dict, advanced: bool, table: pd.DataFrame) -> dict:
    tr = cfg["training"]
    static_num = [c for c in tr["static_features"] if c in table.columns]
    fc_num = [c for c in tr["forecast_features"] if c in table.columns]
    cats = [c for c in tr["categorical_features"] if c in table.columns]
    specs = {
        "A_logreg": ("logreg", static_num, cats),
        "A_rf": ("rf", static_num, cats),
        "B_logreg": ("logreg", static_num + fc_num, cats),
        "B_rf": ("rf", static_num + fc_num, cats),
    }
    haz = [c for c in tr.get("hazard_features", []) if c in table.columns and table[c].notna().any()]
    if tr.get("use_hazard_features_in_event_models", True) and haz:
        # does adding the modelled FastFlood exposure help against OBSERVED events?
        for fam in ("logreg", "rf"):
            specs[f"A_{fam}_haz"] = (fam, static_num + haz, cats)
            specs[f"B_{fam}_haz"] = (fam, static_num + fc_num + haz, cats)
    if advanced:
        specs["B_hgb"] = ("hgb", static_num + fc_num, cats)
    return specs


def split_folds(table: pd.DataFrame, args, cfg: dict, log):
    events = table.groupby("event_id")["forecast_issue_time_utc"].min().sort_values()
    if len(events) < 2:
        raise SystemExit("Need at least 2 events for an event-based split (gate G4).")
    if args.split == "chronological":
        n_test = max(1, int(round(len(events) * args.test_fraction)))
        test_events = set(events.index[-n_test:])
        yield 0, table.index[~table["event_id"].isin(test_events)], table.index[table["event_id"].isin(test_events)]
        return
    n_splits = min(cfg["training"]["n_splits"], len(events))
    gkf = GroupKFold(n_splits=n_splits)
    for k, (tr, te) in enumerate(gkf.split(table, table["flood_label"], groups=table["event_id"])):
        yield k, table.index[tr], table.index[te]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["group_kfold", "chronological"], default="group_kfold")
    ap.add_argument("--test-fraction", type=float, default=0.3)
    ap.add_argument("--advanced", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_settings(args.config)
    log = setup_logging("train", cfg)
    seed = cfg["project"]["random_seed"]

    table = pd.read_parquet(resolve(cfg["paths"]["processed"]) / "modelling_table.parquet")
    table = table[table["data_quality_flag"] == "OK"].reset_index(drop=True)  # complete weather only
    if table["flood_label"].nunique() < 2:
        raise SystemExit("Both positive and confirmed-negative labels are required to train.")
    log.info("Training rows: %d (%d events, %.1f%% positive)", len(table), table["event_id"].nunique(),
             100 * table["flood_label"].mean())
    specs = model_specs(cfg, args.advanced, table)
    rule_thr = cfg["training"]["rule_threshold_mm_24h"]

    oof = []
    for fold, tr_idx, te_idx in split_folds(table, args, cfg, log):
        train, test = table.loc[tr_idx], table.loc[te_idx]
        base = test[["event_id", "segment_id", "forecast_issue_time_utc", "flood_label"]].copy()
        base["fold"] = fold
        # Baseline 0: transparent rule - the score is simply forecast 24 h rainfall (scaled by threshold)
        r = base.copy()
        r["model"] = "rule_rain24h"
        r["score"] = (test["rain_fcst_next_24h_mm"].fillna(0) / rule_thr).values
        oof.append(r)
        if "hazard_pos_fraction" in test.columns:
            # Baseline: the modelled FastFlood hazard used directly (share of segment in hazard cells; 0 if uncovered)
            h = base.copy()
            h["model"] = "rule_fastflood"
            h["score"] = test["hazard_pos_fraction"].fillna(0).values
            oof.append(h)
        if train["flood_label"].nunique() < 2:
            log.warning("Fold %d: training data has one class only - model rows skipped.", fold)
            continue
        for name, (kind, num, cat) in specs.items():
            pipe = make_pipeline(kind, num, cat, seed).fit(train[num + cat], train["flood_label"].astype(int))
            m = base.copy()
            m["model"] = name
            m["score"] = pipe.predict_proba(test[num + cat])[:, 1]
            oof.append(m)
        log.info("Fold %d done: %d train rows, %d test rows, test events %s", fold, len(train), len(test),
                 sorted(test["event_id"].unique()))

    oof = pd.concat(oof, ignore_index=True)
    out = ensure_parent(resolve(cfg["paths"]["outputs"]) / "tables" / "oof_predictions.parquet")
    oof.to_parquet(out, index=False)
    log.info("Held-out predictions -> %s", out)

    # Final models on all labelled data (for the prototype predictions)
    models_dir = resolve(cfg["paths"]["models"])
    models_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    for name, (kind, num, cat) in specs.items():
        pipe = make_pipeline(kind, num, cat, seed).fit(table[num + cat], table["flood_label"].astype(int))
        version = f"{name.lower()}_{cfg['training']['model_version_prefix']}"
        joblib.dump(pipe, models_dir / f"{name}.joblib")
        meta = {"name": name, "model_version": version, "kind": kind, "numeric_features": num,
                "categorical_features": cat, "trained_utc": stamp, "n_rows": int(len(table)),
                "n_events": int(table["event_id"].nunique()), "positive_rate": float(table["flood_label"].mean()),
                "validated": False}
        with open(models_dir / f"{name}.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        log.info("Saved %s (%s)", name, version)


if __name__ == "__main__":
    main()
