Precision, recall, F1, PR-AUC, Brier score, detection, calibration and PR plots, and the G6 bootstrap comparison. Use --static for the hazard models.
"""Evaluate held-out predictions: precision/recall/F1, PR-AUC, Brier, event detection, calibration, G6 comparison.

Usage:
    python src/evaluate.py                # event-based results from train.py
    python src/evaluate.py --static       # spatial-block results from train_static.py (hazard dataset)
Outputs: outputs/metrics/[static/]metrics.csv, metrics.json, g6_comparison.csv and PNG plots.
In --static mode the "event" columns refer to spatial blocks (event_id holds the block id).
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,  # noqa: E402
                             precision_recall_curve, precision_score, recall_score)

from common import ensure_parent, load_settings, resolve, setup_logging  # noqa: E402

RULE = "rule_rain24h"


def is_rule(model: str) -> bool:
    return model.startswith("rule_")


def decision_threshold(model: str, cfg: dict) -> float:
    # the rain rule score is rain/threshold, so its natural decision point is 1.0
    return 1.0 if model == RULE else cfg["training"]["decision_threshold"]


def safe_ap(y, s):
    return float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else np.nan


def model_metrics(df: pd.DataFrame, model: str, cfg: dict) -> dict:
    y, s = df["flood_label"].astype(int).values, df["score"].values
    thr = decision_threshold(model, cfg)
    pred = (s >= thr).astype(int)
    row = {
        "model": model, "n_rows": len(df), "n_events": df["event_id"].nunique(),
        "n_positive": int(y.sum()), "prevalence": float(y.mean()), "threshold": thr,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": safe_ap(y, s),
        "brier": float(brier_score_loss(y, np.clip(s, 0, 1))) if not is_rule(model) else np.nan,
    }
    pos_events = df[df["flood_label"] == 1].groupby("event_id")["score"].max()
    row["events_with_positives"] = int(len(pos_events))
    row["event_detection_rate"] = float((pos_events >= thr).mean()) if len(pos_events) else np.nan
    # false-alarm events: events where a confirmed negative scored above the threshold
    neg_events = df[df["flood_label"] == 0].groupby("event_id")["score"].max()
    row["events_with_negatives"] = int(len(neg_events))
    row["event_false_alarm_rate"] = float((neg_events >= thr).mean()) if len(neg_events) else np.nan
    return row


def cluster_bootstrap_diff(oof: pd.DataFrame, a: str, b: str, reps: int, seed: int):
    """Event-level bootstrap of PR-AUC(b) - PR-AUC(a), resampling whole events."""
    da, db = oof[oof["model"] == a], oof[oof["model"] == b]
    events = np.array(sorted(set(da["event_id"]) & set(db["event_id"])))
    if len(events) < 3:
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    ga, gb = {e: g for e, g in da.groupby("event_id")}, {e: g for e, g in db.groupby("event_id")}
    diffs = []
    for _ in range(reps):
        pick = rng.choice(events, size=len(events), replace=True)
        xa = pd.concat([ga[e] for e in pick])
        xb = pd.concat([gb[e] for e in pick])
        apa, apb = safe_ap(xa["flood_label"], xa["score"]), safe_ap(xb["flood_label"], xb["score"])
        if not (np.isnan(apa) or np.isnan(apb)):
            diffs.append(apb - apa)
    if not diffs:
        return np.nan, np.nan, np.nan, 0
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), len(diffs)


def plots(oof: pd.DataFrame, out_dir, cfg: dict, unit: str = "events") -> None:
    models = sorted(oof["model"].unique())
    fig, ax = plt.subplots(figsize=(6, 5))
    for m in models:
        d = oof[oof["model"] == m]
        if d["flood_label"].nunique() == 2:
            p, r, _ = precision_recall_curve(d["flood_label"], d["score"])
            ax.plot(r, p, label=m)
    prev = oof["flood_label"].mean()
    ax.axhline(prev, color="grey", ls="--", label=f"prevalence {prev:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-recall (held-out {unit})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    for m in models:
        if is_rule(m):
            continue
        d = oof[oof["model"] == m]
        if d["flood_label"].nunique() < 2:
            continue
        try:
            frac, mean = calibration_curve(d["flood_label"], np.clip(d["score"], 0, 1),
                                           n_bins=cfg["evaluation"]["calibration_bins"], strategy="quantile")
        except ValueError:
            continue
        ax.plot(mean, frac, marker="o", label=m)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration (held-out {unit})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "calibration.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--static", action="store_true", help="evaluate hazard-dataset models (spatial blocks)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_settings(args.config)
    log = setup_logging("evaluate", cfg)
    out_root = resolve(cfg["paths"]["outputs"])
    oof_path = out_root / "tables" / ("oof_static_predictions.parquet" if args.static else "oof_predictions.parquet")
    oof = pd.read_parquet(oof_path).dropna(subset=["flood_label"])
    out_dir = out_root / "metrics" / ("static" if args.static else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    unit = "spatial blocks" if args.static else "events"

    rows = [model_metrics(g, m, cfg) for m, g in oof.groupby("model")]
    metrics = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics.to_dict(orient="records"), fh, indent=2, default=float)
    log.info("Held-out metrics (%s):\n%s", unit, metrics.round(3).to_string(index=False))

    present = set(oof["model"])
    comps = []
    if args.static:
        # G6s: does a trained static model beat the transparent low-elevation rule on held-out blocks?
        comps = [("rule_low_elevation", m) for m in sorted(present) if m.startswith("H_")]
        if {"H_logreg", "H_rf"} <= present:
            comps.append(("H_logreg", "H_rf"))
    else:
        # G6: does the forecast-conditioned model beat the static model and the rain rule on the same events?
        for fam in ("logreg", "rf"):
            if {f"A_{fam}", f"B_{fam}"} <= present:
                comps.append((f"A_{fam}", f"B_{fam}"))
        comps += [(RULE, b) for b in sorted(present) if b.startswith("B_") and not b.endswith("_haz")]
        # does the modelled FastFlood exposure add value against observed events?
        comps += [(m[:-4], m) for m in sorted(present) if m.endswith("_haz") and m[:-4] in present]
        if "rule_fastflood" in present and "B_rf" in present:
            comps.append(("rule_fastflood", "B_rf"))
    recs = []
    for a, b in comps:
        mean, lo, hi, n = cluster_bootstrap_diff(oof, a, b, cfg["evaluation"]["bootstrap_replicates"],
                                                 cfg["project"]["random_seed"])
        verdict = "no demonstrated improvement" if (np.isnan(lo) or lo <= 0) else "improvement (CI above 0)"
        recs.append({"baseline": a, "candidate": b, "mean_pr_auc_diff": mean, "ci95_low": lo, "ci95_high": hi,
                     "valid_replicates": n, "verdict": verdict})
    comp = pd.DataFrame(recs)
    comp.to_csv(out_dir / "g6_comparison.csv", index=False)
    if not comp.empty:
        log.info("G6 comparison (bootstrap over whole %s, PR-AUC difference):\n%s", unit, comp.round(3).to_string(index=False))
    plots(oof, out_dir, cfg, unit)
    first = oof["model"].iloc[0]
    log.info("Report counts alongside metrics: %d held-out rows, %d %s. Small samples mean wide uncertainty.",
             int((oof["model"] == first).sum()), oof["event_id"].nunique(), unit)


if __name__ == "__main__":
    main()
