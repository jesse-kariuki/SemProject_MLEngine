"""Build pipeline.ipynb: the end-to-end run of the primary (static hazard) pipeline,
plus the four assignment sections (approach, cleaning/EDA, modelling, validation)."""
import ast
import json
from pathlib import Path


def md(src, cid):
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src, cid):
    return {"cell_type": "code", "execution_count": None, "id": cid, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


cells = []

cells.append(md('''# Nairobi Road Flood Susceptibility - end-to-end pipeline

Runs the **primary workflow** of the project guide (revision 5) on the real Nairobi data, in order, and
finishes with the trained models, their held-out metrics and the static susceptibility map.

| # | Step | Command | Produces |
|---|------|---------|----------|
| 1 | GIS inventory | `prepare_gis.py inventory` | `outputs/tables/asset_inventory.csv` (gates G2/G3) |
| 2 | GIS preparation | `prepare_gis.py all` | boundary, roads, road segments, DEM + land-cover clips |
| 3 | Weather grid | `fetch_openmeteo.py grid` | `config/grid_points.gpkg` |
| 4 | Static features | `build_features.py static` | `static_segment_features.parquet`, `segment_weather_map.parquet` |
| 5 | **Primary labels** | `build_hazard_labels.py` | `hazard_labels.parquet` (gate G4h) |
| 6 | Coverage check | `check_coverage.py` | coverage report, map, uncovered area/segments (G2/G4h) |
| 7 | **Training** | `train_static.py` | `models/H_*.joblib` + `.json`, OOF predictions, importances |
| 8 | Evaluation | `evaluate.py --static` | `outputs/metrics/static/*` (gate G6) |
| 9 | Prediction | `predict.py` | static susceptibility map + per-road CSV/GPKG |
| 10 | Optional weather context | `fetch_openmeteo.py fetch --mode live --limit 8` + `predict.py --latest` | + 24 h rain context columns |

The four report sections (ML approach, data extraction and cleaning with EDA, modelling approach and
validation, further validation of the base model) are interleaved with the steps they describe, so the
numbers in them come from cells that actually run against the data.

> **Research prototype only.** These are *susceptibility* values derived from modelled FastFlood hazard cells
> (no timestamps), not a forecast and not a warning service. Unknown labels are never treated as negatives,
> validation holds out whole spatial blocks, and the tier thresholds are illustrative and not validated.
''', "intro"))

cells.append(md('''## 1. Machine-learning approach and algorithms

**Task.** Supervised **binary classification** on a tabular, spatially referenced dataset: each row is one
~100 m road segment, the target is `hazard_label` (1 = the segment lies mostly inside modelled flood-hazard
cells, 0 = it does not), and the output is a calibrated-style score in [0, 1] per segment that is mapped to
an illustrative Low / Medium / High tier and written to CSV, GeoPackage and a risk map.

**Why this family of algorithms.** The data are mixed (continuous terrain/land-cover fractions plus a
categorical road class), small-to-medium (72,882 labelled rows, 11 predictors), strongly imbalanced (19.1%
positive) and strongly nonlinear in effect (the positive rate does not move monotonically with elevation -
see the EDA section). Tree ensembles handle all of that without scaling, missing-value imputation or
feature engineering, and give importances for the write-up; a regularised linear model is the necessary
benchmark for "does the nonlinearity actually buy anything?". Coordinates are deliberately **not** used as
predictors, so the model cannot learn "this location floods" shortcuts.

| Algorithm | Role | Configuration | Validation |
|-----------|------|---------------|------------|
| **Random Forest** (`H_rf`) | **base model** | 300 trees, `min_samples_leaf=3`, `class_weight=balanced_subsample`, `max_features=sqrt` | 5-fold grouped CV over 171 spatial blocks |
| **Logistic Regression** (`H_logreg`) | regularised linear baseline | L2 penalty, standardised numerics, one-hot road class, `class_weight=balanced`, `max_iter=2000` | same folds |
| **Rule baseline** (`rule_low_elevation`) | transparent floor | "elevation below a threshold" hand rule | same folds |
| **HistGradientBoosting** (`H_gbdt`, optional) | alternative model, same process | `max_depth=4`, `learning_rate=0.05`, `max_iter=200` | same folds, run with `train_static.py --advanced` / `train.py --advanced` |
| **A_rf / B_rf / B_rf_haz** (optional) | forecast-conditioned variants, only with dated flood events | same RF settings, event-grouped CV | leave-events-out CV + event bootstrap (guide steps 11-17) |
| Rain-rule baseline (optional) | forecasting benchmark, 24 h rain >= 20 mm | hand rule | event bootstrap |

**Not used, and why:** K-means / clustering and the elbow method do not apply - there are no unlabelled
groups to discover, the target is already labelled and the task is classification, not segmentation;
deep learning is unjustified at 72,882 rows of tabular data (it would only add variance and training cost);
SVM/k-NN are weaker on mixed-scale, imbalanced, nonlinear tabular data and offer no interpretability.

### Spreadsheet entry (group row)

| Field | Value |
|-------|-------|
| Application | Flood susceptibility mapping for the Nairobi road network (per-segment risk layer) |
| ML approach | Supervised binary classification + rule baseline benchmarking; spatial grouped cross-validation |
| Algorithms | Random Forest (base model), L2-regularised Logistic Regression, Gradient Boosting (optional), hand-rule baseline |
| Input | 11 predictors: elevation, slope, local relief, land-cover fractions (built-up/vegetation/water), distance to river, segment length, bridge/tunnel flags, road class |
| Output | Flood-susceptibility probability (0-1) per road segment -> Low/Medium/High tier, CSV + GeoPackage + risk map |
| Validation | 5-fold GroupKFold over 2 km spatial blocks, PR-AUC / precision / recall / Brier, block bootstrap CI (1,000 replicates) |
''', "sec1-algorithms"))

cells.append(code('''"""Step 0 - environment and helpers."""
import os
import subprocess
import sys
import time
from pathlib import Path

# Run the notebook from the project root (the folder that contains config/ and src/).
ROOT = Path(os.environ.get("FLOOD_PROJECT_ROOT", Path.cwd())).resolve()
if not (ROOT / "config" / "settings.yaml").exists():
    raise SystemExit(f"config/settings.yaml not found under {ROOT} - start Jupyter in the project root.")

PY = sys.executable
TIMINGS = []


def run(*cmd, optional=False):
    """Run one pipeline step as a subprocess (each script keeps its own log and its own settings)."""
    argv = [PY, *cmd]
    print("\\n>>> " + " ".join(argv), flush=True)
    started = time.time()
    code = subprocess.run(argv, cwd=ROOT).returncode
    elapsed = time.time() - started
    TIMINGS.append((" ".join(cmd), code, elapsed))
    print("--- exit=%d in %s s" % (code, format(elapsed, ",.1f")))
    if code and not optional:
        raise SystemExit("Step failed: " + " ".join(argv) + " - see logs/ for the reason.")
    return code


print("Project root:", ROOT)
print("Python      :", PY)''', "setup"))

cells.append(md('''## Steps 1-2 - GIS preparation and boundary check

Verifies the assets (gate G2/G3), checks the study boundary, clips the roads to the boundary and cuts them
into stable ~100 m segments, then warps the DEM (bilinear) and the land cover (nearest neighbour) onto the
working CRS grid. Original downloads are never modified.

The study boundary is the OSM `admin_level4` polygon named *Nairobi* (`inputs.boundary_file` together with
`boundary_name_field` / `boundary_name_value` in `config/settings.yaml`) - the real county, ~696 km2. The
guide's boundary check rejects `data/raw/boundary/NairobiEntire.geojson`, which is a 1489 km2 rectangle.
''', "gis"))

cells.append(code('''# Steps 1-2: GIS preparation
run("src/prepare_gis.py", "inventory")   # metadata + boundary/overlap QA
run("src/prepare_gis.py", "all")         # boundary, roads, segments, raster clips, study_area.geojson''', "gis-run"))

cells.append(md('''## Steps 3-4 - weather grid and static features

The grid step only builds weather points from the study area (no internet needed). Static features are
elevation, slope, local relief, land-cover class fractions, distance to river and road attributes - **no
coordinates and no hazard values** are used as predictors.
''', "features"))

cells.append(code('''# Step 3: weather points (must run before step 4 so segments can be linked to them)
run("src/fetch_openmeteo.py", "grid")
# Step 4: static segment features
run("src/build_features.py", "static")''', "features-run"))

cells.append(md('''## Steps 5-6 - primary labels from the FastFlood hazard dataset, and coverage

`build_hazard_labels.py` cuts every segment against the hazard cells and labels it Positive / Negative /
Unknown from the length of the segment inside hazard cells (gate G4h). `check_coverage.py` then measures how
much of the boundary and the road network the cells actually cover, exports the uncovered area and segments
for QGIS, and reconciles its counts with the label script.
''', "labels"))

cells.append(code('''# Steps 5-6: hazard labels and coverage diagnostics
run("src/build_hazard_labels.py")   # -> data/processed/hazard_labels.parquet, outputs/tables/hazard_label_summary.csv
run("src/check_coverage.py")        # -> outputs/tables/coverage_report.csv, outputs/maps/coverage.png

import pandas as pd
label_summary = pd.read_csv(ROOT / "outputs" / "tables" / "hazard_label_summary.csv")
print("Segment label summary:")
print(label_summary.to_string(index=False))''', "labels-run"))

cells.append(md('''## 2. Data extraction and cleaning - from raw downloads to the modelling table

Every number below is written by the steps that just ran; the full detail is in `logs/prepare_gis.log`,
`logs/build_features.log`, `logs/build_hazard_labels.log` and `logs/check_coverage.log`.

### 2.1 What was extracted

| Source | File | Raw shape / size | Use |
|--------|------|------------------|-----|
| Study boundary | OSM `admin_level4` polygon *Nairobi* | 695.9 km2, 1,881 vertices, fills 45.2% of its bounding box | clip extent, area gates |
| Roads | Geofabrik/HOT OSM extract, LINE layer | **739,195** features, CRS EPSG:4326 | network to segment |
| Terrain | Copernicus DEM `S02E036.tif` (1 arc-sec, int16, nodata -32768) | 3,601 x 3,601 px, covers 88% of the boundary | elevation, slope, relief |
| Land cover | ESA-style LCM-10 map, 2020 (uint8, nodata 255) | 36,000 x 36,000 px (~10 m), covers 100% of the boundary | class fractions |
| **Labels** | SPACE4ALL **FastFlood** output `flood_hazard.csv` | **110,463** rows, one 100 x 100 m cell per row, classes 0/1/2 | target + exposure features |

The boundary gate matters: the raw `data/raw/boundary/NairobiEntire.geojson` is a 1,489 km2 rectangle (it
fills 99.8% of its own bounding box and is 114% above the expected county area), so it is rejected and only
used for comparison. The DEM tile also does not fully cover the county - the inventory records 88% coverage
and the clipped product `dem_clip.tif` (1,668 x 1,082) has 75.7% valid pixels, which is exactly where the
missing elevation values in 2.3 come from.

### 2.2 Cleaning steps and what each one removed

| # | Cleaning action | Effect on the data |
|---|-----------------|--------------------|
| 1 | Harmonise CRS | all vector layers reprojected to the working CRS; DEM warped **bilinear**, land cover **nearest neighbour** (class codes must not be interpolated) |
| 2 | Road class filter (`roads.keep_fclass`) | 739,195 -> **481,190** features; footways, paths, steps, cycleways and pedestrian-only ways are dropped because they are not part of the drivable network |
| 3 | Clip to the study boundary | 481,190 -> **47,904** lines (everything outside Nairobi County removed) |
| 4 | De-duplicate on (`osm_id`, `geometry`) | **0** duplicates found - the extract is already clean here |
| 5 | Segment the lines | **83,596** segments; length median **95.7 m**, p95 120.8 m, capped at 150 m, minimum 10 m (shorter dropped) |
| 6 | Sample rasters per segment | nearest-pixel DEM sampling and a land-cover focal window give elevation **1,480-2,186 m**, mean slope 4.4 deg, class codes {10,20,30,40,50,80,90,100} collapsed to built-up / vegetation / water / other fractions |
| 7 | Distance to river | waterways filtered to real river classes and clipped to 5 km of the boundary: median **492 m**, max 3,858 m |
| 8 | **Missing values** | `elevation_m` **8.5%**, `slope_deg` 8.5%, `local_relief_m` 8.3% missing (DEM edge/void pixels); `surface` **100%** missing -> column dropped; `road_name` **76.2%** missing -> kept for display only, excluded from predictors. Remaining NaNs are imputed **inside the CV pipeline** (median fit on the training fold only), so no test-fold information leaks in |
| 9 | Hazard cell QA | invalid latitude/longitude/hazard rows dropped; conflicting duplicate cells dropped; 52,650 cells outside the study area ignored (57,813 kept); cell overlap of **0.71%** of cell area flagged in the log |
| 10 | **Labelling rules** | a cell counts as hazardous when `flood_hazard >= 1` (depth >= 10 cm); a segment is **Positive** when >= 50% of its covered length sits in hazardous cells, **Negative** when covered >= 50% but below that share, otherwise **Unknown** (`mixed_policy: exclude`); overlaps shorter than 0.5 m do not count as coverage; `0` is treated as a *modelled* negative, not as observed dry ground |
| 11 | **Unknown stays Unknown** | 10,714 segments are excluded from training rather than forced to 0: `mixed_cells` 6,083, `no_hazard_cells` 4,587 (outside the FastFlood footprint), `insufficient_coverage` 44 |
| 12 | Assemble the modelling table | label + features + 2 km `spatial_block` id joined on `segment_id` |

### 2.3 The final dataset used for modelling

- **72,882 labelled segments** (58,979 negative / 13,903 positive = **19.1% prevalence**) over
  **4,932.9 km / 1,192.5 km** of road, plus **10,714 unknown** segments (1,020.0 km) held out entirely.
- **11 predictors**: elevation, slope, local relief, built-up / vegetation / water fractions, distance to
  river, segment length, bridge flag, tunnel flag, road class (categorical). Coordinates, `road_name`,
  `surface` and every hazard column are excluded from the predictors - hazard columns would leak the target.
- **171 spatial blocks** of 2,000 m, used as the grouping unit for cross-validation.
''', "sec2-cleaning"))

cells.append(code('''# Section 2 (EDA): explore the raw inputs and profile the cleaned table. Re-runnable at any time.
import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:                      # plain-Python runner
    display = print

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
INTERIM = ROOT / "data" / "interim"
OUT = ROOT / "outputs"

print("== Raw asset inventory (extraction: CRS, resolution, boundary coverage) ==")
display(pd.read_csv(OUT / "tables" / "asset_inventory.csv"))

print("== Raw hazard cells (label source) ==")
hz = pd.read_csv(RAW / "flood_hazard" / "flood_hazard.csv")
print("rows:", f"{len(hz):,}", "| columns:", list(hz.columns))
print(hz["flood_hazard"].value_counts().sort_index().to_string())

EXPLORE_RAW_ROADS = True     # one pass over the raw OSM extract (a few seconds); set False to skip
static = pd.read_parquet(PROC / "static_segment_features.parquet")
if EXPLORE_RAW_ROADS:
    try:
        import geopandas as gpd
        raw_roads = gpd.read_file(RAW / "osm_shapefiles" / "roads" / "gis_osm_roads_free_1.shp")
        print("raw road features:", f"{len(raw_roads):,}", "| CRS:", raw_roads.crs)
        print("fclass before the keep-filter (top 8):")
        print(raw_roads["fclass"].value_counts().head(8).to_string())
        dropped = sorted(set(raw_roads["fclass"].dropna()) - set(static["road_class"].dropna()))
        print("classes removed by roads.keep_fclass:", dropped)
        del raw_roads
    except Exception as exc:             # pragma: no cover - never block the notebook on this cell
        print("raw road exploration skipped:", exc)

print("== Cleaned segment table (after steps 1-4) ==")
print("shape:", static.shape)
profile = pd.DataFrame({
    "dtype": static.dtypes.astype(str),
    "missing_%": (100 * static.isna().mean()).round(2),
    "n_unique": static.nunique(),
})
display(profile.sort_values("missing_%", ascending=False).head(12))
print("fully empty columns:", [c for c in static if static[c].isna().all()])
print("duplicate segment_id:", int(static["segment_id"].duplicated().sum()))

print("== Labels (step 5) ==")
lab = pd.read_parquet(PROC / "hazard_labels.parquet")
summary = lab.groupby("label_status").agg(
    segments=("segment_id", "size"),
    km=("segment_length_m", lambda s: round(float(s.sum()) / 1000, 1)),
)
display(summary)
print("unknown reasons:")
print(lab.loc[lab["label_status"] == "unknown", "unknown_reason"].value_counts().to_string())
n_labelled = int(lab["label_status"].isin(["positive", "negative"]).sum())
print("prevalence among labelled:", round(float((lab["label_status"] == "positive").sum()) / n_labelled, 4))

print("== Feature vs target (labelled segments) ==")
d = lab.merge(static, on="segment_id", how="left", suffixes=("", "_seg"))
d = d[d["label_status"].isin(["positive", "negative"])].copy()
d["y"] = (d["label_status"] == "positive").astype(int)
feats = ["elevation_m", "slope_deg", "local_relief_m", "distance_to_river_m",
         "builtup_fraction", "vegetation_fraction", "water_fraction"]
rows = []
for c in feats:
    rows.append({
        "feature": c,
        "median_neg": round(float(d.loc[d["y"] == 0, c].median()), 2),
        "median_pos": round(float(d.loc[d["y"] == 1, c].median()), 2),
        "point_biserial_r": round(float(d[c].corr(d["y"])), 3),
        "missing_%": round(float(100 * d[c].isna().mean()), 2),
    })
display(pd.DataFrame(rows))

print("== Positive rate by road class ==")
by_class = d.groupby("road_class")["y"].agg(n="size", positive_rate="mean").sort_values("positive_rate", ascending=False)
display(by_class.round(3))

print("== Positive rate by elevation quintile ==")
d["elev_quintile"] = pd.qcut(d["elevation_m"], 5, duplicates="drop")
display(d.groupby("elev_quintile", observed=True)["y"].agg(n="size", positive_rate="mean").round(3))

print("== Feature correlation with the target (top 10 by |r|) ==")
num = [c for c in feats + ["segment_length_m"] if c in d.columns]
corrs = {c: float(d[c].corr(d["y"])) for c in num}
for name, r in sorted(corrs.items(), key=lambda kv: -abs(kv[1])):
    print(f"  {name:<24} r = {r:+.3f}")''', "sec2-eda"))

cells.append(md('''### Observations from the EDA (all derived from the cells above)

1. **The target is imbalanced and partly unlabelled.** 13,903 of 72,882 labelled segments are positive
   (**19.1%**), and a further 10,714 segments (12.8% of the network, 1,020 km) are *Unknown* because they are
   outside the FastFlood footprint, covered by mixed cells, or too poorly covered to call. Consequences:
   accuracy is meaningless as a headline metric (a trivial all-negative classifier already scores 80.9%),
   so we report **PR-AUC against the 0.191 prevalence baseline**, the models get `class_weight="balanced"`,
   and Unknown rows are dropped rather than relabelled 0.
2. **Proximity to water and land cover carry the signal, not raw elevation.** Positive segments sit closer to
   a river (median **284.5 m** vs **500.4 m** for negatives) and are markedly less built-up (built-up fraction
   **0.84 vs 0.96**, vegetation **0.16 vs 0.03**; point-biserial r = -0.13 / +0.13), while elevation is nearly
   uncorrelated with the target (r = **-0.05**) and slope only mildly positive (r = +0.10). The trained model
   agrees: distance to river is the top feature (importance 0.365), then elevation (0.158), relief (0.104) and
   slope (0.091).
3. **The road hierarchy is unevenly exposed.** Positive rates run 0.364 for `primary`, 0.304 for `motorway` and
   0.249 for `trunk` down to 0.167 for `service` and 0.065 for `living_street` (sample sizes 1,067 / 349 /
   1,663 / 25,734 / 168 segments). Arterials carry more traffic but
   also follow valleys and river crossings, so the class effect is a genuine geographic pattern - which is why
   `road_class` stays as a predictor.
4. **Elevation alone cannot separate the classes.** Positive rate falls only from 0.215 (lowest quintile) to
   0.176 (highest) and is non-monotonic in between, i.e. there is no threshold that splits the data. That is
   precisely why the transparent `rule_low_elevation` baseline reaches only **PR-AUC 0.257** while the
   Random Forest reaches **0.578** - the extra 0.32 comes from feature interactions (river distance x
   land cover x road class), not from a single cut-off.
5. **Missingness is structured, not random, and is handled inside the pipeline.** DEM-derived features are
   missing for **8.5%** of segments (tile edges and voids over the county boundary, consistent with the 75.7%
   valid pixels of `dem_clip.tif`), `surface` is missing for **100%** and `road_name` for **76.2%**. The first
   two are excluded/imputed - `surface` is dropped as a column, the DEM features are median-imputed **within
   each training fold**, so no held-out fold information can leak into training.
''', "sec2-observations"))

cells.append(md('''## Steps 7-8 - training and evaluation

`train_static.py` fits `H_logreg` and `H_rf` on the labelled segments and validates them by **holding out
whole 2 km spatial blocks** (`GroupKFold`), against a transparent low-elevation rule baseline.
`evaluate.py --static` reports PR-AUC, precision/recall, Brier score and calibration on the held-out blocks
and bootstraps the PR-AUC difference over blocks (gate G6).
''', "train"))

cells.append(code('''# Steps 7-8: train the static susceptibility models, then evaluate them
run("src/train_static.py")          # -> models/H_logreg.joblib, models/H_rf.joblib (+ .json), OOF + importances
run("src/evaluate.py", "--static")  # -> outputs/metrics/static/metrics.csv, g6_comparison.csv, PR + calibration plots''', "train-run"))

cells.append(md('''## 3. Modelling approach and why the results can be trusted

**Base model: Random Forest (`H_rf`).** Each fit is a scikit-learn `Pipeline` (a `ColumnTransformer`) so the
preprocessing is part of the model and is re-fit on every training fold:

1. *numeric* (10 columns) - median imputation, then, for the linear baseline only, standard scaling;
2. *categorical* (`road_class`) - one-hot encoding with `handle_unknown="ignore"`;
3. *classifier* - `RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
   class_weight="balanced_subsample", max_features="sqrt", random_state=seed)`.

The RF itself is regularised by `min_samples_leaf=3` (a leaf needs 3 segments, which stops single outliers
defining a rule), `max_features="sqrt"` (each tree sees a random feature subset, decorrelating the trees) and
the fixed depth policy that comes with it. The linear baseline is regularised explicitly - **L2 (ridge)
penalty** on the coefficients after standardisation, with `class_weight="balanced"` for the 19.1% positives.
The optional gradient-boosted model is regularised by depth and shrinkage (`max_depth=4`,
`learning_rate=0.05`, 200 iterations with early stopping). No hyperparameter search is run, so the reported
scores are not tuned-inflated.

**Validation design - cross-validation with grouping, not a random split.**

- `GroupKFold(n_splits=5)` over **171 spatial blocks of 2,000 m**: every segment of a block is either all in
  the training fold or all in the test fold (per fold: ~58,305 train / 14,577 test rows over 34-35 held-out
  blocks). A random split would put neighbouring segments - which share the same FastFlood cells, the same
  DEM pixels and the same land-cover window - on both sides and would inflate every score; this is the
  standard leakage control for spatial data and it is the closest analogue of "k-fold CV for classification"
  in a spatially autocorrelated dataset.
- Out-of-fold (OOF) predictions for all 72,882 rows are collected across the folds and scored once, so every
  reported metric comes from a segment the model never saw.
- **Metrics:** PR-AUC (primary, imbalance-robust) alongside precision/recall/F1 at the 0.5 decision
  threshold, **Brier score** and a reliability/calibration table (5 bins), always printed next to the
  **prevalence baseline (0.191)** and the hand rule.
- **Repeated/nested checks:** the block bootstrap below resamples whole blocks (1,000 replicates) to put a
  95% confidence interval on every PR-AUC difference, which is the honest version of re-running the split.

**On the "elbow method for K-means" clause:** it does not apply here. The target is labelled
(segment is / is not inside hazard cells), so the assignment's clustering guidance (elbow, silhouette)
has no object; the applicable checks for a classification task are grouped cross-validation, a regularised
linear baseline, a naive/prevalence baseline and calibration - all of which are used above.
''', "sec3-model"))

cells.append(md('''## 4. Further validation of the base model - competing models, same process

The recommended pattern (as in figure 6 of the paper) is to push several models through the **identical**
data, folds and metrics, then compare them with an uncertainty interval rather than a single number. That is
what `evaluate.py --static` does:

1. same 72,882 rows, same 171 blocks, same OOF predictions for every candidate;
2. `rule_low_elevation` (hand rule) vs `H_logreg` (regularised linear) vs `H_rf` (base model), each scored on
   PR-AUC, precision, recall, F1, Brier;
3. **cluster bootstrap over whole blocks, 1,000 replicates** -> 95% CI on PR-AUC differences and a plain
   verdict (`improvement (CI above 0)` vs `no demonstrated improvement`).

Current result of that comparison (`outputs/metrics/static/g6_comparison.csv`): both H models beat the rule
with the CI strictly above zero, and H_rf beats H_logreg - so the nonlinearity is demonstrated rather than
assumed, and the rule is shown to be a weak but honest floor.

**Other validation routes available in this project (in increasing order of effort):**

| Method | How to run it here | What it would add |
|--------|--------------------|-------------------|
| Add a gradient-boosted competitor | `train_static.py --advanced` (adds `H_gbdt`) | checks the RF result is not an artefact of one ensemble family |
| Hyper-parameter sensitivity | vary `n_splits`, `min_samples_leaf`, `n_estimators` and re-run | shows scores are stable across reasonable settings |
| Alternative split scheme | `train.py --split chronological --test-fraction 0.3` | guards against a lucky spatial split |
| **External validation against observed events** | fill `data/raw/event_inventory.csv`, then guide steps 11-17 (`build_labels`, `train.py`, `evaluate.py`) | the only route that tests *real* flooding instead of agreement with FastFlood; enables A_rf vs B_rf with event bootstrap |
| Calibration/threshold analysis | `outputs/metrics/static/calibration.png` + reliability table | decides whether the 0.5 threshold and the Low/Med/High tiers are defensible |
| Permutation importance | swap impurity importance for permutation on the OOF model | removes the bias of impurity importance toward continuous features |

Run the cell below for the CV design and the comparison table as actually written to disk.
''', "sec4-validate"))

cells.append(code('''# Section 3/4 evidence: CV design, held-out metrics and the competing-model comparison.
import sys

import pandas as pd

try:
    from IPython.display import display
except ImportError:                      # plain-Python runner
    display = print

sys.path.insert(0, str(ROOT / "src"))
from common import load_settings                       # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

cfg = load_settings()
print("== Validation design ==")
print("CV                 : GroupKFold", cfg["training"]["n_splits"], "splits over whole spatial blocks")
print("block size         :", cfg["hazard"]["block_size_m"], "m")
print("bootstrap          :", cfg["evaluation"]["bootstrap_replicates"], "block-level replicates")
print("decision threshold :", cfg["training"]["decision_threshold"])
print("predictors (numeric):", ", ".join(cfg["training"]["static_features"]))
print("predictors (categorical):", ", ".join(cfg["training"]["categorical_features"]))

ht = pd.read_parquet(PROC / "hazard_modelling_table.parquet")
print("modelling table    :", ht.shape[0], "rows x", ht.shape[1], "cols |",
      ht["spatial_block"].nunique(), "spatial blocks")

print("\\n== Held-out metrics (whole-block folds) ==")
metrics = pd.read_csv(OUT / "metrics" / "static" / "metrics.csv")
display(metrics[["model", "n_rows", "n_positive", "prevalence", "precision", "recall", "pr_auc", "brier"]].round(3))

print("\\n== Competing models compared with a block bootstrap (gate G6) ==")
g6 = pd.read_csv(OUT / "metrics" / "static" / "g6_comparison.csv")
display(g6.round(3))

print("\\n== Feature importance of the base model ==")
display(pd.read_csv(OUT / "tables" / "feature_importance_H_rf.csv").head(10).round(3))''', "sec4-evidence"))

cells.append(md('''## Step 9 - static susceptibility map

Scores every road segment with the default model (`H_rf`, from `prediction.default_model`) and writes a CSV,
a GeoPackage and a map. Each row carries `susceptibility_source`: `fastflood_covered` where FastFlood cells
cover the road, `model_predicted_no_fastflood` where the value is only a model estimate - never present the
second kind as FastFlood output.
''', "predict"))

cells.append(code('''# Step 9: static susceptibility predictions for every road segment
run("src/predict.py")''', "predict-run"))

cells.append(md('''## Step 10 (optional) - rainfall context

The susceptibility score does **not** change with the weather. This step only adds Open-Meteo 24 h rainfall
context columns (`rain_above_threshold`); the live forecast fetch needs internet. Start with `--limit 8` to
confirm timestamps, units and variables before fetching the whole grid.
''', "weather"))

cells.append(code('''RUN_WEATHER_CONTEXT = False   # set True to fetch a live forecast (internet) and add rain context columns

if RUN_WEATHER_CONTEXT:
    run("src/fetch_openmeteo.py", "fetch", "--mode", "live", "--limit", "8", optional=True)
    run("src/predict.py", "--latest", optional=True)
else:
    print("Optional weather context skipped (set RUN_WEATHER_CONTEXT = True to run it).")''', "weather-run"))

cells.append(md('## Results - labels, models, metrics and outputs', "results"))

cells.append(code('''import pandas as pd

try:
    from IPython.display import Image, display
except ImportError:                       # plain-Python runner
    display = print
    Image = None

OUT = ROOT / "outputs"

print("== Gate G4h: label summary ==")
display(pd.read_csv(OUT / "tables" / "hazard_label_summary.csv"))

print("\\n== Gate G2/G4h: coverage report ==")
display(pd.read_csv(OUT / "tables" / "coverage_report.csv", index_col=0))

print("\\n== Held-out static metrics (whole 2 km blocks) ==")
display(pd.read_csv(OUT / "metrics" / "static" / "metrics.csv").round(3))

print("\\n== Gate G6: block-bootstrap PR-AUC differences ==")
display(pd.read_csv(OUT / "metrics" / "static" / "g6_comparison.csv").round(3))

print("\\n== Feature importance (H_rf) ==")
display(pd.read_csv(OUT / "tables" / "feature_importance_H_rf.csv").head(15))

print("\\n== Susceptibility predictions (step 9) ==")
pred = pd.read_csv(OUT / "predictions_H_rf_static.csv")
display(pred["susceptibility_tier"].value_counts().sort_index())
display(pred["susceptibility_source"].value_counts())
display(pred[["susceptibility_probability"]].describe().round(3))

print("\\n== Step timings ==")
display(pd.DataFrame(TIMINGS, columns=["step", "exit_code", "seconds"]).round(1))''', "results-tables"))

cells.append(code('''# Plots produced by the pipeline (PR curves, calibration, coverage, risk map)
pngs = sorted(p for p in OUT.rglob("*.png") if p.is_file())
if Image is None:
    print("Plots:", *[f"  {p.relative_to(ROOT)}" for p in pngs], sep="")
else:
    for p in pngs:
        print(p.relative_to(ROOT))
        display(Image(filename=str(p)))''', "results-plots"))

cells.append(md('''## How to read these numbers

- **They measure agreement with FastFlood, not real flooding.** The H_* models are emulators of a modelled
  hazard dataset that was itself driven by DEM and land cover - the same inputs they train on. Good held-out
  scores mean the emulator reproduces FastFlood, not that roads flood there.
- **Unknown is not negative.** Segments outside the hazard cells, mixed segments and touch-only overlaps stay
  Unknown and are excluded from training; they are never turned into 0.
- **Validation is spatial.** Whole 2 km blocks are held out because neighbouring segments share the same
  hazard cells; a random split would leak and inflate every score. The tiers (Low/Medium/High) are
  illustrative and not validated - describe them as *relative susceptibility*, never as warnings.
- **Next reads:** `logs/build_hazard_labels.log` and `logs/check_coverage.log` (coverage and unknown reasons),
  `outputs/metrics/static/` (metrics, plots, G6 verdict), `config/data_dictionary.csv` (every column), and a
  QGIS check of `data/interim/hazard_cells.gpkg` and `data/interim/uncovered_segments.gpkg` over the roads.
- **Optional dated-event workflow** (steps 11-17 of the guide) needs reviewed, time-stamped flood reports in
  `data/interim/event_inventory.csv` (template provided); only those can test whether rainfall forecasts help.
''', "caveats"))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

for c in cells:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))

out = Path(__file__).resolve().parents[1] / "pipeline.ipynb"
with open(out, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)
    fh.write("\n")
print("wrote", out.name, "with", len(cells), "cells; all code cells parse")
