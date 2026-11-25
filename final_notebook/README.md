# Final notebook: RandomForest_Final

`RandomForest_Final.ipynb` runs the whole road flood-susceptibility pipeline for Nairobi: weather grid, static
features, labels from the flood-hazard dataset, coverage check, Random Forest and logistic-regression
training with spatial-block cross-validation, evaluation, and the per-road susceptibility map.

This folder holds everything the notebook needs, so it can be re-run straight after cloning.

## Run it

```bash
cd final_notebook
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt jupyter
jupyter notebook RandomForest_Final.ipynb
```

Then choose **Run All**. Jupyter must be started inside `final_notebook/`, because the notebook looks for
`config/settings.yaml` in the current folder. You can instead set `FLOOD_PROJECT_ROOT` to this folder's path.

Each step runs a script in `src/`, writes its log to `logs/` and its results to `data/processed/`, `models/`
and `outputs/`. These folders are recreated on each run.

## What is included

| Path | What it is |
|------|------------|
| `src/` | The pipeline scripts the notebook calls |
| `config/settings.yaml` | All settings: CRS, label thresholds, model parameters, random seed |
| `data/raw/flood_hazard/flood_hazard.csv` | The flood-hazard cells used as training labels |
| `data/interim/` | Prepared GIS layers made by steps 1-2: study boundary, ~100 m road segments, DEM and land-cover clips |
| `data/processed/static_segment_features.parquet` | Static features from a full run. Used only to reuse distance-to-river values (see below) |
| `outputs/tables/asset_inventory.csv` | The asset inventory from step 1, read by the data-exploration cell |

## Steps 1-2 and the raw downloads

Steps 1-2 build the files in `data/interim/` from raw downloads that are too large for this repository. If
those downloads are missing, the notebook skips steps 1-2 and uses the prepared files. Every later step
(features, labels, training, evaluation, prediction) still runs from scratch.

To re-run steps 1-2 as well, put these files in place first:

| File | Source |
|------|--------|
| `data/raw/osm_shapefiles/roads/gis_osm_roads_free_1.shp` (+ `.shx`, `.dbf`, `.prj`) | Geofabrik, Kenya extract, free shapefile download |
| `data/raw/osm_shapefiles/roads/gis_osm_adminareas_a_free_1.shp` (+ sidecars) | Same Geofabrik download |
| `data/raw/osm_shapefiles/roads/gis_osm_waterways_free_1.shp` (+ sidecars) | Same Geofabrik download |
| `data/raw/dem/S02E036.tif` | SRTM 1 arc-second tile S02E036 |
| `data/raw/landcover/LCFM_LCM-10_V100_2020_S03E036_MAP.tif` | LCFM LCM-10 10 m land cover, 2020, tile S03E036 (about 128 MB) |

**Distance to river.** This feature is computed from the OSM waterways layer. When that layer is missing,
`src/build_features.py` reuses the distances from `data/processed/static_segment_features.parquet`, which a
full run produced for the same segments. The model is then trained on the same features as the full run.

## Optional step 10

Step 10 adds live Open-Meteo rainfall context to the map and needs internet access. It is off by default. To
turn it on, set `RUN_WEATHER_CONTEXT = True` in that cell.
