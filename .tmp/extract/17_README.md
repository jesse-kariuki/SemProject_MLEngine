Short project readme with setup and run order.
# Nairobi Road Flood Susceptibility (research prototype)

Maps how susceptible road segments in Nairobi County are to flooding, from the SPACE4ALL FastFlood hazard cells
(modelled, no timestamps) + DEM + land cover + OpenStreetMap roads. Open-Meteo 24 h rainfall is optional
context only. This is a susceptibility map, not a flood forecast: no exact flood events are available to train or
validate one. Dated, reviewed flood reports (optional workflow) would be needed for that.

## Setup
```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Place inputs as described in config/settings.yaml (inputs section; hazard file at
data/raw/flood_hazard/flood_hazard.csv), then run in order:
```
python src/prepare_gis.py inventory
python src/prepare_gis.py all
python src/fetch_openmeteo.py grid         # weather points (needed before 'static' to link roads to them)
python src/build_features.py static
python src/build_hazard_labels.py          # PRIMARY labels from the hazard dataset
python src/check_coverage.py               # boundary area + hazard coverage diagnostics (read the log!)
python src/train_static.py                 # spatial-block validation
python src/evaluate.py --static
python src/predict.py                      # static susceptibility map
python src/fetch_openmeteo.py fetch --mode live --limit 8
python src/predict.py --latest             # susceptibility + rain context columns
```
Optional, once you have reviewed time-stamped events: build_labels.py labels, build_features.py forecast
--from-labels, build_labels.py assemble, train.py, evaluate.py, predict.py --model B_rf --latest.
Verify the code first on synthetic data: `python tests/smoke_test.py`.

## Important
Outputs are research-prototype values, not an operational warning service. Unknown labels are never treated as
non-floods; always split by event; report uncertainty and sample counts.

