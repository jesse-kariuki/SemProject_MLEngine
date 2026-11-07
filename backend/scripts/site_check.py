"""Does road susceptibility agree with places that really flooded?

Scores the locations named in flood reports (evidence/flood_events.csv) against the city-wide distribution.
These sites all flooded, so a useful score should rate them above a typical location.
Coordinates are approximate, read off general maps, because the reports give only place names.

    python -m scripts.site_check        # from backend/
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.geo import get_store

EVIDENCE = Path(__file__).resolve().parents[1] / "evidence"
RADIUS_M = 300
SITES = {
    "Mathare 4A": (-1.2600, 36.8570),
    "Mukuru kwa Reuben": (-1.3120, 36.8760),
    "Likoni Bridge, South B": (-1.3095, 36.8370),
    "Kariokor": (-1.2740, 36.8330),
    "Utawala": (-1.2860, 36.9650),
    "Ruai bypass at Kangundo Rd": (-1.2670, 36.9620),
    "Seasons Road": (-1.2190, 36.8900),
    "Jogoo Road": (-1.2930, 36.8600),
}
RANDOM_SAMPLE = 2000


def main() -> None:
    store = get_store()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(store.lat), RANDOM_SAMPLE, replace=False)
    city = [store.locate(float(store.lat[i]), float(store.lon[i]), RADIUS_M) for i in idx]
    city_scores = np.array([p.susceptibility for p in city if p and p.susceptibility is not None])
    tiers = [p.tier for p in city if p]
    baseline = {t: round(tiers.count(t) / len(tiers), 3) for t in ("Low", "Medium", "High", "Unavailable")}

    rows = []
    for name, (lat, lon) in SITES.items():
        place = store.locate(lat, lon, RADIUS_M)
        if place is None:
            rows.append({"site": name, "tier": "outside coverage"})
            continue
        pct = None if place.susceptibility is None else round(float(np.mean(city_scores <= place.susceptibility)), 2)
        rows.append({"site": name, "tier": place.tier, "susceptibility": place.susceptibility,
                     "city_percentile": pct, "distance_to_road_m": place.distance_to_road_m})
    rated = [r for r in rows if r.get("city_percentile") is not None]
    summary = {"sites": len(rows), "sites_rated_low": sum(r["tier"] == "Low" for r in rows),
               "sites_rated_medium_or_high": sum(r["tier"] in ("Medium", "High") for r in rows),
               "median_city_percentile": float(np.median([r["city_percentile"] for r in rated])),
               "city_baseline_tiers": baseline}
    (EVIDENCE / "site_check.json").write_text(json.dumps({"summary": summary, "sites": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
