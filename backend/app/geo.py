"""In-memory road segments and zones, built from the exported assets."""
from __future__ import annotations

import csv
import gzip
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import get_settings

R_EARTH_M = 6_371_000.0
TIER_HIGH, TIER_MEDIUM = 0.6, 0.3   # prediction.risk_tiers in settings.yaml
LOCAL_PERCENTILE = 75               # "any High road nearby" is true for 90% of Nairobi at 500 m, so it cannot rank places


def tier_of(p: float) -> str:
    return "High" if p >= TIER_HIGH else "Medium" if p >= TIER_MEDIUM else "Low"


@dataclass(frozen=True)
class Zone:
    zone_id: str
    lat: float
    lon: float
    river_p99: float | None


@dataclass(frozen=True)
class Place:
    zone_id: str
    tier: str                       # local susceptibility tier
    susceptibility: float | None
    distance_to_road_m: float
    roads: list[dict]               # named roads within the radius, most susceptible first


class GeoStore:
    def __init__(self, assets_dir: Path):
        meta = json.loads((assets_dir / "zones.json").read_text())
        self.model_version: str = meta["model_version"]
        self.zones = {z["zone_id"]: Zone(z["zone_id"], z["lat"], z["lon"], z.get("river_p99")) for z in meta["zones"]}
        with gzip.open(assets_dir / "segments.csv.gz", "rt", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.road_name = np.array([r["road_name"] for r in rows], dtype=object)
        self.tier = np.array([r["tier"] for r in rows], dtype=object)
        self.zone = np.array([r["zone_id"] for r in rows], dtype=object)
        self.lat = np.array([float(r["lat"]) for r in rows])
        self.lon = np.array([float(r["lon"]) for r in rows])
        self.p = np.array([float(r["p"]) if r["p"] else np.nan for r in rows])

    def neighbours(self, radius_km: float) -> dict[str, list[str]]:
        """Zones within `radius_km` of each zone, itself included."""
        def km(a: Zone, b: Zone) -> float:
            dy = math.radians(a.lat - b.lat)
            dx = math.radians(a.lon - b.lon) * math.cos(math.radians(a.lat))
            return R_EARTH_M * math.hypot(dx, dy) / 1000
        return {a.zone_id: [b.zone_id for b in self.zones.values() if km(a, b) <= radius_km]
                for a in self.zones.values()}

    def _distances(self, lat: float, lon: float) -> np.ndarray:
        """Equirectangular distance in metres: accurate enough across one city and fast over 83k rows."""
        dy = np.radians(self.lat - lat)
        dx = np.radians(self.lon - lon) * math.cos(math.radians(lat))
        return R_EARTH_M * np.hypot(dx, dy)

    def locate(self, lat: float, lon: float, radius_m: int) -> Place | None:
        """Describe a point, or return None if it is outside the mapped road network."""
        d = self._distances(lat, lon)
        nearest = int(np.argmin(d))
        if d[nearest] > get_settings().coverage_radius_m:
            return None
        near = d <= radius_m
        if not near.any():
            near[nearest] = True
        scored = self.p[near][~np.isnan(self.p[near])]
        susceptibility = float(np.percentile(scored, LOCAL_PERCENTILE)) if len(scored) else None
        tier = tier_of(susceptibility) if susceptibility is not None else "Unavailable"
        return Place(str(self.zone[nearest]), tier, None if susceptibility is None else round(susceptibility, 3),
                     round(float(d[nearest]), 1), self._named_roads(near))

    def _named_roads(self, mask: np.ndarray, limit: int = 5) -> list[dict]:
        best: dict[str, float] = {}
        for name, p in zip(self.road_name[mask], self.p[mask]):
            if name and not math.isnan(p) and p > best.get(name, -1.0):
                best[name] = float(p)
        ordered = sorted(best.items(), key=lambda kv: -kv[1])[:limit]
        return [{"road_name": n, "susceptibility": round(p, 3), "tier": tier_of(p)} for n, p in ordered]

    def search_roads(self, query: str, limit: int = 10) -> list[dict]:
        q = query.strip().lower()
        hits: dict[str, list[int]] = {}
        for i, name in enumerate(self.road_name):
            if name and q in name.lower():
                hits.setdefault(name, []).append(i)
        ranked = sorted(hits, key=lambda n: (not n.lower().startswith(q), -len(hits[n]), n))[:limit]
        out = []
        for name in ranked:
            idx = np.array(hits[name])
            mid = idx[len(idx) // 2]
            out.append({"road_name": name, "latitude": float(self.lat[mid]), "longitude": float(self.lon[mid]),
                        "segments": len(idx), "high_susceptibility_segments": int(np.sum(self.tier[idx] == "High"))})
        return out

    def roads_at_level(self, levels: dict[str, dict[str, int]], min_level: int, limit: int) -> list[dict]:
        """Named roads whose level reaches `min_level`, given {zone_id: {tier: level}}."""
        roads: dict[str, dict] = {}
        for name, tier, zone in zip(self.road_name, self.tier, self.zone):
            if not name:
                continue
            level = levels.get(zone, {}).get(tier, 0)
            if level < min_level:
                continue
            row = roads.setdefault(name, {"road_name": name, "level": 0, "segments": 0, "zones": set()})
            row["level"] = max(row["level"], level)
            row["segments"] += 1
            row["zones"].add(zone)
        ranked = sorted(roads.values(), key=lambda r: (-r["level"], -r["segments"], r["road_name"]))[:limit]
        return [{**r, "zones": sorted(r["zones"])} for r in ranked]


@lru_cache
def get_store() -> GeoStore:
    return GeoStore(get_settings().assets_dir)
