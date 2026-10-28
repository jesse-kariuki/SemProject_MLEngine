import pandas as pd
import geopandas as gpd
from shapely.geometry import box

CSV_PATH = r"D:\floodguard\data\raw\flood_hazard\flood_hazard.csv"
BOUNDARY_PATH = r"D:\floodguard\data\interim\nairobi_boundary.gpkg"
OUTPUT_PATH = r"D:\floodguard\data\interim\uncovered_boundary.gpkg"

# Load boundary.
boundary = gpd.read_file(
    BOUNDARY_PATH,
    layer="boundary"
).to_crs("EPSG:32737")

# Load hazard grid.
df = pd.read_csv(CSV_PATH)

cells = gpd.GeoDataFrame(
    df[["flood_hazard"]].copy(),
    geometry=[
        box(xmin, ymin, xmax, ymax)
        for xmin, ymin, xmax, ymax in zip(
            df["lon_min"],
            df["lat_min"],
            df["lon_max"],
            df["lat_max"],
        )
    ],
    crs="EPSG:4326",
).to_crs("EPSG:32737")

study_area = boundary.geometry.union_all()
hazard_footprint = cells.geometry.union_all()

# Calculate uncovered study area.
uncovered = study_area.difference(hazard_footprint)

total_area_km2 = study_area.area / 1_000_000
uncovered_area_km2 = uncovered.area / 1_000_000

print(f"Study area: {total_area_km2:.2f} km²")
print(f"Uncovered area: {uncovered_area_km2:.2f} km²")
print(
    f"Uncovered percentage: "
    f"{uncovered_area_km2 / total_area_km2 * 100:.2f}%"
)

# Save uncovered geometry for GIS inspection.
gpd.GeoDataFrame(
    {"area_km2": [uncovered_area_km2]},
    geometry=[uncovered],
    crs="EPSG:32737",
).to_file(
    OUTPUT_PATH,
    layer="uncovered",
    driver="GPKG",
)

print(f"Saved uncovered areas to: {OUTPUT_PATH}")