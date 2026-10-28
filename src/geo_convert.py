import geopandas as gpd

boundary = gpd.read_file("D:/floodguard/data/raw/boundary/NairobiEntire.geojson")

boundary = boundary.set_crs("EPSG:4326")

# Save as a GeoPackage
boundary.to_file(
    "nairobi_boundary.gpkg",
    layer="boundary",
    driver="GPKG"
)

print(boundary)