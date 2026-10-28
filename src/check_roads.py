import geopandas as gpd

roads_path = r"D:\floodguard\data\interim\nairobi_roads.gpkg"
boundary_path = r"D:\floodguard\data\interim\nairobi_boundary.gpkg"

roads = gpd.read_file(roads_path, layer="segments")
boundary = gpd.read_file(boundary_path, layer="boundary")

boundary = boundary.to_crs(roads.crs)
study_area = boundary.geometry.union_all()

print("Segments:", len(roads))
print("Total road length (km):", roads.geometry.length.sum() / 1000)

outside = ~roads.geometry.covered_by(study_area)

print("Segments outside boundary:", int(outside.sum()))
print("Outside segment length (km):", roads.loc[outside].geometry.length.sum() / 1000)