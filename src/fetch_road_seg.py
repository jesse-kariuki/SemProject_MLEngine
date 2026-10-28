
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import substring

HOT_ROADS = Path(r"D:\floodguard\data\raw\roads\nairobi_roads.gpkg")
BOUNDARY = Path(r"D:\floodguard\data\interim\nairobi_boundary.gpkg")
OUTPUT = Path(r"D:\floodguard\data\interim\nairobi_roads.gpkg")

ROAD_LAYER = "nairobi_roads"
BOUNDARY_LAYER = "boundary"

# Nairobi: UTM zone 37S, with coordinates and lengths in metres.
WORKING_CRS = "EPSG:32737"
SEGMENT_LENGTH_M = 100.0


def line_parts(geom):
    """Yield individual LineString geometries."""
    if geom is None or geom.is_empty:
        return

    if geom.geom_type == "LineString":
        yield geom

    elif geom.geom_type == "MultiLineString":
        for part in geom.geoms:
            if not part.is_empty and part.length > 0:
                yield part

    elif geom.geom_type == "GeometryCollection":
        for part in geom.geoms:
            yield from line_parts(part)


def main():
    print("Reading HOT Export roads...")
    roads = gpd.read_file(HOT_ROADS, layer=ROAD_LAYER)

    print("Reading Nairobi boundary...")
    boundary = gpd.read_file(BOUNDARY, layer=BOUNDARY_LAYER)

    if roads.empty or boundary.empty:
        raise ValueError("Road or boundary layer is empty.")

    if roads.crs is None or boundary.crs is None:
        raise ValueError("Both datasets must have a defined CRS.")

    # Project both datasets into metres before clipping/splitting.
    roads = roads.to_crs(WORKING_CRS)
    boundary = boundary.to_crs(WORKING_CRS)

    # Keep only the area within the boundary.
    study_area = boundary.geometry.union_all()
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty]
    roads = gpd.clip(roads, study_area)

    keep = [
        c for c in
        ["osm_id", "highway", "name", "oneway",
         "bridge", "tunnel", "surface"]
        if c in roads.columns
    ]

    records = []
    next_id = 1

    print(f"Clipped road features: {len(roads)}")
    print(f"Splitting roads into {SEGMENT_LENGTH_M:.0f} m sections...")

    for _, row in roads.iterrows():
        for line in line_parts(row.geometry):
            length = line.length

            if length <= 0:
                continue

            start = 0.0

            while start < length:
                end = min(start + SEGMENT_LENGTH_M, length)
                piece = substring(line, start, end)

                if piece.geom_type == "LineString" and piece.length > 0.01:
                    record = {
                        "segment_id": f"SEG_{next_id:08d}",
                        "segment_length_m": piece.length,
                        "geometry": piece,
                    }

                    for col in keep:
                        record[col] = row[col]

                    records.append(record)
                    next_id += 1

                start = end

    if not records:
        raise ValueError(
            "No segments were created. Check the boundary and road coverage."
        )

    segments = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=WORKING_CRS,
    )

    # Save the geometry and length in metres, as expected by the
    # hazard-label pipeline.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Close GIS applications that may have locked the output file.
    if OUTPUT.exists():
        OUTPUT.unlink()

    study_area = boundary.geometry.union_all()

    segments = segments[
        segments.geometry.covered_by(study_area)
    ].copy()

    print("Segments after boundary check:", len(segments))
    segments.to_file(
        OUTPUT,
        layer="segments",
        driver="GPKG",
        index=False,
    )

    print(f"Created {len(segments):,} road segments.")
    print(f"Total road length: {segments['segment_length_m'].sum() / 1000:.2f} km")
    print(f"CRS: {segments.crs}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
