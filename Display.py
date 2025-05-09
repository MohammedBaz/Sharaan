import streamlit as st
import geopandas as gpd
import pandas as pd
from zipfile import ZipFile
import os
import tempfile

def kmz_to_geojson(kmz_file):
    """Converts a KMZ file to a GeoJSON string."""
    temp_dir = tempfile.TemporaryDirectory()
    try:
        with ZipFile(kmz_file, 'r') as kmz:
            kml_file = [f for f in kmz.namelist() if f.lower().endswith('.kml')][0]
            kmz.extract(kml_file, temp_dir.name)
            kml_path = os.path.join(temp_dir.name, kml_file)
            gdf = gpd.read_file(kml_path)
            geojson_str = gdf.to_json()
            return geojson_str
    finally:
        temp_dir.cleanup()

def display_kmz_area(kmz_file):
    """Displays the area defined in a KMZ file on a Streamlit map."""
    geojson_data = kmz_to_geojson(kmz_file)
    if geojson_data:
        gdf = gpd.read_file(geojson_data)
        if not gdf.empty and 'geometry' in gdf.columns:
            # Extract centroid to center the map
            centroid = gdf.geometry.unary_union.centroid
            st.map(pd.DataFrame({'lat': [centroid.y], 'lon': [centroid.x]}))

            # You might want to display the boundaries as well.
            # For polygons, you can extract the exterior coordinates.
            polygon_coords = []
            for geom in gdf.geometry:
                if geom.geom_type == 'Polygon':
                    polygon_coords.extend(list(geom.exterior.coords))
                elif geom.geom_type == 'MultiPolygon':
                    for polygon in geom.geoms:
                        polygon_coords.extend(list(polygon.exterior.coords))

            if polygon_coords:
                df_polygon = pd.DataFrame(polygon_coords, columns=['lon', 'lat'])
                st.map(df_polygon)
        else:
            st.warning("No valid geometry data found in the KMZ file.")
    else:
        st.error("Could not process the KMZ file.")

st.title("Display Area from KMZ")

uploaded_file = st.file_uploader("Upload a KMZ file", type=["kmz"])

if uploaded_file is not None:
    display_kmz_area(uploaded_file)
