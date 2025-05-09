import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import matplotlib
import seaborn as sns
import json
import requests
import geopandas as gpd
import sys
import os
import importlib.metadata

# Define the GeoJSON URL
geojson_url = "https://raw.githubusercontent.com/MohammedBaz/Sharaan/main/Sharaan.geojson"

st.title("Troubleshooting Information")

st.subheader("Python Version")
st.write(f"Python Version: {sys.version}")

st.subheader("Operating System")
st.write(f"Operating System: {sys.platform}")

st.subheader("Environment Variables")
st.write(f"Environment Variables: {os.environ}")

st.subheader("Installed Libraries and Versions")

libraries = {
    "streamlit": st.__version__,
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "folium": folium.__version__,
    "streamlit-folium": importlib.metadata.version("streamlit_folium") if "streamlit_folium" in sys.modules else "Not Imported",
    "matplotlib": matplotlib.__version__,
    "seaborn": sns.__version__,
    "requests": requests.__version__,
    "geopandas": gpd.__version__ if "geopandas" in sys.modules else "Not Imported",
    "fiona": importlib.metadata.version("fiona") if "fiona" in sys.modules else "Not Imported",
    "shapely": importlib.metadata.version("shapely") if "shapely" in sys.modules else "Not Imported",
    "pyproj": importlib.metadata.version("pyproj") if "pyproj" in sys.modules else "Not Imported",
}

for lib, version in libraries.items():
    st.write(f"{lib}: {version}")

st.subheader("Loaded Modules")
st.write(f"Loaded Modules: {list(sys.modules.keys())}")

st.subheader("GeoJSON Content (First 5 Features)")
@st.cache_data
def load_geojson_for_debug(url):
    """Loads GeoJSON data from a URL."""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

sharaan_geojson_debug = load_geojson_for_debug(geojson_url)
if sharaan_geojson_debug and 'features' in sharaan_geojson_debug:
    st.write(f"First 5 GeoJSON Features: {sharaan_geojson_debug['features'][:5]}")
else:
    st.write("Could not load GeoJSON for debugging.")

st.subheader("Spatial Data (First 5 Rows)")
@st.cache_data
def generate_random_spatial_data_for_debug(geojson, variable):
    # Your existing generate_random_spatial_data function
    features = geojson['features']
    spatial_data = []
    for i, feature in enumerate(features):
        if feature['geometry']['type'] == 'Polygon':
            lon, lat = feature['geometry']['coordinates'][0][0][0], feature['geometry']['coordinates'][0][0][1]
            if variable == 'Temperature':
                value = np.random.uniform(15, 30)
            elif variable == 'Precipitation':
                value = np.random.uniform(0, 10)
            elif variable == 'Wind Speed':
                value = np.random.uniform(5, 15)
            else:
                value = np.random.rand()
            spatial_data.append({'latitude': lat, 'longitude': lon, 'value': value})
    return pd.DataFrame(spatial_data)

if sharaan_geojson_debug:
    debug_spatial_data = generate_random_spatial_data(sharaan_geojson_debug, st.sidebar.selectbox("Debug Variable", ['Temperature', 'Precipitation', 'Wind Speed']))
    st.write(f"First 5 Rows of Spatial Data: {debug_spatial_data.head()}")

st.subheader("Heatmap Data (First 5 Rows)")
@st.cache_data
def generate_random_spatial_data_heatmap_for_debug(geojson, variable):
    # Your existing generate_random_spatial_data_heatmap function
    features = geojson['features']
    heatmap_data = []
    for feature in features:
        if feature['geometry']['type'] == 'Polygon':
            coords = feature['geometry']['coordinates'][0]
            lats = [coord[1] for coord in coords]
            lons = [coord[0] for coord in coords]
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)
            if variable == 'Temperature':
                intensity = np.random.uniform(0.5, 1.0)
            elif variable == 'Precipitation':
                intensity = np.random.uniform(0.2, 0.8)
            elif variable == 'Wind Speed':
                intensity = np.random.uniform(0.4, 0.9)
            else:
                intensity = np.random.rand()
            heatmap_data.append([center_lat, center_lon, intensity])
    return heatmap_data

if sharaan_geojson_debug:
    debug_heatmap_data = generate_random_spatial_data_heatmap(sharaan_geojson_debug, st.sidebar.selectbox("Debug Heatmap Variable", ['Temperature', 'Precipitation', 'Wind Speed']))
    if debug_heatmap_data:
        st.write(f"First 5 Rows of Heatmap Data: {debug_heatmap_data[:5]}")
    else:
        st.write("Heatmap data is empty.")

# The rest of your map and plot code can remain, or you can comment it out for focused debugging
