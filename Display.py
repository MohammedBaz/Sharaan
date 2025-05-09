import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import random
import json

# Define the GeoJSON URL
geojson_url = "https://raw.githubusercontent.com/MohammedBaz/Sharaan/main/Sharaan.geojson"

@st.cache_data
def load_geojson(url):
    """Loads GeoJSON data from a URL."""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

@st.cache_data
def generate_random_time_series(variable, n_points=100):
    """Generates random time series data."""
    time = pd.to_datetime(pd.date_range(start='2024-01-01', periods=n_points))
    if variable == 'Temperature':
        data = np.random.uniform(10, 35, n_points).tolist()
    elif variable == 'Precipitation':
        data = np.random.uniform(0, 15, n_points).tolist()
    elif variable == 'Wind Speed':
        data = np.random.uniform(0, 20, n_points).tolist()
    else:
        data = np.random.rand(n_points).tolist()
    return pd.DataFrame({'Time': time, variable: data})

def generate_random_points_in_polygon(polygon, num_points=2000):
    """Generates random points within a Shapely polygon."""
    minx, miny, maxx, maxy = polygon.bounds
    points = []
    while len(points) < num_points:
        point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(point):
            points.append((float(point.y), float(point.x)))  # Explicit float conversion
    return points

@st.cache_data
def generate_random_spatial_data_geojson(geojson, variable, num_points_per_polygon=2000):
    """Generates GeoJSON-like data for heatmap visualization."""
    features = geojson['features']
    point_features = []
    gdf = gpd.read_file(geojson_url)  # Load GeoDataFrame for Shapely geometries

    for index, feature in gdf.iterrows():
        if feature.geometry.geom_type == 'Polygon':
            polygon = feature.geometry
            random_points = generate_random_points_in_polygon(polygon, num_points_per_polygon)
            for lat, lon in random_points:
                if variable == 'Temperature':
                    intensity = np.random.uniform(0.5, 1.0)
                elif variable == 'Precipitation':
                    intensity = np.random.uniform(0.2, 0.8)
                elif variable == 'Wind Speed':
                    intensity = np.random.uniform(0.4, 0.9)
                else:
                    intensity = np.random.rand()
                point_features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                    "properties": {"intensity": float(intensity)}
                })
        elif feature.geometry.geom_type == 'MultiPolygon':
            for polygon in feature.geometry.geoms:
                random_points = generate_random_points_in_polygon(polygon, num_points_per_polygon)
                for lat, lon in random_points:
                    intensity = np.random.uniform(0.5, 1.0)
                    point_features.append({
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                        "properties": {"intensity": float(intensity)}
                    })
    return {"type": "FeatureCollection", "features": point_features}

# Load GeoJSON data
sharaan_geojson = load_geojson(geojson_url)

st.title("Climate Data for Decision Support in Sharaan Protected Area")

# Sidebar for variable selection
st.sidebar.header("Select Variable")
selected_variable = st.sidebar.selectbox("Choose a climate variable", ['Temperature', 'Precipitation', 'Wind Speed'])

# Generate random data
time_series_data = generate_random_time_series(selected_variable)
heatmap_geojson_data = generate_random_spatial_data_geojson(sharaan_geojson, selected_variable, num_points_per_polygon=2000)

# --- Time Series Plot ---
st.subheader(f"{selected_variable} Time Series")
fig_ts, ax_ts = plt.subplots()
sns.lineplot(x='Time', y=selected_variable, data=time_series_data, ax=ax_ts)
ax_ts.set_xlabel("Time")
ax_ts.set_ylabel(selected_variable)
st.pyplot(fig_ts)

# --- Heatmap on Map ---
st.subheader(f"{selected_variable} Heatmap")

# Calculate the center of the Sharaan area for the initial map view
gdf = gpd.read_file(geojson_url)
if not gdf.empty:
    bounds = gdf.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
else:
    center_lat, center_lon = 26.9, 37.8  # Default if GeoJSON fails to load

m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="cartodbpositron")

# Create a GeoJSON layer for the Sharaan boundary with explicit float conversion
folium.GeoJson(
    {
        "type": sharaan_geojson.get("type", "FeatureCollection"),
        "features": [
            {
                "type": feature.get("type", "Feature"),
                "geometry": {
                    "type": feature["geometry"].get("type"),
                    "coordinates": [[(float(lon), float(lat)) for lon, lat in ring] for ring in feature["geometry"].get("coordinates", [])]
                    if feature["geometry"].get("type") == "Polygon"
                    else [[(float(lon), float(lat)) for lon, lat in line] for line in feature["geometry"].get("coordinates", [])]
                    if feature["geometry"].get("type") == "LineString"
                    else [float(coord) for coord in feature["geometry"].get("coordinates", [])]
                    if feature["geometry"].get("type") == "Point"
                    else feature["geometry"].get("coordinates")
                },
                "properties": feature.get("properties", {})
            }
            for feature in sharaan_geojson.get("features", [])
        ]
    },
    style_function=lambda feature: {
        'fillColor': 'purple',
        'color': 'red',
        'weight': 2,
        'fillOpacity': 0.2
    }
).add_to(m)

# Add heatmap as a GeoJSON layer with styling
def heatmap_style(feature):
    intensity = feature['properties']['intensity']
    normalized_intensity = (intensity - 0.2) / 0.8  # Normalize to 0-1
    color = plt.cm.viridis(normalized_intensity)  # Use a matplotlib colormap
    hex_color = matplotlib.colors.rgb2hex(color)
    return {'radius': 10, 'fillColor': hex_color, 'color': hex_color, 'fillOpacity': 0.8, 'weight': 0}

folium.GeoJson(
    heatmap_geojson_data,
    point_to_layer=lambda feature, latlng: folium.CircleMarker(
        location=latlng,
        style_function=heatmap_style
    )
).add_to(m)

st_folium(m, width=700, height=500)
