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
import matplotlib.colors
from folium.plugins import HeatMap

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
            points.append((point.y, point.x))
    return points

@st.cache_data
def generate_random_spatial_data_geojson(geojson, variable, num_points_per_polygon=2000):
    """Generates heatmap data with variable-specific intensities."""
    features = geojson['features']
    point_features = []
    gdf = gpd.GeoDataFrame.from_features(features)

    for _, feature in gdf.iterrows():
        geom = feature.geometry
        polygons = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]

        for polygon in polygons:
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
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {"intensity": intensity}
                })
    return {"type": "FeatureCollection", "features": point_features}

# Load GeoJSON data
sharaan_geojson = load_geojson(geojson_url)
gdf = gpd.GeoDataFrame.from_features(sharaan_geojson["features"])
bounds = gdf.total_bounds
center_lat, center_lon = (bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2

st.title("Climate Data for Decision Support in Sharaan Protected Area")

# Sidebar for variable selection
st.sidebar.header("Select Variable")
selected_variable = st.sidebar.selectbox("Choose a climate variable", 
                                       ['Temperature', 'Precipitation', 'Wind Speed'])

# Generate data
time_series_data = generate_random_time_series(selected_variable)
heatmap_geojson_data = generate_random_spatial_data_geojson(sharaan_geojson, selected_variable)

# --- Time Series Plot ---
st.subheader(f"{selected_variable} Time Series")
fig, ax = plt.subplots()
sns.lineplot(x='Time', y=selected_variable, data=time_series_data, ax=ax)
ax.set_xlabel("Date")
ax.set_ylabel(selected_variable)
st.pyplot(fig)

# --- Heatmap Visualization ---
st.subheader(f"{selected_variable} Intensity Heatmap")

# Create base map
m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="cartodbpositron")

# Add Sharaan boundary with static styling
folium.GeoJson(
    sharaan_geojson,
    style_function=lambda x: {
        'fillColor': 'purple',
        'color': 'red',
        'weight': 2,
        'fillOpacity': 0.2
    }
).add_to(m)

# Prepare heatmap data
heat_data = [
    [
        feature['geometry']['coordinates'][1],  # Latitude
        feature['geometry']['coordinates'][0],  # Longitude
        feature['properties']['intensity']
    ]  # Added missing closing bracket for inner list
    for feature in heatmap_geojson_data['features']
]

# Add heatmap layer
HeatMap(
    heat_data,
    radius=20,
    blur=15,
    min_opacity=0.4,
    gradient={
        0.4: 'blue',
        0.6: 'lime',
        0.75: 'yellow',
        1.0: 'red'
    }
).add_to(m)

# Display map
st_folium(m, width=700, height=500)
