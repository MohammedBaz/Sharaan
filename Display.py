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
    """Generates random time series data for selected variable."""
    time = pd.date_range(start='2024-01-01', periods=n_points, freq='D')
    
    if variable == 'Temperature (°C)':
        data = np.random.normal(25, 5, n_points).round(1)
    elif variable == 'Humidity (%)':
        data = np.random.uniform(20, 100, n_points).round(1)
    elif variable == 'Wind Speed (m/s)':
        data = np.random.weibull(2, n_points).round(1)
    elif variable == 'Precipitation (mm)':
        data = np.random.gamma(2, 2, n_points).round(1)
    else:
        data = np.random.rand(n_points)
    
    return pd.DataFrame({'Date': time, variable: data})

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
def generate_spatial_intensity(variable):
    """Returns intensity ranges based on selected variable."""
    ranges = {
        'Temperature (°C)': (0.5, 1.0),
        'Humidity (%)': (0.3, 0.9),
        'Wind Speed (m/s)': (0.4, 1.0),
        'Precipitation (mm)': (0.2, 0.8)
    }
    return ranges.get(variable, (0, 1))

@st.cache_data
def generate_heatmap_data(geojson, variable, num_points=2000):
    """Generates heatmap data with variable-specific intensities."""
    gdf = gpd.GeoDataFrame.from_features(geojson['features'])
    points = []
    min_intensity, max_intensity = generate_spatial_intensity(variable)
    
    for _, row in gdf.iterrows():
        geom = row.geometry
        polygons = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
        
        for poly in polygons:
            coords = generate_random_points_in_polygon(poly, num_points)
            for lat, lon in coords:
                intensity = random.uniform(min_intensity, max_intensity)
                points.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {"intensity": intensity}
                })
    
    return {"type": "FeatureCollection", "features": points}

# Load GeoJSON data
sharaan_geojson = load_geojson(geojson_url)
gdf = gpd.GeoDataFrame.from_features(sharaan_geojson["features"])
bounds = gdf.total_bounds
center_lat, center_lon = (bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2

# Streamlit App
st.title("Sharaan Protected Area Climate Dashboard")

# Parameter Selection
variables = [
    'Temperature (°C)',
    'Humidity (%)',
    'Wind Speed (m/s)',
    'Precipitation (mm)'
]
selected_var = st.sidebar.selectbox("Select Climate Parameter", variables)

# Generate Data
ts_data = generate_random_time_series(selected_var)
heatmap_data = generate_heatmap_data(sharaan_geojson, selected_var)

# Time Series Plot
st.subheader(f"{selected_var} Time Series")
fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(x='Date', y=selected_var, data=ts_data, ax=ax)
ax.set_title(f"Daily {selected_var} Variation")
ax.grid(True)
st.pyplot(fig)

# Heatmap Visualization
st.subheader(f"{selected_var} Spatial Distribution")

# Create Folium map
m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="cartodbpositron")

# Add protected area boundary
folium.GeoJson(
    sharaan_geojson,
    style_function=lambda x: {
        'fillColor': '#4a148c',
        'color': '#d81b60',
        'weight': 2,
        'fillOpacity': 0.2
    }
).add_to(m)

# Prepare heatmap data
heat_points = []
for feature in heatmap_data['features']:
    try:
        lon = feature['geometry']['coordinates'][0]
        lat = feature['geometry']['coordinates'][1]
        intensity = feature['properties']['intensity']
        heat_points.append([lat, lon, intensity])
    except (KeyError, IndexError):
        continue

# Add heatmap layer with corrected gradient
if heat_points:
    HeatMap(
        heat_points,
        radius=25,
        blur=20,
        min_opacity=0.4,
        gradient={
            '0.4': 'blue',
            '0.6': 'green',
            '0.8': 'yellow',
            '1.0': 'red'
        }
    ).add_to(m)

# Display map
st_folium(m, width=800, height=500)

# Data Statistics
st.subheader("Data Summary")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Average", f"{ts_data[selected_var].mean():.1f}")
with col2:
    st.metric("Maximum", f"{ts_data[selected_var].max():.1f}")
with col3:
    st.metric("Minimum", f"{ts_data[selected_var].min():.1f}")

st.write("Note: All data shown is simulated for demonstration purposes.")
