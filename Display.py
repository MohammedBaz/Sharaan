import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import seaborn as sns
import json
import requests
import geopandas as gpd

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
        data = np.random.uniform(10, 35, n_points)
    elif variable == 'Precipitation':
        data = np.random.uniform(0, 15, n_points)
    elif variable == 'Wind Speed':
        data = np.random.uniform(0, 20, n_points)
    else:
        data = np.random.rand(n_points)
    return pd.DataFrame({'Time': time, variable: data})

@st.cache_data
def generate_random_spatial_data(geojson, variable):
    """Generates random spatial data based on GeoJSON features."""
    features = geojson['features']
    spatial_data = []
    for i, feature in enumerate(features):
        if feature['geometry']['type'] == 'Polygon':
            # Use the first coordinate as a representative point (can be improved)
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

# Load GeoJSON data
sharaan_geojson = load_geojson(geojson_url)

st.title("Climate Data for Decision Support in Sharaan Protected Area")

# Sidebar for variable selection
st.sidebar.header("Select Variable")
selected_variable = st.sidebar.selectbox("Choose a climate variable", ['Temperature', 'Precipitation', 'Wind Speed'])

# Generate random data
time_series_data = generate_random_time_series(selected_variable)
spatial_data = generate_random_spatial_data(sharaan_geojson, selected_variable)

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

m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

# Create a GeoJSON layer for the Sharaan boundary
folium.GeoJson(sharaan_geojson, style_function=lambda feature: {
    'fillColor': 'purple',
    'color': 'red',
    'weight': 2,
    'fillOpacity': 0.2
}).add_to(m)

# Add heatmap layer (using a simple approach for now)
for index, row in spatial_data.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=row['value'] * 2 if selected_variable != 'Precipitation' else row['value'] * 5, # Adjust radius based on variable
        color='red',
        fill=True,
        fill_color='orangered',
        fill_opacity=0.6
    ).add_to(m)

st_folium(m, width=700, height=500)
