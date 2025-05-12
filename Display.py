import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import random
import requests

# Configuration
DATA_URL = "https://raw.githubusercontent.com/MohammedBaz/Sharaan/main/dataset.csv"
GEOJSON_URL = "https://raw.githubusercontent.com/MohammedBaz/Sharaan/main/Sharaan.geojson"

@st.cache_data
def load_data():
    """Load and validate climate dataset"""
    try:
        df = pd.read_csv(DATA_URL)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        if df['Date'].isnull().any():
            invalid_dates = df[df['Date'].isnull()]['Date'].unique()
            st.error(f"Invalid date formats found: {invalid_dates}")
            st.stop()
            
        return df.sort_values('Date')
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        st.stop()

@st.cache_data
def load_geojson():
    """Load protected area boundaries"""
    try:
        response = requests.get(GEOJSON_URL)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Geojson loading failed: {str(e)}")
        st.stop()

@st.cache_data
def generate_sensor_data(_df, _geojson, num_sensors=100):
    """Generate sensor locations with actual parameter values"""
    gdf = gpd.GeoDataFrame.from_features(_geojson['features'])
    polygon = gdf.geometry.union_all() if hasattr(gdf.geometry, 'union_all') else gdf.geometry.unary_union
    
    sensors = []
    minx, miny, maxx, maxy = polygon.bounds
    
    # Create random sensor locations
    for _ in range(num_sensors):
        point = Point(
            random.uniform(minx, maxx),
            random.uniform(miny, maxy)
        )
        if polygon.contains(point):
            # Assign random data from the dataset
            random_sample = _df.sample(1).iloc[0]
            sensors.append({
                "coordinates": [point.y, point.x],
                "values": random_sample.to_dict()
            })
    return sensors

# Initialize session state
if 'map' not in st.session_state:
    st.session_state.map = None

# Load datasets
df = load_data()
geojson = load_geojson()

# App Layout
st.title("🌦️ Sharaan Protected Area Climate Dashboard")

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Controls")
    selected_var = st.selectbox(
        "Climate Parameter",
        options=[col for col in df.columns if col != 'Date'],
        index=6  # Default to Rainfall
    )
    
    date_range = st.date_input(
        "Date Range",
        value=(df['Date'].min().date(), df['Date'].max().date()),
        min_value=df['Date'].min().date(),
        max_value=df['Date'].max().date()
    )

# Data Processing
start_date, end_date = pd.to_datetime(date_range)
filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

# Generate sensor data with actual values
sensor_data = generate_sensor_data(filtered_df, geojson)

# Prepare heatmap data
heatmap_points = []
for sensor in sensor_data:
    try:
        value = sensor['values'][selected_var]
        heatmap_points.append([
            sensor['coordinates'][0],  # lat
            sensor['coordinates'][1],  # lon
            value  # intensity
        ])
    except KeyError:
        continue

# Create map
m = folium.Map(
    location=[filtered_df['Date'].mean().latitude, filtered_df['Date'].mean().longitude],
    zoom_start=10,
    tiles="cartodbpositron"
)

# Add protected area boundary
folium.GeoJson(
    geojson,
    style_function=lambda x: {
        'fillColor': '#4a148c',
        'color': '#d81b60',
        'weight': 2,
        'fillOpacity': 0.2
    }
).add_to(m)

# Add heatmap layer
if heatmap_points:
    HeatMap(
        heatmap_points,
        radius=25,
        blur=15,
        gradient={
            '0.4': 'blue',
            '0.6': 'green',
            '0.8': 'yellow',
            '1.0': 'red'
        }
    ).add_to(m)

# Display map
st_folium(m, width=800, height=500)

# Metrics
st.subheader("📊 Data Statistics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Average", f"{filtered_df[selected_var].mean():.1f}")
with col2:
    st.metric("Maximum", f"{filtered_df[selected_var].max():.1f}")
with col3:
    st.metric("Minimum", f"{filtered_df[selected_var].min():.1f}")

# Time Series Plot
st.subheader("📈 Temporal Trends")
fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(
    data=filtered_df,
    x='Date',
    y=selected_var,
    color='#2ecc71',
    linewidth=2,
    ax=ax
)
ax.set_title(f"{selected_var} Over Time", fontsize=14)
ax.grid(True, alpha=0.2)
sns.despine()
st.pyplot(fig)

# Footer
st.markdown("---")
st.caption(f"Data updated: {df['Date'].max().strftime('%Y-%m-%d %H:%M')} | Map tiles by CartoDB")
