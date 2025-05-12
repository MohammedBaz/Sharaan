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
HEATMAP_GRADIENT = {'0.4': '#0000ff', '0.6': '#00ff00', '0.8': '#ffff00', '1.0': '#ff0000'}

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
def generate_heatmap_data(_geojson, num_points=2000):
    """Generate spatial intensity data within boundaries"""
    gdf = gpd.GeoDataFrame.from_features(_geojson['features'])
    polygon = gdf.geometry.union_all() if hasattr(gdf.geometry, 'union_all') else gdf.geometry.unary_union
    
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    attempts = 0
    
    while len(points) < num_points and attempts < num_points * 2:
        point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(point):
            points.append({
                "coordinates": [point.y, point.x],
                "intensity": random.uniform(0.4, 1.0)
            })
        attempts += 1
        
    return points

# Initialize session state
if 'map' not in st.session_state:
    st.session_state.map = None

# Load datasets
df = load_data()
geojson = load_geojson()
heatmap_points = generate_heatmap_data(geojson)

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

# Metrics
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

# Map Visualization
st.subheader("🗺️ Spatial Distribution")

if not st.session_state.map:
    m = folium.Map(
        location=[centroid.y.mean(), centroid.x.mean()],
        zoom_start=10,
        tiles="cartodbpositron",
        control_scale=True
    )
    
    folium.GeoJson(
        geojson,
        style_function=lambda x: {
            'fillColor': '#4a148c',
            'color': '#d81b60',
            'weight': 2,
            'fillOpacity': 0.2
        },
        name="Protected Area"
    ).add_to(m)

    HeatMap(
        data=[[p['coordinates'][0], p['coordinates'][1], p['intensity'] for p in heatmap_points],
        radius=25,
        blur=20,
        gradient=HEATMAP_GRADIENT,
        name="Intensity Heatmap"
    ).add_to(m)

    folium.LayerControl().add_to(m)
    st.session_state.map = m

map_container = st_folium(
    st.session_state.map,
    width=800,
    height=500,
    key="main_map",
    returned_objects=[]
)

# Footer
st.markdown("---")
st.caption(f"Data updated: {df['Date'].max().strftime('%Y-%m-%d %H:%M')} | Map tiles by CartoDB")
