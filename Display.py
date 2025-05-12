import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from datetime import datetime
from shapely.geometry import Point
import random

# Custom CSS styling
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .header-text { 
        color: #2c3e50;
        font-size: 2.5rem !important;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px;
    }
    .map-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .chart-container {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Dashboard Header
st.markdown('<h1 class="header-text">🌧️ Sharaan Climate Dashboard</h1>', unsafe_allow_html=True)

# ========== Sidebar ==========
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/566/566985.png", width=80)
    st.title("Settings")
    
    # Date range selector
    date_range = st.date_input(
        "Select Date Range",
        value=[datetime(2016, 1, 1), datetime(2016, 8, 1)],
        min_value=datetime(2016, 1, 1),
        max_value=datetime(2016, 12, 31)
    )
    
    # Parameter selection
    parameter = st.selectbox(
        "Climate Parameter",
        ["Precipitation (mm)", "Temperature (°C)", "Humidity (%)", "Wind Speed (m/s)"],
        index=0
    )
    
    st.markdown("---")
    st.caption("🌍 Spatial Analysis Settings")
    map_style = st.selectbox(
        "Map Style",
        ["CartoDB Positron", "OpenStreetMap", "Stamen Terrain"]
    )

# ========== Main Content ==========
# Metrics Row
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-card">📊 **Average**<br><h2>4.4 mm</h2></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card">🔥 **Maximum**<br><h2>13.8 mm</h2></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card">❄️ **Minimum**<br><h2>0.1 mm</h2></div>', unsafe_allow_html=True)

# Main Content Tabs
tab1, tab2 = st.tabs(["📈 Temporal Analysis", "🗺️ Spatial Analysis"])

with tab1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Generate time series data
    dates = pd.date_range(start=date_range[0], end=date_range[1], freq='D')
    precipitation = np.random.gamma(2, 2, len(dates)).round(1)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(
        x=dates,
        y=precipitation,
        color='#3498db',
        linewidth=2.5
    )
    
    # Style plot
    ax.set_title(f"Daily {parameter} Variation", fontsize=16, pad=20)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(parameter, fontsize=12)
    ax.grid(True, alpha=0.3)
    sns.despine()
    plt.tight_layout()
    
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="map-container">', unsafe_allow_html=True)
    
    # Create Folium map
    m = folium.Map(location=[25.5, 37.5], zoom_start=9, 
                  tiles=map_style if map_style != "CartoDB Positron" else "CartoDB positron")
    
    # Generate heatmap data
    heat_data = [[25.3 + np.random.rand()/2, 37.2 + np.random.rand()/2, np.random.rand()] 
                for _ in range(200)]
    
    # Add heatmap layer
    HeatMap(
        heat_data,
        radius=15,
        blur=20,
        gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}
    ).add_to(m)
    
    # Add protected area boundary
    folium.GeoJson(
        "https://raw.githubusercontent.com/MohammedBaz/Sharaan/main/Sharaan.geojson",
        style_function=lambda x: {
            'fillColor': '#2c3e50',
            'color': '#e74c3c',
            'weight': 1.5,
            'fillOpacity': 0.1
        }
    ).add_to(m)
    
    # Display map
    st_folium(m, width=800, height=500)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("""
🔍 **Data Source**: Simulated demonstration data  
🗺️ **Base Map**: © OpenStreetMap contributors  
📅 **Last Updated**: {date}  
""".format(date=datetime.now().strftime("%Y-%m-%d %H:%M")))
