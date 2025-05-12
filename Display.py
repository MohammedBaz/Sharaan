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

# Load data from GitHub
DATA_URL = "https://raw.githubusercontent.com/MohammedBaz/Sharaan/main/dataset.csv"
GEOJSON_URL = "https://raw.githubusercontent.com/MohammedBaz/Sharaan/main/Sharaan.geojson"

@st.cache_data
def load_data():
    # Load and validate dataset
    try:
        df = pd.read_csv(DATA_URL)
        
        # Convert Date column with validation
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
        if df['Date'].isnull().any():
            bad_dates = df[df['Date'].isnull()]['Date'].unique()
            st.error(f"Invalid date format found: {bad_dates}")
            st.stop()
            
        return df.sort_values('Date')
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        st.stop()

@st.cache_data
def load_geojson():
    # Load and validate geospatial data
    try:
        response = requests.get(GEOJSON_URL)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to load geojson: {str(e)}")
        st.stop()

# Load datasets
df = load_data()
geojson = load_geojson()

# Process geospatial data with updated methods
gdf = gpd.GeoDataFrame.from_features(geojson['features'])
try:
    polygon = gdf.geometry.union_all()  # Modern method
except AttributeError:
    polygon = gdf.geometry.unary_union  # Legacy fallback

def generate_sensor_locations(num_sensors=50):
    # Generate valid sensor locations
    sensors = []
    minx, miny, maxx, maxy = polygon.bounds
    attempts = 0
    
    while len(sensors) < num_sensors and attempts < num_sensors * 2:
        point = Point(
            random.uniform(minx, maxx),
            random.uniform(miny, maxy)
        )
        if polygon.contains(point):
            sensors.append((point.y, point.x))
        attempts += 1
        
    if len(sensors) < num_sensors:
        st.warning(f"Only generated {len(sensors)}/{num_sensors} valid sensor locations")
    return sensors
    
    while len(sensors) < num_sensors and attempts < num_sensors * 2:
        point = Point(
            random.uniform(minx, maxx),
            random.uniform(miny, maxy)
        if polygon.contains(point):
            sensors.append((point.y, point.x))
        attempts += 1
        
    if len(sensors) < num_sensors:
        st.warning(f"Only generated {len(sensors)}/{num_sensors} valid sensor locations")
    return sensors

sensor_locations = generate_sensor_locations()

# Streamlit App
st.title("🌍 Sharaan Protected Area Climate Monitoring")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Parameter selection
    parameter = st.selectbox(
        "Select Monitoring Parameter",
        options=[col for col in df.columns if col != 'Date'],
        index=16,
        help="Choose climate parameter to visualize"
    )
    
    # Date range selection with validation
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    date_range = st.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Select analysis period"
    )

# Convert dates to pandas types
try:
    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1])
except IndexError:
    st.error("Please select both start and end dates")
    st.stop()

# Filter data with bounds checking
filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
if filtered_df.empty:
    st.error("No data available for selected dates")
    st.stop()

# Dashboard Metrics
st.subheader("📊 Summary Metrics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Average", f"{filtered_df[parameter].mean():.1f}")
with col2:
    st.metric("Maximum", f"{filtered_df[parameter].max():.1f}")
with col3:
    st.metric("Minimum", f"{filtered_df[parameter].min():.1f}")

# Time Series Visualization
st.subheader("📈 Temporal Analysis")
fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(
    data=filtered_df,
    x='Date',
    y=parameter,
    color='#2ecc71',
    linewidth=2,
    ax=ax
)
ax.set_title(f"{parameter} Trend", fontsize=14)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel(parameter, fontsize=12)
ax.grid(True, alpha=0.2)
sns.despine()
st.pyplot(fig)

# Spatial Visualization
st.subheader("🗺️ Spatial Distribution")
m = folium.Map(location=[25.5, 37.5], zoom_start=9, tiles="CartoDB positron")

# Add protected area boundary
folium.GeoJson(
    geojson,
    style_function=lambda x: {
        'fillColor': '#34495e',
        'color': '#e74c3c',
        'weight': 1.5,
        'fillOpacity': 0.15
    },
    name="Protected Area"
).add_to(m)

# Add sensor network
latest_readings = df.iloc[-1]
for (lat, lon) in sensor_locations:
    folium.CircleMarker(
        location=[lat, lon],
        radius=6,
        color='#3498db',
        fill=True,
        fill_opacity=0.7,
        popup=f"""<b>{parameter}</b><br>
                  Current: {latest_readings[parameter]:.1f}<br>
                  Max: {filtered_df[parameter].max():.1f}<br>
                  Min: {filtered_df[parameter].min():.1f}"""
    ).add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Display map
st_folium(m, width=800, height=500)

# Data Summary
st.subheader("🔍 Data Preview")
st.dataframe(
    filtered_df.describe(),
    use_container_width=True,
    height=300
)

# Footer
st.markdown("---")
st.caption(f"""
**Data Source**: Sharaan Protected Area Monitoring Network  
**Last Updated**: {df['Date'].max().strftime('%Y-%m-%d %H:%M')}  
**Sensor Locations**: {len(sensor_locations)} active sensors  
**Map Attribution**: © OpenStreetMap contributors  
""")
