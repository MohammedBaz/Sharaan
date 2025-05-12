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
DATA_URL = "https://raw.githubusercontent.com/MohammedBaz/Sharaan/main/your_data.csv"
GEOJSON_URL = "https://raw.githubusercontent.com/MohammedBaz/Sharaan/main/Sharaan.geojson"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, parse_dates=['Date'], dayfirst=True)
    # Flatten multi-index columns
    df.columns = [f"{col[0]} ({col[1]})" if col[1] else col[0] for col in df.columns]
    return df

@st.cache_data
def load_geojson():
    response = requests.get(GEOJSON_URL)
    return response.json()

# Load datasets
df = load_data()
geojson = load_geojson()

# Generate sensor locations within protected area
gdf = gpd.GeoDataFrame.from_features(geojson['features'])
polygon = gdf.unary_union

def generate_sensor_locations(num_sensors=50):
    sensors = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(sensors) < num_sensors:
        point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(point):
            sensors.append((point.y, point.x))
    return sensors

sensor_locations = generate_sensor_locations()

# Streamlit App
st.title("Sharaan Protected Area Climate Monitoring")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    parameter = st.selectbox(
        "Select Parameter",
        options=[col for col in df.columns if col != 'Date'],
        index=6  # Default to Rainfall (Sum)
    )
    
    date_range = st.date_input(
        "Select Date Range",
        value=(df['Date'].min(), df['Date'].max())
    )

# Filter data based on selection
mask = (df['Date'] >= pd.to_datetime(date_range[0])) & (df['Date'] <= pd.to_datetime(date_range[1]))
filtered_df = df.loc[mask]

# Metrics row
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Average", f"{filtered_df[parameter].mean():.1f}")
with col2:
    st.metric("Maximum", f"{filtered_df[parameter].max():.1f}")
with col3:
    st.metric("Minimum", f"{filtered_df[parameter].min():.1f}")

# Time Series Plot
st.subheader(f"Time Series: {parameter}")
fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=filtered_df, x='Date', y=parameter, ax=ax)
ax.set_xlabel("Date")
ax.set_ylabel(parameter)
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Map Visualization
st.subheader("Sensor Network Map")

# Create Folium map
m = folium.Map(location=[25.5, 37.5], zoom_start=9, tiles="CartoDB positron")

# Add protected area boundary
folium.GeoJson(
    geojson,
    style_function=lambda x: {
        'fillColor': '#2c3e50',
        'color': '#e74c3c',
        'weight': 1.5,
        'fillOpacity': 0.1
    }
).add_to(m)

# Add sensor markers with latest readings
latest_data = df.iloc[-1]  # Get latest readings
for (lat, lon) in sensor_locations:
    folium.CircleMarker(
        location=[lat, lon],
        radius=5,
        color='#3498db',
        fill=True,
        fill_color='#3498db',
        popup=f"{parameter}: {latest_data[parameter]:.1f}"
    ).add_to(m)

# Display map
st_folium(m, width=800, height=500)

# Data summary
st.subheader("Data Summary")
st.dataframe(filtered_df.describe(), use_container_width=True)

# Footer
st.markdown("---")
st.caption("🌍 Sharaan Protected Area Climate Monitoring System | Data updated: " + str(df['Date'].max().date()))
