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
    # Read CSV and ensure proper date parsing
    df = pd.read_csv(DATA_URL)
    
    # Convert Date column to datetime
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    except KeyError:
        df = df.rename(columns={df.columns[0]: 'Date'})
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    
    return df

@st.cache_data
def load_geojson():
    response = requests.get(GEOJSON_URL)
    return response.json()

# Load datasets
try:
    df = load_data()
    geojson = load_geojson()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Generate polygon from GeoJSON
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
        index=16  # Default to Rainfall
    )
    
    # Get date range with proper type conversion
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    date_range = st.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

# Handle date range selection
try:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
except IndexError:
    st.error("Please select a date range")
    st.stop()

# Filter data
mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
filtered_df = df.loc[mask]

# Rest of the code remains the same...
