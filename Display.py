import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap # Import HeatMap
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
            # Find rows with invalid dates to show specific examples if needed
            invalid_rows = df[df['Date'].isnull()]
            st.error(f"Invalid date formats found in data. Please check rows like: \n{invalid_rows.head()}")
            # Optionally display problematic rows: st.dataframe(invalid_rows)
            st.stop() # Stop execution if dates are invalid

        return df.sort_values('Date')
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        st.stop()

@st.cache_data
def load_geojson():
    """Load protected area boundaries"""
    try:
        response = requests.get(GEOJSON_URL)
        response.raise_for_status() # Raise an exception for bad status codes
        return response.json()
    except Exception as e:
        st.error(f"Geojson loading failed: {str(e)}")
        st.stop()

@st.cache_data
def generate_heatmap_data(_geojson, num_points=2000):
    """Generate spatial intensity data within boundaries"""
    try:
        gdf = gpd.GeoDataFrame.from_features(_geojson['features'])
        # Use unary_union which is the standard way to combine geometries
        polygon = gdf.geometry.unary_union
    except Exception as e:
        st.error(f"Failed to process GeoJSON features: {e}")
        st.stop()

    points = []
    minx, miny, maxx, maxy = polygon.bounds
    attempts = 0
    max_attempts = num_points * 5 # Increase max attempts to ensure points are found

    while len(points) < num_points and attempts < max_attempts:
        point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        # Check if the generated point is actually within the polygon
        if polygon.contains(point):
            points.append({
                "coordinates": [point.y, point.x], # Lat, Lon format for Folium
                "intensity": random.uniform(0.4, 1.0)
            })
        attempts += 1

    if len(points) < num_points:
        st.warning(f"Could only generate {len(points)} points within the polygon boundary after {max_attempts} attempts.")

    return points

# --- App Initialization ---
st.set_page_config(layout="wide") # Use wide layout for better map visibility

# Initialize session state for the map object if it doesn't exist
if 'map' not in st.session_state:
    st.session_state.map = None

# --- Data Loading ---
df = load_data()
geojson = load_geojson()
heatmap_points = generate_heatmap_data(geojson)

# --- App Layout ---
st.title("🌦️ Sharaan Protected Area Climate Dashboard")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    # Dynamically get numeric columns for selection, excluding Date/Time columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.error("No numeric climate parameters found in the data.")
        st.stop()

    # Try to find 'Rainfall' or default to the first numeric column
    default_index = numeric_cols.index('Rainfall') if 'Rainfall' in numeric_cols else 0

    selected_var = st.selectbox(
        "Climate Parameter",
        options=numeric_cols,
        index=default_index
    )

    # Ensure date inputs are valid Date objects
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()

    date_range = st.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

# --- Data Processing based on Inputs ---
# Ensure date_range has two values before unpacking
if len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

    # Add time component to end_date to include the full day
    end_date = end_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    # Filter the DataFrame
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy() # Use .copy() to avoid SettingWithCopyWarning
else:
    st.warning("Please select a valid date range.")
    filtered_df = df.copy() # Default to full dataframe if range is incomplete

# --- Main Panel Display ---

# Key Metrics
st.subheader(f"📊 Key Metrics for {selected_var}")
if not filtered_df.empty and selected_var in filtered_df:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average", f"{filtered_df[selected_var].mean():.1f}")
    with col2:
        st.metric("Maximum", f"{filtered_df[selected_var].max():.1f}")
    with col3:
        st.metric("Minimum", f"{filtered_df[selected_var].min():.1f}")
else:
    st.warning(f"No data available for '{selected_var}' in the selected date range.")


# Time Series Plot
st.subheader("📈 Temporal Trends")
if not filtered_df.empty and selected_var in filtered_df:
    fig, ax = plt.subplots(figsize=(12, 4)) # Adjusted figsize
    sns.lineplot(
        data=filtered_df,
        x='Date',
        y=selected_var,
        color='#2ecc71',
        linewidth=2,
        ax=ax
    )
    ax.set_title(f"{selected_var} Over Time", fontsize=14)
    ax.set_ylabel(selected_var)
    ax.set_xlabel("Date")
    ax.grid(True, linestyle='--', alpha=0.6)
    sns.despine()
    plt.tight_layout() # Adjust layout
    st.pyplot(fig)
else:
     st.info("No data to display for the time series plot based on current selections.")


# Map Visualization
st.subheader("🗺️ Spatial Distribution & Intensity")

# Initialize map only once and store in session state
if st.session_state.map is None and geojson:
    try:
        # Calculate centroid for initial map location
        gdf_map = gpd.GeoDataFrame.from_features(geojson['features'])
        map_centroid = gdf_map.geometry.unary_union.centroid
        start_location = [map_centroid.y, map_centroid.x]

        m = folium.Map(
            location=start_location,
            zoom_start=10,
            tiles="cartodbpositron", # Using a clean base map
            control_scale=True # Show scale bar
        )

        # Add GeoJSON layer for the protected area boundary
        folium.GeoJson(
            geojson,
            name="Sharaan Protected Area",
            style_function=lambda x: {
                'fillColor': '#9b59b6', # Purple fill
                'color': '#8e44ad',     # Darker purple border
                'weight': 2,
                'fillOpacity': 0.2
            },
            tooltip=folium.features.GeoJsonTooltip(fields=['Name'], aliases=['Area:'], localize=True) # Example tooltip
        ).add_to(m)

        # Add HeatMap layer if points were generated
        if heatmap_points:
             # *** CORRECTED THIS LINE ***
             # Removed extra brackets around the list comprehension
            HeatMap(
                data=[[p['coordinates'][0], p['coordinates'][1], p['intensity']] for p in heatmap_points],
                radius=20, # Adjusted radius
                blur=15,   # Adjusted blur
                gradient=HEATMAP_GRADIENT,
                name="Simulated Intensity Heatmap"
            ).add_to(m)

        # Add Layer Control to toggle layers
        folium.LayerControl().add_to(m)

        # Store the initialized map in session state
        st.session_state.map = m

    except Exception as e:
        st.error(f"Failed to create map: {e}")
        st.session_state.map = "Error" # Indicate map creation failed

# Render map using streamlit-folium, checking if map object exists and isn't an error string
if st.session_state.map and st.session_state.map != "Error":
    st_folium(
        st.session_state.map,
        width='100%', # Use full width
        height=500,
        key="sharaan_map", # Unique key for the map component
        returned_objects=[] # Specify if you need interactions back from the map
    )
elif st.session_state.map == "Error":
     st.error("Map could not be displayed due to an earlier error.")
else:
     st.info("Map is initializing or GeoJSON data is missing.")


# --- Footer ---
st.markdown("---")
st.caption(f"Data source: Simulated climate data | GeoJSON source: Provided URL | Last data point: {df['Date'].max().strftime('%Y-%m-%d') if not df.empty else 'N/A'}")
