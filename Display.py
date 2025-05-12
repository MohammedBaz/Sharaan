import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
from sklearn.preprocessing import MinMaxScaler # For normalizing heatmap intensity
import random
import requests
import warnings

# Configuration
DATA_URL = "https://raw.githubusercontent.com/MohammedBaz/Sharaan/main/dataset.csv"
GEOJSON_URL = "https://raw.githubusercontent.com/MohammedBaz/Sharaan/main/Sharaan.geojson"
# Adjusted gradient for potentially normalized values (0-1)
HEATMAP_GRADIENT = {'0.2': 'blue', '0.4': 'lime', '0.6': 'yellow', '0.8': 'orange', '1.0': 'red'}
MAP_HEIGHT = 550 # Define map height for components.html
NUM_HEATMAP_POINTS = 1500 # Number of points for the heatmap overlay

# --- Caching Functions ---

@st.cache_data
def load_data():
    """Load and validate climate dataset"""
    try:
        df = pd.read_csv(DATA_URL)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if df['Date'].isnull().any():
            invalid_rows = df[df['Date'].isnull()]
            st.error(f"Invalid date formats found: \n{invalid_rows.head()}")
            st.stop()
        # Ensure data is numeric where expected
        for col in df.columns:
             if col != 'Date':
                  df[col] = pd.to_numeric(df[col], errors='coerce')
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

# --- Helper Functions ---

# Removed caching as it now depends on filtered_df which changes
def generate_spatial_points(_geojson, num_points=NUM_HEATMAP_POINTS):
    """Generate random spatial points within the GeoJSON boundaries."""
    points_coords = []
    if not _geojson or 'features' not in _geojson or not _geojson['features']:
         st.warning("GeoJSON data missing/invalid for point generation.")
         return points_coords
    try:
        gdf = gpd.GeoDataFrame.from_features(_geojson['features'])
        polygon = gdf.geometry.union_all()
        if not polygon.is_valid:
             polygon = polygon.buffer(0)
        if polygon.is_empty:
             st.warning("Cannot generate points: GeoJSON polygon is empty.")
             return points_coords

        minx, miny, maxx, maxy = polygon.bounds
        attempts = 0
        max_attempts = num_points * 10

        while len(points_coords) < num_points and attempts < max_attempts:
            point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            if polygon.contains(point):
                points_coords.append([point.y, point.x]) # Lat, Lon
            attempts += 1

        if len(points_coords) < num_points:
            st.warning(f"Generated {len(points_coords)}/{num_points} points within boundary.")

    except Exception as e:
        st.error(f"Failed to generate spatial points: {e}")
    return points_coords

def normalize_value(value, min_val, max_val):
    """Normalize a value between 0 and 1."""
    if max_val == min_val: # Avoid division by zero if all values are the same
        return 0.5 # Return a neutral value
    return (value - min_val) / (max_val - min_val)

# --- App Initialization ---
st.set_page_config(layout="wide")

# --- Data Loading ---
df = load_data()
geojson = load_geojson()

if df is None or geojson is None:
     st.error("Failed to load necessary data. Dashboard cannot proceed.")
     st.stop()

# Generate base spatial points ONCE and cache them
@st.cache_data # Cache the spatial points generation
def get_cached_spatial_points(_geojson):
     return generate_spatial_points(_geojson, NUM_HEATMAP_POINTS)

spatial_points_coords = get_cached_spatial_points(geojson)


# --- App Layout ---
st.title("🌦️ Sharaan Protected Area Climate Dashboard")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    # Filter out non-numeric columns before presenting options
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.error("No numeric climate parameters found in the data.")
        st.stop()

    try:
        # Default to 'Rainfall' if available, otherwise first numeric column
        default_index = numeric_cols.index('Rainfall')
    except ValueError:
        default_index = 0

    selected_var = st.selectbox(
        "Climate Parameter",
        options=numeric_cols,
        index=default_index,
        key='climate_parameter_selector'
    )

    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()

    if min_date >= max_date:
        st.info("Data available for only one day.")
        default_date_range = (min_date, max_date)
    else:
        default_date_range = (min_date, max_date)

    date_range = st.date_input(
        "Select Date Range",
        value=default_date_range,
        min_value=min_date,
        max_value=max_date,
        key='date_range_selector'
    )

# --- Data Processing based on Inputs ---
if date_range and len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1]).replace(hour=23, minute=59, second=59)
    # Filter main dataframe
    filtered_df = df.loc[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
    # Drop rows with NaN in the selected variable for calculations
    filtered_var_df = filtered_df[[selected_var]].dropna()
else:
    st.warning("Please select a valid date range. Displaying all data.")
    filtered_df = df.copy()
    filtered_var_df = filtered_df[[selected_var]].dropna()


# --- Main Panel Display ---

# Key Metrics
st.subheader(f"📊 Key Metrics for {selected_var}")
if not filtered_var_df.empty:
    col1, col2, col3 = st.columns(3)
    mean_val = filtered_var_df[selected_var].mean()
    max_val = filtered_var_df[selected_var].max()
    min_val = filtered_var_df[selected_var].min()
    with col1:
        st.metric("Average", f"{mean_val:.1f}" if pd.notna(mean_val) else "N/A")
    with col2:
        st.metric("Maximum", f"{max_val:.1f}" if pd.notna(max_val) else "N/A")
    with col3:
        st.metric("Minimum", f"{min_val:.1f}" if pd.notna(min_val) else "N/A")
else:
    st.warning(f"No valid data available for '{selected_var}' in the selected date range after removing missing values.")


# --- Layout for Plot and Map ---
col_plot, col_map = st.columns(2) # Create two columns

with col_plot:
    st.subheader("📈 Temporal Trends")
    if not filtered_var_df.empty:
        # Use the original filtered_df for plotting to keep dates aligned, but only plot valid points
        plot_df_temporal = filtered_df[['Date', selected_var]].dropna(subset=[selected_var])
        if not plot_df_temporal.empty:
            fig, ax = plt.subplots(figsize=(10, 4)) # Adjust size as needed
            sns.lineplot(data=plot_df_temporal, x='Date', y=selected_var, color='#2ecc71', linewidth=1.5, ax=ax)
            ax.set_title(f"{selected_var} Over Time", fontsize=12)
            ax.set_ylabel(selected_var)
            ax.set_xlabel("Date")
            ax.grid(True, linestyle='--', alpha=0.6)
            sns.despine()
            plt.tight_layout()
            st.pyplot(fig)
        else:
             st.info(f"No valid data points to plot for '{selected_var}'.")
    else:
         st.info(f"No data to display for '{selected_var}' in the selected time period.")

with col_map:
    st.subheader("🗺️ Spatial Distribution")
    st.markdown(f"Heatmap intensity represents the **normalized average {selected_var}** for the period.")

    map_html = None # Initialize map_html to None for this run
    map_init_error = False

    # --- Map Generation (on each run) ---
    if geojson and spatial_points_coords: # Check if base data is available
        try:
            # Calculate average value for the selected variable in the filtered period
            avg_value = filtered_var_df[selected_var].mean()

            # Get overall min/max for the selected variable from the *entire* dataset for consistent normalization
            overall_min = df[selected_var].min()
            overall_max = df[selected_var].max()

            # Normalize the average value (0 to 1)
            normalized_intensity = 0.5 # Default if no data or single value
            if pd.notna(avg_value) and pd.notna(overall_min) and pd.notna(overall_max):
                 normalized_intensity = normalize_value(avg_value, overall_min, overall_max)

            # Create the list of points for the heatmap with the calculated intensity
            heatmap_data = [[lat, lon, normalized_intensity] for lat, lon in spatial_points_coords]

            # --- Create Folium Map Object ---
            gdf_map = gpd.GeoDataFrame.from_features(geojson['features'])
            if not gdf_map.geometry.is_valid.all():
                 gdf_map.geometry = gdf_map.geometry.buffer(0)

            combined_geometry = gdf_map.geometry.union_all()

            if gdf_map.empty or combined_geometry.is_empty:
                 st.error("GeoJSON data resulted in empty geometry.")
                 map_init_error = True
            else:
                 map_centroid = combined_geometry.centroid
                 start_location = [map_centroid.y, map_centroid.x]

                 m = folium.Map(location=start_location, zoom_start=10, tiles="cartodbpositron", control_scale=True)

                 # Add GeoJSON layer (persistent style)
                 folium.GeoJson(
                     geojson,
                     name="Sharaan Protected Area",
                     style_function=lambda x: {'fillColor': '#9b59b6', 'color': '#8e44ad', 'weight': 1.5, 'fillOpacity': 0.15}
                 ).add_to(m)

                 # Add HeatMap layer (dynamic intensity)
                 if heatmap_data:
                     HeatMap(
                         data=heatmap_data,
                         radius=18, # Adjusted parameters
                         blur=12,
                         gradient=HEATMAP_GRADIENT,
                         name=f"Avg {selected_var} Intensity"
                     ).add_to(m)

                 folium.LayerControl().add_to(m)

                 # Generate HTML for the map
                 map_html = m._repr_html_()

        except Exception as e:
            st.error(f"Failed to create map: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            map_init_error = True

    # --- Render Map HTML ---
    if map_html and not map_init_error:
        try:
            components.html(map_html, height=MAP_HEIGHT)
        except Exception as e:
             st.error(f"Error rendering map HTML component: {e}")
    elif map_init_error:
         st.error("Map could not be displayed due to an error during creation.")
    else:
         st.info("Map could not be generated. Check GeoJSON and data availability.")


# --- Footer ---
st.markdown("---")
last_data_point_str = "N/A"
if not df.empty and 'Date' in df.columns:
     last_data_point = df['Date'].max()
     if pd.notna(last_data_point):
          last_data_point_str = last_data_point.strftime('%Y-%m-%d')

st.caption(f"Data source: Simulated climate data | GeoJSON source: Provided URL | Last data point: {last_data_point_str}")

