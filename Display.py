import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
# No longer need streamlit_folium
import streamlit.components.v1 as components # Import components
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import random
import requests
import warnings

# Configuration
DATA_URL = "https://raw.githubusercontent.com/MohammedBaz/Sharaan/main/dataset.csv"
GEOJSON_URL = "https://raw.githubusercontent.com/MohammedBaz/Sharaan/main/Sharaan.geojson"
HEATMAP_GRADIENT = {'0.4': '#0000ff', '0.6': '#00ff00', '0.8': '#ffff00', '1.0': '#ff0000'}
MAP_HEIGHT = 550 # Define map height for components.html

@st.cache_data
def load_data():
    """Load and validate climate dataset"""
    try:
        df = pd.read_csv(DATA_URL)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if df['Date'].isnull().any():
            invalid_rows = df[df['Date'].isnull()]
            st.error(f"Invalid date formats found in data. Please check rows like: \n{invalid_rows.head()}")
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
    except requests.exceptions.RequestException as e:
        st.error(f"Network error loading GeoJSON: {str(e)}")
        st.stop()
    except ValueError as e:
        st.error(f"Error decoding GeoJSON: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"Geojson loading failed: {str(e)}")
        st.stop()

@st.cache_data
def generate_heatmap_data(_geojson, num_points=2000):
    """Generate spatial intensity data within boundaries"""
    if not _geojson or 'features' not in _geojson or not _geojson['features']:
         st.warning("GeoJSON data is missing or invalid for heatmap generation.")
         return []
    try:
        gdf = gpd.GeoDataFrame.from_features(_geojson['features'])
        polygon = gdf.geometry.union_all()
        if not polygon.is_valid:
             st.warning("The combined GeoJSON geometry is invalid. Attempting to fix.")
             polygon = polygon.buffer(0)
             if not polygon.is_valid:
                  st.error("Failed to fix invalid GeoJSON geometry. Heatmap may be incorrect or missing.")
                  return []
    except Exception as e:
        st.error(f"Failed to process GeoJSON features for heatmap: {e}")
        return []

    points = []
    if polygon.is_empty:
        st.warning("Cannot generate heatmap points: GeoJSON polygon is empty.")
        return []

    minx, miny, maxx, maxy = polygon.bounds
    attempts = 0
    max_attempts = num_points * 10

    while len(points) < num_points and attempts < max_attempts:
        point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(point):
            points.append({
                "coordinates": [point.y, point.x],
                "intensity": random.uniform(0.4, 1.0)
            })
        attempts += 1

    if len(points) < num_points:
        st.warning(f"Could only generate {len(points)} points within the polygon boundary after {max_attempts} attempts.")

    return points

# --- App Initialization ---
st.set_page_config(layout="wide")

# Initialize session state for the map HTML and error flag
if 'map_html' not in st.session_state:
    st.session_state.map_html = None
    st.session_state.map_init_error = False

# --- Data Loading ---
df = load_data()
geojson = load_geojson()

if df is None or geojson is None:
     st.error("Failed to load necessary data. Dashboard cannot proceed.")
     st.stop()

heatmap_points = generate_heatmap_data(geojson)

# --- App Layout ---
st.title("🌦️ Sharaan Protected Area Climate Dashboard")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.error("No numeric climate parameters found in the data.")
        st.stop()

    try:
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
    filtered_df = df.loc[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
else:
    st.warning("Please select a valid date range. Displaying all data.")
    filtered_df = df.copy()


# --- Main Panel Display ---

# Key Metrics
st.subheader(f"📊 Key Metrics for {selected_var}")
if not filtered_df.empty and selected_var in filtered_df and pd.api.types.is_numeric_dtype(filtered_df[selected_var]):
    col1, col2, col3 = st.columns(3)
    mean_val = filtered_df[selected_var].mean()
    max_val = filtered_df[selected_var].max()
    min_val = filtered_df[selected_var].min()
    with col1:
        st.metric("Average", f"{mean_val:.1f}" if pd.notna(mean_val) else "N/A")
    with col2:
        st.metric("Maximum", f"{max_val:.1f}" if pd.notna(max_val) else "N/A")
    with col3:
        st.metric("Minimum", f"{min_val:.1f}" if pd.notna(min_val) else "N/A")
elif selected_var not in filtered_df.columns:
     st.warning(f"Selected variable '{selected_var}' not found.")
elif not pd.api.types.is_numeric_dtype(df[selected_var]):
     st.warning(f"Selected variable '{selected_var}' is not numeric.")
else:
    st.warning(f"No data available for '{selected_var}' in the selected date range.")


# Time Series Plot
st.subheader("📈 Temporal Trends")
if not filtered_df.empty and selected_var in filtered_df and pd.api.types.is_numeric_dtype(filtered_df[selected_var]):
    plot_df = filtered_df[['Date', selected_var]].dropna(subset=[selected_var])
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.lineplot(data=plot_df, x='Date', y=selected_var, color='#2ecc71', linewidth=1.5, ax=ax)
        ax.set_title(f"{selected_var} Over Time", fontsize=14)
        ax.set_ylabel(selected_var)
        ax.set_xlabel("Date")
        ax.grid(True, linestyle='--', alpha=0.6)
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info(f"No valid data points to plot for '{selected_var}' after removing missing values.")
else:
     if selected_var in df.columns:
          st.info(f"No data to display for '{selected_var}' in the selected time period or variable is non-numeric.")


# Map Visualization
st.subheader("🗺️ Spatial Distribution & Intensity")

# Generate map HTML only ONCE if it's not in session state and no previous error
if st.session_state.map_html is None and not st.session_state.map_init_error:
    if geojson:
        try:
            gdf_map = gpd.GeoDataFrame.from_features(geojson['features'])
            if not gdf_map.geometry.is_valid.all():
                 gdf_map.geometry = gdf_map.geometry.buffer(0)
                 if not gdf_map.geometry.is_valid.all():
                      st.warning("Could not fix invalid geometries in GeoJSON.")

            combined_geometry = gdf_map.geometry.union_all()

            if gdf_map.empty or combined_geometry.is_empty:
                 st.error("GeoJSON data resulted in empty geometry. Cannot create map.")
                 st.session_state.map_init_error = True
            else:
                 map_centroid = combined_geometry.centroid
                 start_location = [map_centroid.y, map_centroid.x]

                 # Create the Folium map object
                 m = folium.Map(location=start_location, zoom_start=10, tiles="cartodbpositron", control_scale=True)

                 # Add GeoJSON layer
                 folium.GeoJson(
                     geojson,
                     name="Sharaan Protected Area",
                     style_function=lambda x: {'fillColor': '#9b59b6', 'color': '#8e44ad', 'weight': 2, 'fillOpacity': 0.2}
                 ).add_to(m)

                 # Add HeatMap layer
                 if heatmap_points:
                     HeatMap(
                         data=[[p['coordinates'][0], p['coordinates'][1], p['intensity']] for p in heatmap_points],
                         radius=20, blur=15, gradient=HEATMAP_GRADIENT, name="Simulated Intensity Heatmap"
                     ).add_to(m)

                 # Add Layer Control
                 folium.LayerControl().add_to(m)

                 # *** Generate and store the HTML ***
                 st.session_state.map_html = m._repr_html_()

        except Exception as e:
            st.error(f"Failed to create map HTML: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            st.session_state.map_init_error = True
            st.session_state.map_html = None # Ensure HTML is None if creation failed
    else:
        st.error("GeoJSON data not available, cannot create map.")
        st.session_state.map_init_error = True


# Render map HTML using components.html if it exists and no init error
if st.session_state.get('map_html') and not st.session_state.get('map_init_error'):
    try:
        # *** Use components.html to render the stored map HTML ***
        components.html(st.session_state.map_html, height=MAP_HEIGHT)
    except Exception as e:
         st.error(f"Error rendering map HTML component: {e}")
         # Optionally clear the HTML if rendering fails consistently
         # st.session_state.map_html = None
         # st.session_state.map_init_error = True

# Display message if map creation failed or is pending
elif st.session_state.get('map_init_error'):
     st.error("Map could not be displayed due to an earlier error during creation.")
else:
     st.info("Map is initializing or required GeoJSON data is missing.")


# --- Footer ---
st.markdown("---")
last_data_point_str = "N/A"
if not df.empty and 'Date' in df.columns:
     last_data_point = df['Date'].max()
     if pd.notna(last_data_point):
          last_data_point_str = last_data_point.strftime('%Y-%m-%d')

st.caption(f"Data source: Simulated climate data | GeoJSON source: Provided URL | Last data point: {last_data_point_str}")
