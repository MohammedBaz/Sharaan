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
    except requests.exceptions.RequestException as e:
        st.error(f"Network error loading GeoJSON: {str(e)}")
        st.stop()
    except ValueError as e: # Catches JSON decoding errors
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
        # Use unary_union which is the standard way to combine geometries
        polygon = gdf.geometry.unary_union
        if not polygon.is_valid:
             st.warning("The combined GeoJSON geometry is invalid. Heatmap points might be inaccurate.")
             polygon = polygon.buffer(0) # Attempt to fix invalid geometry

    except Exception as e:
        st.error(f"Failed to process GeoJSON features for heatmap: {e}")
        # Don't stop the whole app, just return empty points
        return []

    points = []
    # Check if polygon has bounds (it might be empty if GeoJSON was invalid)
    if polygon.is_empty:
        st.warning("Cannot generate heatmap points: GeoJSON polygon is empty.")
        return []

    minx, miny, maxx, maxy = polygon.bounds
    attempts = 0
    max_attempts = num_points * 10 # Increased max attempts further

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
        st.warning(f"Could only generate {len(points)} points within the polygon boundary after {max_attempts} attempts. Check GeoJSON validity.")

    return points

# --- App Initialization ---
st.set_page_config(layout="wide") # Use wide layout for better map visibility

# Initialize session state for the map object if it doesn't exist
if 'map' not in st.session_state:
    st.session_state.map = None

# --- Data Loading ---
# Add checks to ensure data loaded correctly before proceeding
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

    # Check if min_date and max_date are the same
    if min_date == max_date:
        # Provide a default range or adjust logic if only one date exists
        st.warning("Only data for a single date is available.")
        # Option 1: Use the single date for both start and end
        default_date_range = (min_date, max_date)
        # Option 2: Or potentially disable the date input if it doesn't make sense
        # date_range = (min_date, max_date) # Keep it fixed
    else:
        default_date_range = (min_date, max_date)


    date_range = st.date_input(
        "Select Date Range",
        value=default_date_range,
        min_value=min_date,
        max_value=max_date
    )

# --- Data Processing based on Inputs ---
# Ensure date_range has two values before unpacking
if date_range and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

    # Add time component to end_date to include the full day
    # Check if start and end dates are the same day
    if start_date.date() == end_date.date():
         end_date = end_date + pd.Timedelta(hours=23, minutes=59, seconds=59)
    else:
         # For multi-day ranges, make the end date inclusive of the whole day
         end_date = end_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)


    # Filter the DataFrame
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy() # Use .copy() to avoid SettingWithCopyWarning
else:
    st.warning("Please select a valid date range (start and end date).")
    # Fallback to the full dataframe if the date range is not set correctly
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
        st.metric("Average", f"{mean_val:.1f}" if not pd.isna(mean_val) else "N/A")
    with col2:
        st.metric("Maximum", f"{max_val:.1f}" if not pd.isna(max_val) else "N/A")
    with col3:
        st.metric("Minimum", f"{min_val:.1f}" if not pd.isna(min_val) else "N/A")
elif selected_var not in filtered_df:
     st.warning(f"Selected variable '{selected_var}' not found in the data.")
elif not pd.api.types.is_numeric_dtype(filtered_df[selected_var]):
     st.warning(f"Selected variable '{selected_var}' is not numeric and cannot be aggregated.")
else: # filtered_df is empty
    st.warning(f"No data available for '{selected_var}' in the selected date range.")


# Time Series Plot
st.subheader("📈 Temporal Trends")
if not filtered_df.empty and selected_var in filtered_df and pd.api.types.is_numeric_dtype(filtered_df[selected_var]):
    # Drop NaN values for plotting if they exist for the selected variable
    plot_df = filtered_df[['Date', selected_var]].dropna(subset=[selected_var])
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(12, 4)) # Adjusted figsize
        sns.lineplot(
            data=plot_df,
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
        st.info(f"No valid data points to plot for '{selected_var}' after removing missing values.")
else:
     # More specific message if variable exists but has no data in range
     if selected_var in df.columns:
          st.info(f"No data to display for '{selected_var}' in the selected time period.")
     # else: handled by metric section


# Map Visualization
st.subheader("🗺️ Spatial Distribution & Intensity")

# Initialize map only once and store in session state
if st.session_state.map is None and geojson:
    try:
        # Calculate centroid for initial map location
        gdf_map = gpd.GeoDataFrame.from_features(geojson['features'])
        # Ensure the geometry is valid before calculating centroid
        if not gdf_map.geometry.is_valid.all():
             gdf_map.geometry = gdf_map.geometry.buffer(0)
             if not gdf_map.geometry.is_valid.all():
                  st.warning("Could not fix invalid geometries in GeoJSON. Map centroid might be inaccurate.")

        # Handle potentially empty GeoDataFrame after fixing
        if gdf_map.empty or gdf_map.geometry.unary_union.is_empty:
             st.error("GeoJSON data resulted in empty geometry. Cannot display map.")
             st.session_state.map = "Error" # Mark map as errored
        else:
             map_centroid = gdf_map.geometry.unary_union.centroid
             start_location = [map_centroid.y, map_centroid.x]

             m = folium.Map(
                 location=start_location,
                 zoom_start=10,
                 tiles="cartodbpositron", # Using a clean base map
                 control_scale=True # Show scale bar
             )

             # Add GeoJSON layer for the protected area boundary
             # *** REMOVED TOOLTIP ARGUMENT HERE ***
             folium.GeoJson(
                 geojson,
                 name="Sharaan Protected Area",
                 style_function=lambda x: {
                     'fillColor': '#9b59b6', # Purple fill
                     'color': '#8e44ad',     # Darker purple border
                     'weight': 2,
                     'fillOpacity': 0.2
                 }
                 # tooltip=... # Removed this line
             ).add_to(m)

             # Add HeatMap layer if points were generated
             if heatmap_points:
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
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}") # Show full traceback for debugging
        st.session_state.map = "Error" # Indicate map creation failed

# Render map using streamlit-folium, checking if map object exists and isn't an error string
if isinstance(st.session_state.get('map'), folium.Map): # Check if it's actually a folium map object
    try:
        st_folium(
            st.session_state.map,
            width='100%', # Use full width
            height=500,
            key="sharaan_map", # Unique key for the map component
            returned_objects=[] # Specify if you need interactions back from the map
        )
    except Exception as e:
         st.error(f"Error rendering map with streamlit-folium: {e}")
         import traceback
         st.error(f"Traceback: {traceback.format_exc()}") # Show full traceback for debugging
         st.session_state.map = "Error" # Mark as error if rendering fails

elif st.session_state.get('map') == "Error":
     st.error("Map could not be displayed due to an earlier error during creation or rendering.")
else:
     # This case might happen if geojson was None or map initialization failed silently
     st.info("Map could not be initialized. Please check GeoJSON data source and potential errors above.")


# --- Footer ---
st.markdown("---")
last_data_point_str = "N/A"
if not df.empty and 'Date' in df.columns:
     last_data_point = df['Date'].max()
     if pd.notna(last_data_point):
          last_data_point_str = last_data_point.strftime('%Y-%m-%d')

st.caption(f"Data source: Simulated climate data | GeoJSON source: Provided URL | Last data point: {last_data_point_str}")

