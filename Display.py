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
import warnings # Import warnings module

# Suppress the specific Shapely deprecation warning if needed, though fixing the code is better
# warnings.filterwarnings("ignore", category=UserWarning, message="The Shapely GEOS version")

# Configuration
DATA_URL = "https://raw.githubusercontent.com/MohammedBaz/Sharaan/main/dataset.csv"
GEOJSON_URL = "https://raw.githubusercontent.com/MohammedBaz/Sharaan/main/Sharaan.geojson"
HEATMAP_GRADIENT = {'0.4': '#0000ff', '0.6': '#00ff00', '0.8': '#ffff00', '1.0': '#ff0000'}

@st.cache_data
def load_data():
    """Load and validate climate dataset"""
    try:
        df = pd.read_csv(DATA_URL)
        # Specify date format for robustness if known, otherwise let pandas infer
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce') #, format='%Y-%m-%d %H:%M:%S' # Example format

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
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
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
        # Use union_all() method instead of unary_union attribute
        polygon = gdf.geometry.union_all()
        if not polygon.is_valid:
             st.warning("The combined GeoJSON geometry is invalid. Attempting to fix.")
             polygon = polygon.buffer(0) # Attempt to fix invalid geometry
             if not polygon.is_valid:
                  st.error("Failed to fix invalid GeoJSON geometry. Heatmap may be incorrect or missing.")
                  return [] # Return empty if fixing failed

    except Exception as e:
        st.error(f"Failed to process GeoJSON features for heatmap: {e}")
        return []

    points = []
    if polygon.is_empty:
        st.warning("Cannot generate heatmap points: GeoJSON polygon is empty.")
        return []

    minx, miny, maxx, maxy = polygon.bounds
    attempts = 0
    max_attempts = num_points * 10 # Increased max attempts

    while len(points) < num_points and attempts < max_attempts:
        # Generate point within the bounding box
        point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        # Check if the point is actually within the polygon geometry
        if polygon.contains(point):
            points.append({
                "coordinates": [point.y, point.x], # Lat, Lon format for Folium
                "intensity": random.uniform(0.4, 1.0) # Example intensity
            })
        attempts += 1

    if len(points) < num_points:
        st.warning(f"Could only generate {len(points)} points within the polygon boundary after {max_attempts} attempts. Check GeoJSON validity/complexity.")

    return points

# --- App Initialization ---
st.set_page_config(layout="wide") # Use wide layout

# Initialize session state for the map object if it doesn't exist
if 'map' not in st.session_state:
    st.session_state.map = None
    st.session_state.map_init_error = False # Flag for map creation error

# --- Data Loading ---
# Use cached functions
df = load_data()
geojson = load_geojson()

# Check if data loading failed (functions use st.stop() on error)
if df is None or geojson is None:
     st.error("Failed to load necessary data. Dashboard cannot proceed.")
     st.stop() # Stop execution if core data is missing

# Generate heatmap points (only if geojson is valid)
heatmap_points = generate_heatmap_data(geojson)

# --- App Layout ---
st.title("🌦️ Sharaan Protected Area Climate Dashboard")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    # Get numeric columns dynamically
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.error("No numeric climate parameters found in the data.")
        st.stop()

    # Set default selection (e.g., 'Rainfall' if exists)
    try:
        default_index = numeric_cols.index('Rainfall')
    except ValueError:
        default_index = 0 # Default to the first numeric column if 'Rainfall' not found

    selected_var = st.selectbox(
        "Climate Parameter",
        options=numeric_cols,
        index=default_index,
        key='climate_parameter_selector' # Add key for stability
    )

    # Get date range from data
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()

    # Handle case where min and max date are the same
    if min_date >= max_date:
        st.info("Data available for only one day.")
        # Set default range to that single day, disable input?
        default_date_range = (min_date, max_date)
        # Consider disabling date input if only one day: disabled=True
    else:
        default_date_range = (min_date, max_date)

    date_range = st.date_input(
        "Select Date Range",
        value=default_date_range,
        min_value=min_date,
        max_value=max_date,
        key='date_range_selector' # Add key for stability
    )

# --- Data Processing based on Inputs ---
# Ensure date_range has two valid dates before processing
if date_range and len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

    # Make end_date inclusive of the whole day
    end_date = end_date.replace(hour=23, minute=59, second=59)

    # Filter the DataFrame
    # Use .loc for potentially better performance and clarity
    filtered_df = df.loc[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy() # Use .copy()
else:
    st.warning("Please select a valid date range (start and end date). Displaying all data.")
    filtered_df = df.copy() # Fallback to the full dataframe


# --- Main Panel Display ---

# Key Metrics
st.subheader(f"📊 Key Metrics for {selected_var}")
if not filtered_df.empty and selected_var in filtered_df and pd.api.types.is_numeric_dtype(filtered_df[selected_var]):
    col1, col2, col3 = st.columns(3)
    # Calculate metrics, handle potential NaN results if filtered_df is empty for the var
    mean_val = filtered_df[selected_var].mean()
    max_val = filtered_df[selected_var].max()
    min_val = filtered_df[selected_var].min()
    with col1:
        st.metric("Average", f"{mean_val:.1f}" if pd.notna(mean_val) else "N/A")
    with col2:
        st.metric("Maximum", f"{max_val:.1f}" if pd.notna(max_val) else "N/A")
    with col3:
        st.metric("Minimum", f"{min_val:.1f}" if pd.notna(min_val) else "N/A")
# Handle cases where the variable might not be numeric or data is empty
elif selected_var not in filtered_df.columns:
     st.warning(f"Selected variable '{selected_var}' not found in the data columns.")
elif not pd.api.types.is_numeric_dtype(df[selected_var]): # Check original df dtype
     st.warning(f"Selected variable '{selected_var}' is not numeric and cannot be aggregated.")
else: # filtered_df is empty
    st.warning(f"No data available for '{selected_var}' in the selected date range.")


# Time Series Plot
st.subheader("📈 Temporal Trends")
if not filtered_df.empty and selected_var in filtered_df and pd.api.types.is_numeric_dtype(filtered_df[selected_var]):
    # Drop rows where the selected variable is NaN before plotting
    plot_df = filtered_df[['Date', selected_var]].dropna(subset=[selected_var])
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(12, 4)) # Adjusted figsize
        sns.lineplot(
            data=plot_df,
            x='Date',
            y=selected_var,
            color='#2ecc71', # Example color
            linewidth=1.5,   # Adjusted linewidth
            ax=ax
        )
        ax.set_title(f"{selected_var} Over Time", fontsize=14)
        ax.set_ylabel(selected_var)
        ax.set_xlabel("Date")
        ax.grid(True, linestyle='--', alpha=0.6) # Customize grid
        sns.despine() # Remove top and right spines
        plt.tight_layout() # Adjust layout to prevent overlap
        st.pyplot(fig)
    else:
        st.info(f"No valid data points to plot for '{selected_var}' after removing missing values in the selected range.")
else:
     # Provide info if variable exists but has no data in range or is non-numeric
     if selected_var in df.columns:
          st.info(f"No data to display for '{selected_var}' in the selected time period or variable is non-numeric.")
     # else: error handled above


# Map Visualization
st.subheader("🗺️ Spatial Distribution & Intensity")

# Initialize map only ONCE if it's not in session state and no previous error occurred
if st.session_state.map is None and not st.session_state.map_init_error:
    if geojson: # Proceed only if geojson loaded successfully
        try:
            # Create GeoDataFrame from GeoJSON features
            gdf_map = gpd.GeoDataFrame.from_features(geojson['features'])

            # Validate and fix geometries if necessary
            if not gdf_map.geometry.is_valid.all():
                 gdf_map.geometry = gdf_map.geometry.buffer(0)
                 if not gdf_map.geometry.is_valid.all():
                      st.warning("Could not fix invalid geometries in GeoJSON. Map centroid might be inaccurate.")

            # Calculate combined geometry and centroid
            combined_geometry = gdf_map.geometry.union_all()

            if gdf_map.empty or combined_geometry.is_empty:
                 st.error("GeoJSON data resulted in empty geometry. Cannot display map.")
                 st.session_state.map_init_error = True # Set error flag
            else:
                 map_centroid = combined_geometry.centroid
                 start_location = [map_centroid.y, map_centroid.x]

                 # Create the Folium map object
                 m = folium.Map(
                     location=start_location,
                     zoom_start=10,
                     tiles="cartodbpositron", # Clean base map
                     control_scale=True # Show scale bar
                 )

                 # Add GeoJSON layer for the protected area boundary
                 folium.GeoJson(
                     geojson,
                     name="Sharaan Protected Area",
                     style_function=lambda x: {
                         'fillColor': '#9b59b6', # Example fill color
                         'color': '#8e44ad',     # Example border color
                         'weight': 2,
                         'fillOpacity': 0.2
                     }
                     # Tooltip removed earlier, keep it removed for now
                 ).add_to(m)

                 # Add HeatMap layer if points were generated
                 if heatmap_points:
                     HeatMap(
                         data=[[p['coordinates'][0], p['coordinates'][1], p['intensity']] for p in heatmap_points],
                         radius=20,
                         blur=15,
                         gradient=HEATMAP_GRADIENT,
                         name="Simulated Intensity Heatmap"
                     ).add_to(m)

                 # Add Layer Control
                 folium.LayerControl().add_to(m)

                 # Store the successfully created map in session state
                 st.session_state.map = m

        except Exception as e:
            st.error(f"Failed to create map: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}") # Log detailed error
            st.session_state.map_init_error = True # Set error flag
            st.session_state.map = None # Ensure map is None if creation failed
    else:
        st.error("GeoJSON data not available, cannot create map.")
        st.session_state.map_init_error = True # Set error flag


# Render map using streamlit-folium if it exists in session state and no init error
if isinstance(st.session_state.get('map'), folium.Map): # Check if it's a valid map object
    try:
        # *** REMOVED key ARGUMENT HERE ***
        st_folium(
            st.session_state.map,
            width='100%', # Use percentage for responsiveness
            height=500,
            returned_objects=[] # No objects needed back from map currently
        )
    except Exception as e:
         # Catch potential rendering errors
         st.error(f"Error rendering map with streamlit-folium: {e}")
         import traceback
         st.error(f"Traceback: {traceback.format_exc()}")
         # Optionally set map state to error here too if rendering fails persistently
         # st.session_state.map_init_error = True
         # st.session_state.map = None

# Display message if map creation failed or is pending
elif st.session_state.get('map_init_error'):
     st.error("Map could not be displayed due to an earlier error during creation.")
else:
     # This case might happen if geojson was None initially or still loading
     st.info("Map is initializing or required GeoJSON data is missing.")


# --- Footer ---
st.markdown("---")
last_data_point_str = "N/A"
# Safely access max date
if not df.empty and 'Date' in df.columns:
     last_data_point = df['Date'].max()
     # Check if max date is NaT (Not a Time) which can happen
     if pd.notna(last_data_point):
          last_data_point_str = last_data_point.strftime('%Y-%m-%d')

st.caption(f"Data source: Simulated climate data | GeoJSON source: Provided URL | Last data point: {last_data_point_str}")
