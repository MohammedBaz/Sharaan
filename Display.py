import streamlit as st
import pandas as pd
import numpy as np
# Folium is no longer needed for this approach
# import folium
# from folium.plugins import HeatMap
# import streamlit.components.v1 as components # No longer needed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # For color mapping
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
# MinMaxScaler is no longer needed if we map intensity directly to a colormap
# from sklearn.preprocessing import MinMaxScaler
import random
import requests
import warnings

# Configuration
DATA_URL = "https://raw.githubusercontent.com/MohammedBaz/Sharaan/main/dataset.csv"
GEOJSON_URL = "https://raw.githubusercontent.com/MohammedBaz/Sharaan/main/Sharaan.geojson"
# Use a Matplotlib colormap for intensity (e.g., 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm', 'RdYlGn')
COLORMAP_NAME = 'viridis'

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
        numeric_cols_to_check = [col for col in df.columns if col != 'Date']
        for col in numeric_cols_to_check:
             # Only attempt conversion if not already numeric to avoid warnings
             if not pd.api.types.is_numeric_dtype(df[col]):
                  df[col] = pd.to_numeric(df[col], errors='coerce')
             # Handle potential NaNs introduced by coercion if necessary, e.g., fillna(0) or dropna()
        # Drop columns that are entirely NaN after coercion attempt
        df.dropna(axis=1, how='all', inplace=True)
        return df.sort_values('Date')
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        st.stop()

@st.cache_data
def load_geojson_gdf():
    """Load protected area boundaries into a GeoDataFrame"""
    try:
        response = requests.get(GEOJSON_URL)
        response.raise_for_status()
        geojson_data = response.json()
        gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])
        # Ensure valid geometries
        if not gdf.geometry.is_valid.all():
            gdf.geometry = gdf.geometry.buffer(0)
        if gdf.empty or gdf.geometry.union_all().is_empty:
             st.error("GeoJSON data resulted in empty geometry.")
             return None
        return gdf
    except Exception as e:
        st.error(f"Geojson loading/processing failed: {str(e)}")
        st.stop()

# --- Helper Function ---

def normalize_value(value, min_val, max_val):
    """Normalize a value between 0 and 1."""
    if pd.isna(value) or pd.isna(min_val) or pd.isna(max_val):
        return 0.5 # Default for missing data
    if max_val == min_val: # Avoid division by zero
        return 0.5 # Neutral value if range is zero
    # Clip value to be within min/max to handle potential outliers if needed
    value = np.clip(value, min_val, max_val)
    return (value - min_val) / (max_val - min_val)

# --- App Initialization ---
st.set_page_config(layout="wide")

# --- Data Loading ---
df = load_data()
gdf_map = load_geojson_gdf() # Load directly into GeoDataFrame

if df is None or gdf_map is None:
     st.error("Failed to load necessary data. Dashboard cannot proceed.")
     st.stop()

# --- App Layout ---
st.title("🌦️ Sharaan Protected Area Climate Dashboard")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    # Get numeric columns from the loaded dataframe
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
    st.warning(f"No valid data available for '{selected_var}' in the selected date range.")


# --- Layout for Plot and Map ---
col_plot, col_map = st.columns(2)

with col_plot:
    st.subheader("📈 Temporal Trends")
    if not filtered_var_df.empty:
        plot_df_temporal = filtered_df[['Date', selected_var]].dropna(subset=[selected_var])
        if not plot_df_temporal.empty:
            fig_ts, ax_ts = plt.subplots(figsize=(10, 4))
            sns.lineplot(data=plot_df_temporal, x='Date', y=selected_var, color='#2ecc71', linewidth=1.5, ax=ax_ts)
            ax_ts.set_title(f"{selected_var} Over Time", fontsize=12)
            ax_ts.set_ylabel(selected_var)
            ax_ts.set_xlabel("Date")
            ax_ts.grid(True, linestyle='--', alpha=0.6)
            sns.despine()
            plt.tight_layout()
            st.pyplot(fig_ts) # Pass the figure object
        else:
             st.info(f"No valid data points to plot for '{selected_var}'.")
    else:
         st.info(f"No data to display for '{selected_var}' in the selected time period.")

with col_map:
    st.subheader("🗺️ Area Climate Intensity")
    st.markdown(f"Color represents the **normalized average {selected_var}** for the period (Colormap: {COLORMAP_NAME}).")

    # --- GeoPandas Plot Generation ---
    if not gdf_map.empty:
        try:
            # Calculate average value for the selected variable in the filtered period
            avg_value = filtered_var_df[selected_var].mean()

            # Get overall min/max for the selected variable from the *entire* dataset for consistent normalization
            overall_min = df[selected_var].min()
            overall_max = df[selected_var].max()

            # Normalize the average value (0 to 1)
            normalized_intensity = normalize_value(avg_value, overall_min, overall_max)

            # Get the colormap
            cmap = plt.get_cmap(COLORMAP_NAME)
            # Map the normalized intensity to a color
            fill_color = cmap(normalized_intensity) if pd.notna(normalized_intensity) else 'lightgrey' # Grey if no data

            # Create the plot
            fig_map, ax_map = plt.subplots(1, 1, figsize=(8, 8)) # Adjust figsize as needed

            # Plot the GeoDataFrame
            gdf_map.plot(ax=ax_map, facecolor=fill_color, edgecolor='black', linewidth=0.5)

            # Customize appearance
            ax_map.set_xticks([]) # Remove x-axis ticks
            ax_map.set_yticks([]) # Remove y-axis ticks
            ax_map.set_xlabel('') # Remove x-axis label
            ax_map.set_ylabel('') # Remove y-axis label
            ax_map.set_title(f'Avg {selected_var} Intensity', fontsize=10)
            # Remove the frame/spines
            for spine in ax_map.spines.values():
                spine.set_visible(False)
            # Ensure equal aspect ratio
            ax_map.set_aspect('equal', adjustable='box')
            plt.tight_layout()

            # Display the plot in Streamlit
            st.pyplot(fig_map) # Pass the figure object

        except Exception as e:
            st.error(f"Failed to create map plot: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
    else:
         st.info("Map plot could not be generated. Check GeoJSON data.")


# --- Footer ---
st.markdown("---")
last_data_point_str = "N/A"
if not df.empty and 'Date' in df.columns:
     last_data_point = df['Date'].max()
     if pd.notna(last_data_point):
          last_data_point_str = last_data_point.strftime('%Y-%m-%d')

st.caption(f"Data source: Simulated climate data | GeoJSON source: Provided URL | Last data point: {last_data_point_str}")
