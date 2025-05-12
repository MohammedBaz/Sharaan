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
# 'coolwarm' might be good for variation (blue=low, red=high)
COLORMAP_NAME = 'coolwarm'

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
        all_numeric_cols = {}
        for col in numeric_cols_to_check:
             # Only attempt conversion if not already numeric to avoid warnings
             if not pd.api.types.is_numeric_dtype(df[col]):
                  # Store original dtype before potential coercion
                  original_dtype = df[col].dtype
                  df[col] = pd.to_numeric(df[col], errors='coerce')
                  # If coercion resulted in all NaNs, maybe revert or warn
                  if df[col].isnull().all() and not pd.api.types.is_numeric_dtype(original_dtype):
                       st.warning(f"Column '{col}' could not be converted to numeric and contains non-numeric data.")
                       # Option: Keep as object or drop? For now, keep but it won't be selectable.
             # Check if column is numeric after potential conversion
             if pd.api.types.is_numeric_dtype(df[col]):
                  all_numeric_cols[col] = df[col] # Add valid numeric columns

        # Recreate DataFrame with only valid columns (Date + numeric)
        valid_cols_df = pd.DataFrame(all_numeric_cols)
        valid_cols_df['Date'] = df['Date']
        # Drop rows where Date is NaT as they are unusable
        valid_cols_df.dropna(subset=['Date'], inplace=True)
        # Drop original df columns that are entirely NaN after coercion attempt (if any were kept)
        valid_cols_df.dropna(axis=1, how='all', inplace=True)

        return valid_cols_df.sort_values('Date')
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
        # Set Coordinate Reference System (CRS) if known, e.g., WGS84
        # gdf.set_crs("EPSG:4326", inplace=True) # Uncomment if CRS is known and needed
        return gdf
    except Exception as e:
        st.error(f"Geojson loading/processing failed: {str(e)}")
        st.stop()

# --- Helper Function ---

def normalize_variation(std_dev, overall_min, overall_max):
    """Normalize standard deviation relative to the overall data range."""
    if pd.isna(std_dev):
        return 0.0 # No variation if std dev is NaN (e.g., single point)
    overall_range = overall_max - overall_min
    if pd.isna(overall_range) or overall_range == 0:
        return 0.0 # No variation if overall range is zero or NaN
    # Normalize std dev by the overall range and clip between 0 and 1
    normalized = std_dev / overall_range
    return np.clip(normalized, 0.0, 1.0)

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
    # Get numeric columns from the *cleaned* dataframe
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.error("No valid numeric climate parameters found after cleaning.")
        st.stop()

    try:
        default_index = numeric_cols.index('Rainfall')
    except ValueError:
        default_index = 0 # Default to first valid numeric column

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
    # Get the data for the selected variable, drop NaNs for calculations
    filtered_var_data = filtered_df[selected_var].dropna()
else:
    st.warning("Please select a valid date range. Displaying all data.")
    filtered_df = df.copy()
    filtered_var_data = filtered_df[selected_var].dropna()


# --- Main Panel Display ---

# Key Metrics (calculated on filtered_var_data which has NaNs dropped)
st.subheader(f"📊 Key Metrics for {selected_var}")
if not filtered_var_data.empty:
    col1, col2, col3 = st.columns(3)
    mean_val = filtered_var_data.mean()
    max_val = filtered_var_data.max()
    min_val = filtered_var_data.min()
    std_dev_val = filtered_var_data.std() # Calculate std dev for display
    with col1:
        st.metric("Average", f"{mean_val:.1f}" if pd.notna(mean_val) else "N/A")
    with col2:
        st.metric("Maximum", f"{max_val:.1f}" if pd.notna(max_val) else "N/A")
    with col3:
        # Display Standard Deviation as a measure of variation
        st.metric("Std Deviation", f"{std_dev_val:.2f}" if pd.notna(std_dev_val) else "N/A")
else:
    st.warning(f"No valid data available for '{selected_var}' in the selected date range.")


# --- Layout for Plot and Map ---
col_plot, col_map = st.columns(2)

with col_plot:
    st.subheader("📈 Temporal Trends")
    # Use original filtered_df for plotting to keep date axis correct
    plot_df_temporal = filtered_df[['Date', selected_var]].dropna(subset=[selected_var])
    if not plot_df_temporal.empty:
        fig_ts, ax_ts = plt.subplots(figsize=(10, 6)) # Keep adjusted height
        sns.lineplot(data=plot_df_temporal, x='Date', y=selected_var, color='#2ecc71', linewidth=1.5, ax=ax_ts)
        ax_ts.set_title(f"{selected_var} Over Time", fontsize=12)
        ax_ts.set_ylabel(selected_var)
        ax_ts.set_xlabel("Date")
        ax_ts.grid(True, linestyle='--', alpha=0.6)
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig_ts)
    else:
         st.info(f"No valid data points to plot for '{selected_var}'.")


with col_map:
    # *** Updated Subheader and Markdown Text ***
    st.subheader("🗺️ Area Climate Variation")
    st.markdown(f"Color represents the **normalized standard deviation** of **{selected_var}** for the period (Colormap: {COLORMAP_NAME}). Red indicates higher variation, Blue indicates lower variation.")

    # --- GeoPandas Plot Generation ---
    if not gdf_map.empty:
        try:
            # *** Calculate Standard Deviation for the period ***
            # Use the pre-filtered filtered_var_data
            period_std_dev = filtered_var_data.std()

            # Get overall min/max for the selected variable from the *entire* dataset
            # Ensure the selected variable still exists after cleaning
            if selected_var in df.columns:
                overall_min = df[selected_var].min()
                overall_max = df[selected_var].max()
            else:
                # Handle case where selected variable might have been dropped if all NaN
                overall_min, overall_max = np.nan, np.nan
                st.warning(f"Could not determine overall range for '{selected_var}'.")


            # *** Normalize the standard deviation ***
            normalized_variation = normalize_variation(period_std_dev, overall_min, overall_max)

            # Get the colormap
            cmap = plt.get_cmap(COLORMAP_NAME)
            # Map the normalized variation to a color
            fill_color = cmap(normalized_variation) if pd.notna(normalized_variation) else 'lightgrey' # Grey if no data/variation

            # Create the plot
            fig_map, ax_map = plt.subplots(1, 1, figsize=(8, 8))

            # Plot the GeoDataFrame
            gdf_map.plot(ax=ax_map, facecolor=fill_color, edgecolor='black', linewidth=0.5)

            # Customize appearance
            ax_map.set_xticks([])
            ax_map.set_yticks([])
            ax_map.set_xlabel('')
            ax_map.set_ylabel('')
            # *** Updated Map Title ***
            ax_map.set_title(f'Std Dev of {selected_var}', fontsize=10)
            for spine in ax_map.spines.values():
                spine.set_visible(False)
            ax_map.set_aspect('equal', adjustable='box')
            plt.tight_layout()

            # Display the plot in Streamlit
            st.pyplot(fig_map)

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
