import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import requests
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# --- App Setup ---
# **FIX:** Moved st.set_page_config to be the first Streamlit command after imports.
st.set_page_config(layout="wide", page_title="EcoMonitor", page_icon="🌿")

# --- Configuration ---
# URL for the dataset CSV file
DATA_URL = "https://raw.githubusercontent.com/MohammedBaz/Sharaan/main/dataset.csv"
# URL for the GeoJSON file defining the area boundaries
GEOJSON_URL = "https://raw.githubusercontent.com/MohammedBaz/Sharaan/main/Sharaan.geojson"
# Path to the video file (ensure this file exists in the same directory or provide a correct path)
VIDEO_PATH = "GreenCover.mp4"
# Configuration settings for the video player
VIDEO_CONFIG = {"autoplay": False, "muted": True, "loop": False}

# --- Custom CSS Styling ---
# Apply custom styles to the Streamlit app elements for better aesthetics
# This st.markdown call now correctly comes *after* st.set_page_config
st.markdown("""
    <style>
        /* Style the main background */
        .main {background-color: #f8f9fa;}
        /* Style the video player */
        .stVideo {border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
        /* Style containers (used for metrics) */
        .css-1aumxhk, .metric-container {background-color: #ffffff; border-radius: 10px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
        /* Style the main title */
        h1 {color: #2c3e50;}
        /* Specific styling for metric containers */
        .metric-container {padding: 15px; margin-bottom: 10px;}
        .metric-container h2 {margin-top: 5px;} /* Adjust spacing for metric value */
    </style>
""", unsafe_allow_html=True)

# --- Data Loading Functions ---
@st.cache_data # Cache the data to avoid reloading on every interaction
def load_data():
    """Loads, cleans, and preprocesses the dataset from the DATA_URL."""
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(DATA_URL)
        # Convert the 'Date' column to datetime objects, coercing errors to NaT (Not a Time)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # Replace spaces in column names with underscores for easier access
        df.columns = df.columns.str.replace(' ', '_')

        # Check if any dates failed to parse
        if df['Date'].isnull().any():
            # **CORRECTION:** Fixed the error message to be more informative and avoid undefined variable.
            st.error("Error: Some date values could not be parsed. Please check the 'Date' column format in the CSV.")
            st.stop() # Stop the app execution if dates are invalid

        # Identify numeric columns (excluding 'Date')
        numeric_cols = [col for col in df.columns if col != 'Date']
        # Convert numeric columns to numeric types, coercing errors to NaN (Not a Number)
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Sort the DataFrame by date and drop columns that are entirely empty
        return df.sort_values('Date').dropna(how='all', axis=1)
    except Exception as e:
        # Display an error message if data loading fails
        st.error(f"Fatal Error: Data loading failed. Cannot start the application. Details: {str(e)}")
        st.stop() # Stop the app execution

@st.cache_data # Cache the GeoJSON data
def load_geojson():
    """Loads the GeoJSON data from the GEOJSON_URL."""
    try:
        # Fetch the GeoJSON data from the URL
        response = requests.get(GEOJSON_URL)
        response.raise_for_status() # Raise an exception for bad status codes (like 404)
        geojson = response.json() # Parse the JSON response
        # Create a GeoDataFrame from the GeoJSON features
        gdf = gpd.GeoDataFrame.from_features(geojson['features'])
        # Ensure geometries are valid; buffer by 0 if not (common fix for invalid polygons)
        gdf.geometry = gdf.geometry.buffer(0) if not gdf.geometry.is_valid.all() else gdf.geometry
        # Return the raw GeoJSON and the combined geometry of all features
        return geojson, gdf.geometry.unary_union
    except Exception as e:
        # Display an error message if GeoJSON loading fails
        st.error(f"Error: GeoJSON loading failed. Map features might be unavailable. Details: {str(e)}")
        return None, None # Return None if loading fails

# --- Helper Functions ---
def get_parameter_groups(df):
    """Identifies parameter groups (Max, Min, Mean) based on column naming conventions."""
    groups = {}
    for col in df.columns:
        if col == 'Date': continue # Skip the 'Date' column
        if '_' in col:
            parts = col.split('_')
            # Expecting format like "Stat_ParameterName" (e.g., "Max_Temperature")
            if len(parts) >= 2:
                prefix = parts[0] # Should be 'Max', 'Min', or 'Mean'
                parameter = '_'.join(parts[1:]) # The rest is the parameter name
                # Ensure the prefix is one of the expected types
                if prefix in ['Max', 'Min', 'Mean']:
                    # Initialize the group if it doesn't exist
                    if parameter not in groups:
                         groups[parameter] = {}
                    # Store the full column name under the corresponding key ('Max', 'Min', 'Mean')
                    groups[parameter][prefix] = col
    # Filter out any groups that don't have all three (Max, Min, Mean) columns
    return {param: data for param, data in groups.items() if all(k in data for k in ['Max', 'Min', 'Mean'])}


def normalize_value(value, overall_min, overall_max):
    """Normalizes a value to a 0-1 range based on overall min/max."""
    # Handle potential NaN or invalid min/max values by returning a default mid-value (0.5)
    if pd.isna(value) or pd.isna(overall_min) or pd.isna(overall_max) or overall_max == overall_min:
        return 0.5 # Default to middle color if data is missing or range is zero
    # Clip the value to be within the min/max bounds, then normalize
    normalized = (np.clip(value, overall_min, overall_max) - overall_min) / (overall_max - overall_min)
    return normalized

# --- Statistical Functions ---
def run_ttest(data, variable, group_var):
    """Performs an independent two-sample t-test."""
    groups = data[group_var].dropna().unique()
    # Check if exactly two groups are present
    if len(groups) != 2:
        return None, f"T-Test requires exactly two groups. Found {len(groups)} for '{group_var}'."
    # Prepare data for each group
    group_data = [data[data[group_var] == grp][variable].dropna() for grp in groups]
    # Check if each group has sufficient data points
    if any(len(d) < 3 for d in group_data):
        return None, f"T-Test requires at least 3 valid samples per group for '{variable}'. Check for missing data."
    # Perform the t-test
    try:
        t_stat, p_value = stats.ttest_ind(*group_data, nan_policy='omit') # Omit NaNs within groups
        return (t_stat, p_value), None
    except Exception as e:
        return None, f"T-Test failed: {str(e)}"

def run_anova(data, variable, group_var):
    """Performs a one-way ANOVA test."""
    # Check if at least two groups are present
    if data[group_var].nunique() < 2:
        return None, f"ANOVA requires at least two groups for '{group_var}'."
    try:
        # Fit the Ordinary Least Squares (OLS) model
        # Use backticks to handle potential special characters in column names
        model = ols(f'`{variable}` ~ C(`{group_var}`)', data=data.dropna(subset=[variable, group_var])).fit()
        # Perform ANOVA on the fitted model
        anova_table = sm.stats.anova_lm(model, typ=2)
        return anova_table, None
    except Exception as e:
        return None, f"ANOVA failed for '{variable}' by '{group_var}': {str(e)}"

def run_regression(data, x_var, y_var):
    """Performs a simple linear regression."""
    try:
        # Prepare data by dropping rows with missing values in either variable
        data_clean = data[[x_var, y_var]].dropna()
        # Check for sufficient data points
        if len(data_clean) < 10:
            return None, f"Regression requires at least 10 non-missing data points for '{x_var}' and '{y_var}'. Found {len(data_clean)}."
        # Check if the independent variable has variance
        if data_clean[x_var].nunique() == 1:
            return None, f"Independent variable '{x_var}' has no variation (all values are the same)."
        # Add a constant (intercept) to the independent variable
        X = sm.add_constant(data_clean[x_var])
        # Fit the OLS regression model
        model = sm.OLS(data_clean[y_var], X).fit()
        return model, None
    except Exception as e:
        return None, f"Regression failed for '{y_var}' vs '{x_var}': {str(e)}"


# --- Load Data ---
# Load data and GeoJSON - handle potential errors during loading
df = load_data()
geojson, sharaan_boundary = load_geojson() # Unpack boundary as well

# Get parameter groups after data is loaded
param_groups = get_parameter_groups(df)
if not param_groups:
    st.error("Error: Could not identify valid parameter groups (Max, Min, Mean) from column names. Please check CSV format.")
    st.stop()

# --- Main App Structure ---
# Define the tabs for the application
tabs = ["🌿 Green Cover", "📊 Dashboard", "🔗 Correlation", "📈 Temporal", "📉 Statistics"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(tabs)

# --- Tab1: Green Cover ---
with tab1:
    st.title("🌳 Sharaan Vegetation Dynamics Monitor")
    st.markdown("Visualizing changes in vegetation cover over time within the Sharaan Nature Reserve.")
    # Center the video using columns
    col1, col2, col3 = st.columns([1, 8, 1]) # Adjust ratios as needed
    with col2:
        try:
            # Attempt to open and display the video file
            with open(VIDEO_PATH, 'rb') as video_file:
                video_bytes = video_file.read()
            st.video(video_bytes, format="video/mp4", start_time=0, **VIDEO_CONFIG)
            st.caption("This animation illustrates the fluctuations in vegetation indices derived from satellite imagery.")
        except FileNotFoundError:
            st.error(f"Error: Video file not found at '{VIDEO_PATH}'. Please ensure the video file is in the correct location.")
        except Exception as e:
            st.error(f"Error loading video: {str(e)}")

# --- Tab2: Climate Dashboard ---
with tab2:
    st.title("📊 Climate & Environmental Dashboard")
    st.markdown("Explore trends and spatial patterns of key environmental parameters.")

    # Sidebar for user selections
    with st.sidebar:
        st.header("Dashboard Controls")
        # Ensure param_groups is not empty before creating selectbox
        if param_groups:
            groups_list = sorted(param_groups.keys())
            # Dropdown to select the parameter group (e.g., Temperature, Humidity)
            selected_group_key = st.selectbox(
                "Select Parameter Group",
                groups_list,
                key="dashboard_group_select",
                index=0 # Default to the first group
            )
        else:
            st.warning("No parameter groups available.")
            selected_group_key = None # Set to None if no groups

        # Date range input
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        selected_date_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="dashboard_date_range"
        )

    # Filter data based on sidebar selections
    if selected_group_key and len(selected_date_range) == 2:
        group_cols_info = param_groups[selected_group_key]
        start_date, end_date = pd.to_datetime(selected_date_range[0]), pd.to_datetime(selected_date_range[1])

        # Ensure start_date is not after end_date
        if start_date > end_date:
            st.error("Error: Start date cannot be after end date.")
            filtered_df = pd.DataFrame() # Empty DataFrame
        else:
             filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy() # Use .copy() to avoid SettingWithCopyWarning

        # Display metrics if data is available
        if not filtered_df.empty:
            st.subheader(f"Summary for {selected_group_key.replace('_', ' ').title()} ({selected_date_range[0]} to {selected_date_range[1]})")
            cols = st.columns(3)
            metrics_data = {
                'MAX': filtered_df[group_cols_info['Max']].max(),
                'MIN': filtered_df[group_cols_info['Min']].min(),
                'AVG': filtered_df[group_cols_info['Mean']].mean()
            }
            metric_labels = {'MAX': 'Maximum', 'MIN': 'Minimum', 'AVG': 'Average'}

            for i, (label, value) in enumerate(metrics_data.items()):
                with cols[i]:
                    # Use custom HTML for styled metric display
                    st.markdown(f"""
                        <div class="metric-container">
                            <span style="font-size: 0.9em; color: #555;">{metric_labels[label]}</span>
                            <h2>{value:.2f}</h2>
                        </div>
                    """, unsafe_allow_html=True)

            # Visualizations
            st.subheader("Visualizations")
            col1, col2 = st.columns([2, 1]) # Allocate more space for the line chart

            # Line chart for trends over time
            with col1:
                fig_line, ax_line = plt.subplots(figsize=(10, 4))
                for prefix in ['Max', 'Mean', 'Min']:
                    col_name = group_cols_info[prefix]
                    sns.lineplot(data=filtered_df, x='Date', y=col_name, label=prefix, ax=ax_line, marker='o', markersize=3, linestyle='-')
                ax_line.set_title(f"{selected_group_key.replace('_', ' ').title()} Trend")
                ax_line.set_ylabel(selected_group_key.replace('_', ' '))
                ax_line.set_xlabel("Date")
                ax_line.legend(title="Statistic")
                ax_line.grid(True, linestyle='--', alpha=0.6)
                plt.xticks(rotation=45)
                st.pyplot(fig_line)

            # Geographical map showing average intensity
            with col2:
                st.markdown("**Average Intensity Map**")
                if geojson and sharaan_boundary: # Check if GeoJSON was loaded successfully
                    try:
                        # Calculate the average of the 'Mean' column for the selected period
                        current_mean_value = filtered_df[group_cols_info['Mean']].mean()
                        # Get overall min/max for normalization from the original DataFrame
                        overall_min = df[group_cols_info['Mean']].min()
                        overall_max = df[group_cols_info['Mean']].max()
                        # Normalize the current average value
                        normalized_mean = normalize_value(current_mean_value, overall_min, overall_max)

                        # Create the map plot
                        fig_map, ax_map = plt.subplots(1, 1, figsize=(6, 6))
                        # Create a GeoDataFrame for plotting
                        map_gdf = gpd.GeoDataFrame([1], geometry=[sharaan_boundary], crs="EPSG:4326") # Assuming WGS84
                        # Plot the GeoDataFrame with color based on normalized value
                        map_gdf.plot(
                            ax=ax_map,
                            facecolor=plt.get_cmap('viridis')(normalized_mean), # Use viridis colormap
                            edgecolor='black',
                            linewidth=0.5
                        )
                        ax_map.set_axis_off() # Hide axes
                        ax_map.set_title(f"Avg. {selected_group_key.replace('_', ' ')}\nValue: {current_mean_value:.2f}", fontsize=10)
                        st.pyplot(fig_map)
                        st.caption(f"Color intensity represents the normalized average value ({current_mean_value:.2f}) compared to the overall historical range ({overall_min:.2f} - {overall_max:.2f}).")
                    except Exception as e:
                        st.error(f"Map generation failed: {str(e)}")
                else:
                    st.warning("GeoJSON data not available, cannot display map.")

        else:
            st.warning("No data available for the selected date range.")
    else:
         st.warning("Please select a parameter group and a valid date range in the sidebar.")


# --- Tab3: Correlation Analysis ---
with tab3:
    st.title("🔗 Cross-Parameter Correlation Analysis")
    st.markdown("Explore relationships between different environmental parameters (Max, Min, Mean values).")

    # Exclude the group currently selected in the dashboard for clarity
    excluded_group = st.session_state.get("dashboard_group_select", None) # Get selection from dashboard
    available_groups = sorted([p for p in param_groups.keys() if p != excluded_group])

    if len(available_groups) >= 2:
        # Multiselect for choosing parameters to correlate
        selected_corr_groups = st.multiselect(
            "Select parameters to correlate (select at least 2)",
            available_groups,
            default=available_groups[:min(len(available_groups), 3)], # Default to first 2 or 3
            key="correlation_params_select"
        )

        if len(selected_corr_groups) >= 2:
            # Collect the relevant column names (Max, Min, Mean for each selected group)
            corr_vars_cols = []
            corr_labels = []
            for p_group in selected_corr_groups:
                for stat_type in ['Max', 'Min', 'Mean']:
                    # Check if the specific stat type exists for the group (it should based on get_parameter_groups logic)
                    if stat_type in param_groups[p_group]:
                         col_name = param_groups[p_group][stat_type]
                         corr_vars_cols.append(col_name)
                         # Create shorter labels for the heatmap axes
                         corr_labels.append(f"{p_group[:10]}\n({stat_type})") # Abbreviate group name if long
                    else:
                        st.warning(f"Missing '{stat_type}' column for parameter group '{p_group}'. Skipping.")


            if corr_vars_cols: # Proceed only if we have columns to correlate
                # Calculate the correlation matrix
                correlation_matrix = df[corr_vars_cols].corr()

                # Plot the heatmap
                fig_corr, ax_corr = plt.subplots(figsize=(max(8, len(corr_vars_cols)*0.8), max(6, len(corr_vars_cols)*0.7))) # Adjust size dynamically
                sns.heatmap(
                    correlation_matrix,
                    annot=True,       # Show correlation values
                    fmt=".2f",        # Format values to 2 decimal places
                    cmap="coolwarm",  # Color scheme (cool=negative, warm=positive)
                    linewidths=.5,    # Add lines between cells
                    linecolor='lightgray',
                    ax=ax_corr,
                    xticklabels=corr_labels, # Use generated labels
                    yticklabels=corr_labels
                )
                ax_corr.set_title("Correlation Matrix Heatmap", fontsize=14)
                plt.xticks(rotation=45, ha='right', fontsize=9) # Rotate x-axis labels for readability
                plt.yticks(rotation=0, fontsize=9)
                plt.tight_layout() # Adjust layout to prevent labels overlapping
                st.pyplot(fig_corr)
            else:
                st.warning("No valid columns found for the selected parameters to generate a correlation matrix.")
        else:
            st.info("Please select at least 2 parameter groups to calculate correlations.")
    else:
        st.warning("Not enough parameter groups available (minimum 2 required) to perform correlation analysis.")


# --- Tab4: Temporal Analysis ---
with tab4:
    st.title("📈 Temporal Analysis with Rolling Averages")
    st.markdown("Smooth out short-term fluctuations to observe longer-term trends using rolling averages.")

    if param_groups:
        # Selectbox for parameter group
        temporal_group_key = st.selectbox(
            "Select Parameter Group",
            sorted(param_groups.keys()),
            key="temporal_group_select"
        )
        # Slider for rolling window size
        rolling_window_days = st.slider(
            "Select Rolling Window Size (days)",
            min_value=1,
            max_value=90, # Adjust max window size if needed
            value=7,      # Default to 7 days
            step=1,
            key="temporal_window_slider"
        )

        if temporal_group_key:
            group_cols_info = param_groups[temporal_group_key]
            # Prepare data, setting Date as index
            ts_data = df.set_index('Date')[[
                group_cols_info['Max'],
                group_cols_info['Mean'],
                group_cols_info['Min']
            ]].copy() # Select the relevant columns

            # Calculate rolling mean
            ts_rolling_avg = ts_data.rolling(window=f'{rolling_window_days}D').mean() # Use 'D' for days

            # Plotting
            fig_temporal, ax_temporal = plt.subplots(figsize=(12, 5))

            # Plot rolling Max, Mean, Min lines
            for col in ts_rolling_avg.columns:
                prefix = col.split('_')[0] # Get 'Max', 'Mean', or 'Min'
                sns.lineplot(x=ts_rolling_avg.index, y=ts_rolling_avg[col], label=f'{prefix} ({rolling_window_days}-day avg)', ax=ax_temporal)

            # Fill between rolling Min and Max
            min_col = group_cols_info['Min']
            max_col = group_cols_info['Max']
            if min_col in ts_rolling_avg.columns and max_col in ts_rolling_avg.columns:
                 ax_temporal.fill_between(
                     ts_rolling_avg.index,
                     ts_rolling_avg[min_col], # Rolling Min
                     ts_rolling_avg[max_col], # Rolling Max
                     alpha=0.15,              # Transparency
                     color='gray',
                     label='Min-Max Range (Rolling Avg)'
                 )

            ax_temporal.set_title(f"{temporal_group_key.replace('_', ' ').title()} - {rolling_window_days}-Day Rolling Statistics")
            ax_temporal.set_ylabel(temporal_group_key.replace('_', ' '))
            ax_temporal.set_xlabel("Date")
            ax_temporal.legend(loc='upper left', bbox_to_anchor=(1, 1)) # Place legend outside plot
            ax_temporal.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig_temporal)
        else:
            st.warning("Please select a parameter group.")
    else:
        st.warning("No parameter groups available for temporal analysis.")


# --- Tab5: Statistics ---
with tab5:
    st.title("📉 Statistical Hypothesis Testing")
    st.markdown("Perform basic statistical tests to compare groups or analyze relationships.")

    # Select the type of statistical test
    test_type = st.selectbox(
        "Select Analysis Type",
        ["T-Test (Compare 2 Groups)", "ANOVA (Compare 2+ Groups)", "Linear Regression (Relationship)"],
        key="stats_test_type_select"
    )

    st.markdown("---") # Separator

    # --- T-Test Section ---
    if "T-Test" in test_type:
        st.subheader("Independent Samples T-Test")
        st.markdown("Compares the means of a variable between two distinct groups.")
        col1, col2 = st.columns(2)
        with col1:
            # Select the numeric variable to compare
            t_test_variable = st.selectbox(
                "Select Variable (Numeric)",
                df.select_dtypes(include=np.number).columns,
                key="ttest_variable_select"
            )
        with col2:
            # Find columns suitable for grouping (exactly 2 unique non-null values)
            potential_group_vars = [c for c in df.columns if df[c].dropna().nunique() == 2]
            if potential_group_vars:
                 t_test_group_var = st.selectbox(
                     "Select Grouping Variable (Must have exactly 2 groups)",
                     potential_group_vars,
                     key="ttest_group_select"
                 )
            else:
                 st.warning("No suitable grouping variables found (need columns with exactly 2 unique values).")
                 t_test_group_var = None

        # Button to run the test
        if t_test_group_var and st.button("Run T-Test", key="ttest_run_button"):
            result, error_msg = run_ttest(df, t_test_variable, t_test_group_var)
            if error_msg:
                st.error(f"T-Test Error: {error_msg}")
            elif result:
                t_stat, p_value = result
                st.metric("T-Statistic", f"{t_stat:.3f}")
                st.metric("P-Value", f"{p_value:.4f}")
                alpha = 0.05 # Significance level
                if p_value < alpha:
                    st.success(f"Result is statistically significant (p < {alpha}). There is a significant difference in '{t_test_variable}' between the two groups of '{t_test_group_var}'.")
                else:
                    st.info(f"Result is not statistically significant (p >= {alpha}). There is no significant difference in '{t_test_variable}' between the two groups of '{t_test_group_var}'.")
            else:
                st.error("T-Test failed to produce results.")

    # --- ANOVA Section ---
    elif "ANOVA" in test_type:
        st.subheader("One-Way ANOVA")
        st.markdown("Compares the means of a variable across two or more groups.")
        col1, col2 = st.columns(2)
        with col1:
            # Select the numeric variable
            anova_variable = st.selectbox(
                "Select Variable (Numeric)",
                df.select_dtypes(include=np.number).columns,
                key="anova_variable_select"
            )
        with col2:
            # Select the grouping variable (can have more than 2 groups)
            # Allow selection from all columns, but ANOVA function handles validation
            anova_group_var = st.selectbox(
                "Select Grouping Variable",
                [c for c in df.columns if c != anova_variable and df[c].nunique() > 1], # Exclude variable itself and constant columns
                key="anova_group_select"
            )

        # Button to run ANOVA
        if anova_variable and anova_group_var and st.button("Run ANOVA", key="anova_run_button"):
            anova_results, error_msg = run_anova(df, anova_variable, anova_group_var)
            if error_msg:
                st.error(f"ANOVA Error: {error_msg}")
            elif anova_results is not None:
                st.write("ANOVA Results Table:")
                # Display the ANOVA table, formatting p-value
                st.dataframe(anova_results.style.format({'PR(>F)': '{:.4f}'}))
                # Safely access p-value, check if 'PR(>F)' column exists and has values
                if 'PR(>F)' in anova_results.columns and not anova_results['PR(>F)'].empty:
                    p_value_anova = anova_results['PR(>F)'].iloc[0] # Get the p-value from the first row
                    alpha = 0.05
                    if p_value_anova < alpha:
                        st.success(f"Result is statistically significant (p < {alpha}). There is a significant difference in '{anova_variable}' means across the groups defined by '{anova_group_var}'.")
                        # Suggest Tukey's HSD for post-hoc analysis if significant
                        if df[anova_group_var].nunique() > 2:
                            st.info("Consider running a post-hoc test (like Tukey's HSD) to see which specific groups differ.")
                            # Example of how to potentially add Tukey's HSD (optional, can be complex)
                            # try:
                            #     tukey_data = df[[anova_variable, anova_group_var]].dropna()
                            #     if len(tukey_data) > 0: # Ensure data exists after dropping NaNs
                            #         tukey = pairwise_tukeyhsd(endog=tukey_data[anova_variable], groups=tukey_data[anova_group_var], alpha=0.05)
                            #         st.write("Tukey's HSD Post-Hoc Test Results:")
                            #         st.dataframe(pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0]))
                            #     else:
                            #         st.warning("Not enough data after handling missing values to run Tukey's HSD.")
                            # except Exception as tukey_e:
                            #     st.warning(f"Could not run Tukey's HSD: {tukey_e}")
                    else:
                        st.info(f"Result is not statistically significant (p >= {alpha}). There is no significant difference in '{anova_variable}' means across the groups defined by '{anova_group_var}'.")
                else:
                     st.warning("Could not extract p-value from ANOVA results.")
            else:
                st.error("ANOVA failed to produce results.")

    # --- Regression Section ---
    elif "Regression" in test_type:
        st.subheader("Simple Linear Regression")
        st.markdown("Analyzes the linear relationship between two numeric variables.")
        col1, col2 = st.columns(2)
        numeric_cols_list = df.select_dtypes(include=np.number).columns.tolist()
        with col1:
            # Select the independent variable (X)
            reg_x_variable = st.selectbox(
                "Select Independent Variable (X)",
                numeric_cols_list,
                key="regression_x_select"
            )
        with col2:
            # Select the dependent variable (Y)
            # Ensure Y is different from X
            available_y = [c for c in numeric_cols_list if c != reg_x_variable]
            if available_y:
                 reg_y_variable = st.selectbox(
                     "Select Dependent Variable (Y)",
                     available_y,
                     key="regression_y_select"
                 )
            else:
                 st.warning("Only one numeric variable available. Cannot perform regression.")
                 reg_y_variable = None

        # Button to run regression
        if reg_x_variable and reg_y_variable and st.button("Run Regression", key="regression_run_button"):
            model_fit, error_msg = run_regression(df, reg_x_variable, reg_y_variable)
            if error_msg:
                st.error(f"Regression Error: {error_msg}")
            elif model_fit:
                st.write(f"**Regression Summary: {reg_y_variable} ~ {reg_x_variable}**")
                # Display key results
                st.metric("R-squared (R²)", f"{model_fit.rsquared:.3f}")
                # Safely access params and pvalues, checking if keys exist
                if reg_x_variable in model_fit.params:
                    st.metric(f"Coefficient for {reg_x_variable}", f"{model_fit.params[reg_x_variable]:.3f}")
                if reg_x_variable in model_fit.pvalues:
                     st.metric(f"P-value for {reg_x_variable}", f"{model_fit.pvalues[reg_x_variable]:.4f}")
                if 'const' in model_fit.params:
                    st.metric("Intercept", f"{model_fit.params['const']:.3f}")

                # Interpretation based on p-value of the coefficient
                alpha = 0.05
                if reg_x_variable in model_fit.pvalues:
                    if model_fit.pvalues[reg_x_variable] < alpha:
                        st.success(f"The relationship between '{reg_x_variable}' and '{reg_y_variable}' is statistically significant (p < {alpha}).")
                    else:
                        st.info(f"The relationship between '{reg_x_variable}' and '{reg_y_variable}' is not statistically significant (p >= {alpha}).")
                else:
                    st.warning("Could not determine statistical significance (p-value missing).")


                # Optional: Plot the regression line
                try:
                    fig_reg, ax_reg = plt.subplots(figsize=(8, 5))
                    sns.regplot(x=reg_x_variable, y=reg_y_variable, data=df, ax=ax_reg,
                                line_kws={'color': 'red', 'linestyle': '--'},
                                scatter_kws={'alpha': 0.5})
                    ax_reg.set_title(f"Regression Plot: {reg_y_variable} vs {reg_x_variable}")
                    st.pyplot(fig_reg)
                except Exception as plot_e:
                    st.warning(f"Could not generate regression plot: {plot_e}")

            else:
                st.error("Regression failed to produce results.")

# --- Footer ---
st.markdown("---")
st.caption(f"EcoMonitor Dashboard | Data sourced from specified URLs | Last data point: {df['Date'].max().strftime('%Y-%m-%d')}")

