import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Import seaborn
import geopandas as gpd
import requests
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# --- App Setup ---
st.set_page_config(layout="wide", page_title="EcoMonitor", page_icon="🌿")

# --- Plot Style ---
# Set a default plot style for potentially cleaner visuals
sns.set_style("whitegrid")
# Set default font size for plots (optional, adjust as needed)
plt.rcParams.update({'font.size': 11})


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
st.markdown("""
    <style>
        /* Style the main background */
        .main { background-color: #f8f9fa; padding: 1.5rem; }

        /* Adjust block container padding (may need tweaking with sidebar) */
        .block-container {
            padding-top: 2rem !important; /* Re-adjust top padding slightly */
            padding-bottom: 1rem !important;
            padding-left: 3rem !important; /* Increase left/right padding */
            padding-right: 3rem !important;
        }

        /* Style the sidebar */
        [data-testid="stSidebar"] {
            background-color: #eaf2f8; /* Light blue background */
            padding: 1rem;
        }
         [data-testid="stSidebar"] h1 {
            color: #1a5276; /* Darker blue title */
            font-size: 1.8em;
            margin-bottom: 1rem;
         }
         [data-testid="stSidebar"] .stRadio > label {
             padding-bottom: 10px; /* Space below radio title */
         }
         [data-testid="stSidebar"] .stRadio > div > div {
             padding: 8px 0px; /* Spacing between radio buttons */
             font-size: 1.05em;
         }


        /* Style the video player */
        .stVideo { border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 1rem;}
        /* Style containers (used for metrics) */
        .metric-container {
            background-color: #ffffff;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            padding: 15px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            margin-bottom: 15px;
            text-align: center;
        }
        /* Style the main titles in the main area */
        h1, h2 { color: #2c3e50; font-weight: 600; margin-top: 0rem; padding-top: 0rem;}
        /* Style subheaders in the main area */
        h3 { color: #34495e; margin-top: 1.5rem; margin-bottom: 0.8rem; border-bottom: 1px solid #ddd; padding-bottom: 5px;}
        /* Specific styling for metric containers */
        .metric-container h2 { margin-top: 8px; margin-bottom: 5px; font-size: 2.2em; color: #1a5276; font-weight: 700;}
        .metric-container span { font-size: 0.95em; color: #555; font-weight: 500;}
        /* Ensure plots have some breathing room */
        .stPlotlyChart, .stpyplot { margin-bottom: 1.5rem; background-color: #ffffff; border-radius: 8px; padding: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); border: 1px solid #e0e0e0;}
        /* Remove tab styling as tabs are gone */
        /* Style selectbox and date input */
        .stSelectbox div[data-baseweb="select"] > div { background-color: #ffffff; border-radius: 6px;}
        .stDateInput div[data-baseweb="input"] > div { background-color: #ffffff; border-radius: 6px;}
        /* Style buttons */
        .stButton>button { border-radius: 6px; border: 1px solid #1a5276; background-color: #1a5276; color: white; padding: 8px 16px;}
        .stButton>button:hover { background-color: #154360; border-color: #154360;}

    </style>
""", unsafe_allow_html=True)

# --- Data Loading Functions ---
@st.cache_data # Cache the data to avoid reloading on every interaction
def load_data():
    """Loads, cleans, and preprocesses the dataset from the DATA_URL."""
    try:
        df = pd.read_csv(DATA_URL)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.columns = df.columns.str.replace(' ', '_')
        if df['Date'].isnull().any():
            st.error("Error: Some date values could not be parsed. Please check the 'Date' column format in the CSV.")
            st.stop()
        numeric_cols = [col for col in df.columns if col != 'Date']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.sort_values('Date').dropna(how='all', axis=1)
    except Exception as e:
        st.error(f"Fatal Error: Data loading failed. Cannot start the application. Details: {str(e)}")
        st.stop()

@st.cache_data # Cache the GeoJSON data
def load_geojson():
    """Loads the GeoJSON data from the GEOJSON_URL."""
    try:
        response = requests.get(GEOJSON_URL)
        response.raise_for_status()
        geojson = response.json()
        gdf = gpd.GeoDataFrame.from_features(geojson['features'])
        if not gdf.geometry.is_valid.all():
             gdf.geometry = gdf.geometry.buffer(0)
        return geojson, gdf.geometry.unary_union
    except Exception as e:
        st.error(f"Error: GeoJSON loading failed. Map features might be unavailable. Details: {str(e)}")
        return None, None

# --- Helper Functions ---
def get_parameter_groups(df):
    """Identifies parameter groups (Max, Min, Mean) based on column naming conventions."""
    groups = {}
    for col in df.columns:
        if col == 'Date': continue
        if '_' in col:
            parts = col.split('_')
            if len(parts) >= 2:
                prefix = parts[0]
                parameter = '_'.join(parts[1:])
                if prefix in ['Max', 'Min', 'Mean']:
                    if parameter not in groups:
                         groups[parameter] = {}
                    groups[parameter][prefix] = col
    return {param: data for param, data in groups.items() if param and all(k in data for k in ['Max', 'Min', 'Mean'])}


def normalize_value(value, overall_min, overall_max):
    """Normalizes a value to a 0-1 range based on overall min/max."""
    if pd.isna(value) or pd.isna(overall_min) or pd.isna(overall_max) or overall_max == overall_min:
        return 0.5
    normalized = (np.clip(value, overall_min, overall_max) - overall_min) / (overall_max - overall_min)
    return normalized

# --- Statistical Functions ---
# (Statistical functions remain the same as previous version - run_ttest, run_anova, run_regression)
def run_ttest(data, variable, group_var):
    """Performs an independent two-sample t-test."""
    groups = data[group_var].dropna().unique()
    if len(groups) != 2:
        return None, f"T-Test requires exactly two groups. Found {len(groups)} for '{group_var}'."
    group_data = [data.loc[data[group_var] == grp, variable].dropna() for grp in groups]
    if any(len(d) < 3 for d in group_data):
        return None, f"T-Test requires at least 3 valid (non-NaN) samples per group for '{variable}'. Check data for selected groups."
    try:
        t_stat, p_value = stats.ttest_ind(*group_data, nan_policy='omit')
        return (t_stat, p_value), None
    except Exception as e:
        return None, f"T-Test failed: {str(e)}"

def run_anova(data, variable, group_var):
    """Performs a one-way ANOVA test."""
    if data[group_var].dropna().nunique() < 2:
        return None, f"ANOVA requires at least two distinct groups for '{group_var}'."
    try:
        clean_data = data.dropna(subset=[variable, group_var])
        if clean_data[group_var].nunique() < 2:
             return None, f"After removing missing values, fewer than two groups remain for '{group_var}'."
        if clean_data.groupby(group_var)[variable].nunique().min() == 0:
             return None, f"The variable '{variable}' has no variation within at least one group of '{group_var}'."
        model = ols(f'`{variable}` ~ C(`{group_var}`)', data=clean_data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        return anova_table, None
    except ValueError as ve:
         return None, f"ANOVA failed for '{variable}' by '{group_var}': {str(ve)}"
    except Exception as e:
        return None, f"ANOVA failed for '{variable}' by '{group_var}': {str(e)}"

def run_regression(data, x_var, y_var):
    """Performs a simple linear regression."""
    try:
        data_clean = data[[x_var, y_var]].dropna()
        if len(data_clean) < 10:
            return None, f"Regression requires at least 10 non-missing data points for '{x_var}' and '{y_var}'. Found {len(data_clean)}."
        if data_clean[x_var].nunique() < 2:
            return None, f"Independent variable '{x_var}' has insufficient variation (needs at least 2 unique values)."
        if data_clean[y_var].nunique() < 2:
             return None, f"Dependent variable '{y_var}' has insufficient variation (needs at least 2 unique values)."
        X = sm.add_constant(data_clean[x_var])
        model = sm.OLS(data_clean[y_var], X).fit()
        return model, None
    except Exception as e:
        return None, f"Regression failed for '{y_var}' vs '{x_var}': {str(e)}"

# --- Load Data ---
df = load_data()
geojson, sharaan_boundary = load_geojson()
param_groups = get_parameter_groups(df)
if not param_groups:
    st.error("Error: Could not identify valid parameter groups (Max, Min, Mean) from column names. Please check CSV format.")
    st.stop()

# --- Sidebar Navigation ---
st.sidebar.title("EcoMonitor Navigation")
# Define page names (consider removing emojis if they cause issues with selection logic)
pages = ["Green Cover", "Dashboard", "Correlation", "Temporal", "Statistics"]
# Use radio buttons for navigation
selected_page = st.sidebar.radio("Go to", pages)

# --- Main Page Content (Conditional Display) ---

# --- Page 1: Green Cover ---
if selected_page == "Green Cover":
    st.title("🌳 Sharaan Vegetation Dynamics Monitor")
    col1, col2, col3 = st.columns([1, 5, 1])
    with col2:
        try:
            with open(VIDEO_PATH, 'rb') as video_file:
                video_bytes = video_file.read()
            st.video(video_bytes, format="video/mp4", start_time=0, **VIDEO_CONFIG)
            st.caption("Animation illustrating fluctuations in vegetation indices derived from satellite imagery.")
        except FileNotFoundError:
            st.error(f"Error: Video file not found at '{VIDEO_PATH}'. Please ensure the video file is in the correct location relative to the script.")
        except Exception as e:
            st.error(f"Error loading video: {str(e)}")

# --- Page 2: Climate Dashboard ---
elif selected_page == "Dashboard":
    st.title("📊 Climate & Environmental Dashboard")

    st.subheader("Dashboard Controls")
    control_col1, control_col2 = st.columns(2)

    with control_col1:
        if param_groups:
            groups_list = sorted(param_groups.keys())
            # Use a unique key for this specific selectbox instance
            selected_group_key_dashboard = st.selectbox(
                "Select Parameter Group",
                groups_list,
                key="dashboard_group_select_main", # Unique key
                index=0
            )
        else:
            st.warning("No parameter groups available.")
            selected_group_key_dashboard = None

    with control_col2:
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        # Use a unique key for this specific date input instance
        selected_date_range_dashboard = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="dashboard_date_range_main" # Unique key
        )
    st.markdown("---", unsafe_allow_html=True)

    # --- Data Filtering and Display ---
    if selected_group_key_dashboard and len(selected_date_range_dashboard) == 2:
        group_cols_info = param_groups[selected_group_key_dashboard]
        start_date, end_date = pd.to_datetime(selected_date_range_dashboard[0]), pd.to_datetime(selected_date_range_dashboard[1])

        if start_date > end_date:
            st.error("Error: Start date cannot be after end date.")
            filtered_df = pd.DataFrame()
        else:
             filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

        if not filtered_df.empty:
            # Display Metrics
            metric_cols = st.columns(3)
            metrics_data = {
                'MAX': filtered_df[group_cols_info['Max']].max(),
                'MIN': filtered_df[group_cols_info['Min']].min(),
                'AVG': filtered_df[group_cols_info['Mean']].mean()
            }
            metric_labels = {'MAX': 'Maximum', 'MIN': 'Minimum', 'AVG': 'Average'}

            for i, (label, value) in enumerate(metrics_data.items()):
                with metric_cols[i]:
                    st.markdown(f"""
                        <div class="metric-container">
                            <span>{metric_labels[label]}</span>
                            <h2>{value:.2f}</h2>
                        </div>
                    """, unsafe_allow_html=True)

            st.markdown("---", unsafe_allow_html=True)

            # Visualizations
            st.subheader("Visualizations")
            vis_col1, vis_col2 = st.columns([3, 2])

            # Line chart for trends over time
            with vis_col1:
                fig_line, ax_line = plt.subplots(figsize=(10, 5))
                plot_title = f"{selected_group_key_dashboard.replace('_', ' ').title()} Trend ({selected_date_range_dashboard[0]} to {selected_date_range_dashboard[1]})"
                ax_line.set_title(plot_title, fontsize=14)
                for prefix in ['Max', 'Mean', 'Min']:
                    if prefix in group_cols_info:
                        col_name = group_cols_info[prefix]
                        sns.lineplot(data=filtered_df, x='Date', y=col_name, label=prefix, ax=ax_line, marker='o', markersize=4, linestyle='-')

                ax_line.set_ylabel(selected_group_key_dashboard.replace('_', ' '), fontsize=12)
                ax_line.set_xlabel("Date", fontsize=12)
                ax_line.legend(title="Statistic")
                plt.xticks(rotation=30, ha='right')
                plt.tight_layout()
                st.pyplot(fig_line)

            # Geographical map showing average intensity
            with vis_col2:
                if geojson and sharaan_boundary:
                    try:
                        current_mean_value = filtered_df[group_cols_info['Mean']].mean()
                        overall_min = df[group_cols_info['Mean']].min()
                        overall_max = df[group_cols_info['Mean']].max()
                        normalized_mean = normalize_value(current_mean_value, overall_min, overall_max)

                        fig_map, ax_map = plt.subplots(1, 1, figsize=(6, 6))
                        map_gdf = gpd.GeoDataFrame([1], geometry=[sharaan_boundary], crs="EPSG:4326")
                        map_gdf.plot(
                            ax=ax_map,
                            facecolor=plt.get_cmap('viridis')(normalized_mean),
                            edgecolor='black',
                            linewidth=0.7
                        )
                        ax_map.set_axis_off()
                        ax_map.set_title(f"Average Intensity\nValue: {current_mean_value:.2f}", fontsize=12)
                        plt.tight_layout()
                        st.pyplot(fig_map)
                        st.caption(f"Normalized Avg: {normalized_mean:.2f} (Range: {overall_min:.2f}-{overall_max:.2f})")
                    except Exception as e:
                        st.error(f"Map generation failed: {str(e)}")
                else:
                    st.warning("GeoJSON data not available, cannot display map.")

        else:
            st.warning("No data available for the selected date range and parameter group.")
    else:
         if not selected_group_key_dashboard:
             st.warning("Please select a parameter group using the controls above.")
         elif len(selected_date_range_dashboard) != 2:
              st.warning("Please select a valid date range using the controls above.")


# --- Page 3: Correlation Analysis ---
elif selected_page == "Correlation":
    st.title("🔗 Cross-Parameter Correlation Analysis")

    # Access the selection from the Dashboard page using its unique key
    excluded_group = st.session_state.get("dashboard_group_select_main", None)
    available_groups = sorted([p for p in param_groups.keys() if p != excluded_group])

    if len(available_groups) >= 2:
        selected_corr_groups = st.multiselect(
            "Select parameters to correlate (select at least 2)",
            available_groups,
            default=available_groups[:min(len(available_groups), 3)],
            key="correlation_params_select_main" # Unique key
        )
        st.markdown("---", unsafe_allow_html=True)

        if len(selected_corr_groups) >= 2:
            corr_vars_cols = []
            corr_labels = []
            valid_selection = True
            for p_group in selected_corr_groups:
                if p_group not in param_groups:
                    st.warning(f"Parameter group '{p_group}' seems invalid. Skipping.")
                    valid_selection = False
                    continue
                for stat_type in ['Max', 'Min', 'Mean']:
                    if stat_type in param_groups[p_group]:
                         col_name = param_groups[p_group][stat_type]
                         corr_vars_cols.append(col_name)
                         label_text = p_group.replace('_', ' ').title()
                         corr_labels.append(f"{label_text[:15]}\n({stat_type})")
                    else:
                        st.warning(f"Missing '{stat_type}' column for parameter group '{p_group}'. Skipping this stat.")
                        valid_selection = False

            if corr_vars_cols and valid_selection and len(corr_vars_cols) > 1:
                correlation_matrix = df[corr_vars_cols].corr()
                fig_width = max(8, len(corr_vars_cols) * 0.9)
                fig_height = max(6, len(corr_vars_cols) * 0.8)
                fig_corr, ax_corr = plt.subplots(figsize=(fig_width, fig_height))
                sns.heatmap(
                    correlation_matrix,
                    annot=True, fmt=".2f", cmap="coolwarm",
                    linewidths=.5, linecolor='lightgray',
                    ax=ax_corr, xticklabels=corr_labels, yticklabels=corr_labels,
                    annot_kws={"size": 9}
                )
                ax_corr.set_title("Correlation Matrix Heatmap", fontsize=14)
                plt.xticks(rotation=45, ha='right', fontsize=10)
                plt.yticks(rotation=0, fontsize=10)
                plt.tight_layout(pad=2.0)
                st.pyplot(fig_corr)
            elif len(corr_vars_cols) <= 1:
                 st.warning("Not enough valid columns found for the selected parameters to generate a correlation matrix (need at least 2).")

        else:
            st.info("Please select at least 2 parameter groups to calculate correlations.")
    else:
        st.warning("Not enough parameter groups available (minimum 2 required) to perform correlation analysis, possibly due to the parameter selected on the Dashboard page being excluded.")


# --- Page 4: Temporal Analysis ---
elif selected_page == "Temporal":
    st.title("📈 Temporal Analysis with Rolling Averages")

    if param_groups:
        col1_temp, col2_temp = st.columns([1,1])
        with col1_temp:
            temporal_group_key = st.selectbox(
                "Select Parameter Group",
                sorted(param_groups.keys()),
                key="temporal_group_select_main" # Unique key
            )
        with col2_temp:
            rolling_window_days = st.slider(
                "Select Rolling Window Size (days)",
                min_value=1, max_value=90, value=7, step=1,
                key="temporal_window_slider_main" # Unique key
            )
        st.markdown("---", unsafe_allow_html=True)

        if temporal_group_key:
            if temporal_group_key not in param_groups:
                 st.error(f"Selected parameter group '{temporal_group_key}' is invalid.")
            else:
                group_cols_info = param_groups[temporal_group_key]
                required_cols = [group_cols_info[stat] for stat in ['Max', 'Mean', 'Min'] if stat in group_cols_info]

                if len(required_cols) == 3:
                    ts_data = df.set_index('Date')[required_cols].copy()
                    ts_rolling_avg = ts_data.rolling(window=f'{rolling_window_days}D', min_periods=1).mean()

                    fig_temporal, ax_temporal = plt.subplots(figsize=(12, 5))
                    plot_title_temp = f"{temporal_group_key.replace('_', ' ').title()} - {rolling_window_days}-Day Rolling Statistics"
                    ax_temporal.set_title(plot_title_temp, fontsize=14)

                    for col in ts_rolling_avg.columns:
                        prefix = col.split('_')[0]
                        if prefix in ['Max', 'Mean', 'Min']:
                            sns.lineplot(x=ts_rolling_avg.index, y=ts_rolling_avg[col], label=f'{prefix} ({rolling_window_days}-day avg)', ax=ax_temporal)

                    min_col = group_cols_info.get('Min')
                    max_col = group_cols_info.get('Max')
                    if min_col in ts_rolling_avg.columns and max_col in ts_rolling_avg.columns:
                         ax_temporal.fill_between(
                             ts_rolling_avg.index, ts_rolling_avg[min_col], ts_rolling_avg[max_col],
                             alpha=0.15, color='gray', label='Min-Max Range (Rolling Avg)'
                         )

                    ax_temporal.set_ylabel(temporal_group_key.replace('_', ' '), fontsize=12)
                    ax_temporal.set_xlabel("Date", fontsize=12)
                    ax_temporal.legend(loc='best')
                    plt.tight_layout()
                    st.pyplot(fig_temporal)
                else:
                    st.warning(f"Missing one or more required columns (Max, Mean, Min) for parameter group '{temporal_group_key}'. Cannot generate plot.")
        else:
            st.warning("Please select a parameter group.")
    else:
        st.warning("No parameter groups available for temporal analysis.")


# --- Page 5: Statistics ---
elif selected_page == "Statistics":
    st.title("📉 Statistical Hypothesis Testing")

    test_type = st.selectbox(
        "Select Analysis Type",
        ["T-Test (Compare 2 Groups)", "ANOVA (Compare 2+ Groups)", "Linear Regression (Relationship)"],
        key="stats_test_type_select_main" # Unique key
    )
    st.markdown("---", unsafe_allow_html=True)

    # --- T-Test Section ---
    if "T-Test" in test_type:
        st.subheader("Independent Samples T-Test")
        col1, col2 = st.columns(2)
        with col1:
            numeric_vars_ttest = df.select_dtypes(include=np.number).columns.tolist()
            if not numeric_vars_ttest:
                 st.warning("No numeric variables found in the data for T-Test.")
                 t_test_variable = None
            else:
                t_test_variable = st.selectbox("Variable (Numeric)", numeric_vars_ttest, key="ttest_variable_select_main") # Unique key
        with col2:
            potential_group_vars = [c for c in df.columns if df[c].dropna().nunique() == 2]
            if potential_group_vars:
                 t_test_group_var = st.selectbox("Grouping Variable (2 Groups)", potential_group_vars, key="ttest_group_select_main") # Unique key
            else:
                 st.warning("No suitable grouping variables found (need columns with exactly 2 unique values).")
                 t_test_group_var = None

        if t_test_variable and t_test_group_var:
             if st.button("Run T-Test", key="ttest_run_button_main"): # Unique key
                result, error_msg = run_ttest(df, t_test_variable, t_test_group_var)
                if error_msg: st.error(f"T-Test Error: {error_msg}")
                elif result:
                    t_stat, p_value = result
                    st.markdown("##### Results")
                    res_col1, res_col2 = st.columns(2)
                    with res_col1: st.metric("T-Statistic", f"{t_stat:.3f}")
                    with res_col2: st.metric("P-Value", f"{p_value:.4g}")
                    alpha = 0.05
                    if p_value < alpha: st.success(f"Significant difference found (p < {alpha}).")
                    else: st.info(f"No significant difference found (p >= {alpha}).")
                else: st.error("T-Test failed to produce results.")
        else: st.info("Select a numeric variable and a grouping variable.")

    # --- ANOVA Section ---
    elif "ANOVA" in test_type:
        st.subheader("One-Way ANOVA")
        col1, col2 = st.columns(2)
        with col1:
            numeric_vars_anova = df.select_dtypes(include=np.number).columns.tolist()
            if not numeric_vars_anova:
                 st.warning("No numeric variables found for ANOVA.")
                 anova_variable = None
            else:
                anova_variable = st.selectbox("Variable (Numeric)", numeric_vars_anova, key="anova_variable_select_main") # Unique key
        with col2:
            potential_anova_groups = [c for c in df.columns if c != anova_variable and df[c].dropna().nunique() > 1]
            if potential_anova_groups:
                anova_group_var = st.selectbox("Grouping Variable (2+ Groups)", potential_anova_groups, key="anova_group_select_main") # Unique key
            else:
                 st.warning("No suitable grouping variables found (need columns with >1 unique value).")
                 anova_group_var = None

        if anova_variable and anova_group_var:
            if st.button("Run ANOVA", key="anova_run_button_main"): # Unique key
                anova_results, error_msg = run_anova(df, anova_variable, anova_group_var)
                if error_msg: st.error(f"ANOVA Error: {error_msg}")
                elif anova_results is not None and not anova_results.empty:
                    st.markdown("##### ANOVA Results Table")
                    st.dataframe(anova_results.style.format({'PR(>F)': '{:.4g}'}))
                    if 'PR(>F)' in anova_results.columns and not anova_results['PR(>F)'].empty:
                        p_value_anova = anova_results['PR(>F)'].iloc[0]
                        alpha = 0.05
                        if p_value_anova < alpha:
                            st.success(f"Significant difference found across groups (p < {alpha}).")
                            if df[anova_group_var].dropna().nunique() > 2: st.info("Consider post-hoc tests to compare specific groups.")
                        else: st.info(f"No significant difference found across groups (p >= {alpha}).")
                    else: st.warning("Could not extract p-value from ANOVA results.")
                else: st.error("ANOVA failed or produced empty results.")
        else: st.info("Select a numeric variable and a grouping variable.")

    # --- Regression Section ---
    elif "Regression" in test_type:
        st.subheader("Simple Linear Regression")
        col1, col2 = st.columns(2)
        numeric_cols_list = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols_list) < 2:
             st.warning("Regression requires at least two numeric variables.")
             reg_x_variable, reg_y_variable = None, None
        else:
            with col1:
                reg_x_variable = st.selectbox("Independent Variable (X)", numeric_cols_list, key="regression_x_select_main", index = 0) # Unique key
            with col2:
                available_y = [c for c in numeric_cols_list if c != reg_x_variable]
                if available_y:
                     reg_y_variable = st.selectbox("Dependent Variable (Y)", available_y, key="regression_y_select_main", index = 0 if available_y else -1) # Unique key
                else:
                     st.warning("Error selecting dependent variable.")
                     reg_y_variable = None

        if reg_x_variable and reg_y_variable:
             if st.button("Run Regression", key="regression_run_button_main"): # Unique key
                model_fit, error_msg = run_regression(df, reg_x_variable, reg_y_variable)
                if error_msg: st.error(f"Regression Error: {error_msg}")
                elif model_fit:
                    st.markdown(f"##### Regression Summary: {reg_y_variable} ~ {reg_x_variable}")
                    res_col1, res_col2, res_col3 = st.columns(3)
                    with res_col1: st.metric("R-squared (R²)", f"{model_fit.rsquared:.3f}")
                    if reg_x_variable in model_fit.params:
                         with res_col2: st.metric(f"Coefficient ({reg_x_variable})", f"{model_fit.params[reg_x_variable]:.3f}")
                    if reg_x_variable in model_fit.pvalues:
                         with res_col3: st.metric(f"P-value ({reg_x_variable})", f"{model_fit.pvalues[reg_x_variable]:.4g}")
                    if 'const' in model_fit.params: st.write(f"**Intercept:** {model_fit.params['const']:.3f}")

                    alpha = 0.05
                    if reg_x_variable in model_fit.pvalues:
                        p_val_coeff = model_fit.pvalues[reg_x_variable]
                        if p_val_coeff < alpha: st.success(f"Significant relationship found (p < {alpha}).")
                        else: st.info(f"No significant relationship found (p >= {alpha}).")
                    else: st.warning("Could not determine significance (p-value missing).")

                    st.markdown("---", unsafe_allow_html=True)
                    st.markdown("##### Regression Plot")
                    try:
                        fig_reg, ax_reg = plt.subplots(figsize=(8, 5))
                        sns.regplot(x=reg_x_variable, y=reg_y_variable, data=df, ax=ax_reg,
                                    line_kws={'color': 'red', 'linestyle': '--', 'linewidth': 2},
                                    scatter_kws={'alpha': 0.5, 's': 50})
                        ax_reg.set_title(f"Regression: {reg_y_variable} vs {reg_x_variable}", fontsize=14)
                        ax_reg.set_xlabel(reg_x_variable.replace('_',' ').title(), fontsize=12)
                        ax_reg.set_ylabel(reg_y_variable.replace('_',' ').title(), fontsize=12)
                        plt.tight_layout()
                        st.pyplot(fig_reg)
                    except Exception as plot_e: st.warning(f"Could not generate regression plot: {plot_e}")
                else: st.error("Regression failed to produce results.")
        else: st.info("Select both an independent (X) and a dependent (Y) variable.")


# --- Footer ---
# Display footer only if a page is selected (optional, for cleaner look)
if selected_page:
    st.markdown("---", unsafe_allow_html=True)
    st.caption(f"EcoMonitor Dashboard | Data sourced from specified URLs | Last data point: {df['Date'].max().strftime('%Y-%m-%d')}")
