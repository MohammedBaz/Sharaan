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
# Removed: import plotly.graph_objects as go - No longer needed

# --- App Setup ---
st.set_page_config(layout="wide", page_title="EcoMonitor", page_icon="🌿")

# --- Plot Style ---
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11})


# --- Configuration ---
DATA_URL = "https://raw.githubusercontent.com/MohammedBaz/Sharaan/main/dataset.csv"
GEOJSON_URL = "https://raw.githubusercontent.com/MohammedBaz/Sharaan/main/Sharaan.geojson"
VIDEO_PATH = "GreenCover.mp4"
VIDEO_CONFIG = {"autoplay": False, "muted": True, "loop": False}

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        /* Style the main background */
        .main { background-color: #f8f9fa; padding: 1.5rem; }

        /* Adjust block container padding */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 1rem !important;
            padding-left: 3rem !important;
            padding-right: 3rem !important;
        }

        /* Style the sidebar */
        [data-testid="stSidebar"] {
            background-color: #eaf2f8;
            padding: 1rem;
        }
         [data-testid="stSidebar"] h1 {
            color: #1a5276;
            font-size: 1.8em;
            margin-bottom: 1rem;
         }
         [data-testid="stSidebar"] .stRadio > label {
             padding-bottom: 10px;
             font-weight: 500;
         }
         [data-testid="stSidebar"] .stRadio > div > div {
             padding: 10px 0px;
             font-size: 1.05em;
         }

        /* Style the video player */
        .stVideo { border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 1rem;}

        /* Style st.metric */
        [data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.95em; color: #555; font-weight: 500;
        }
        [data-testid="stMetricValue"] {
             font-size: 2.2em; color: #1a5276; font-weight: 700;
        }

        /* Style the main titles in the main area */
        h1, h2 { color: #2c3e50; font-weight: 600; margin-top: 0rem; padding-top: 0rem;}
        /* Style subheaders in the main area */
        h3 { color: #34495e; margin-top: 1.5rem; margin-bottom: 0.8rem; border-bottom: 1px solid #ddd; padding-bottom: 5px;}

        /* Ensure plots have some breathing room */
        .stpyplot { /* Target only matplotlib plots */
             margin-bottom: 1.5rem;
             background-color: #ffffff;
             border-radius: 8px;
             padding: 10px;
             box-shadow: 0 1px 3px rgba(0,0,0,0.04);
             border: 1px solid #e0e0e0;
        }
        /* Removed CSS for plotly charts */

        /* Style selectbox and date input */
        .stSelectbox div[data-baseweb="select"] > div { background-color: #ffffff; border-radius: 6px;}
        .stDateInput div[data-baseweb="input"] > div { background-color: #ffffff; border-radius: 6px;}
        /* Style buttons */
        .stButton>button { border-radius: 6px; border: 1px solid #1a5276; background-color: #1a5276; color: white; padding: 8px 16px;}
        .stButton>button:hover { background-color: #154360; border-color: #154360;}

    </style>
""", unsafe_allow_html=True)

# --- Data Loading Functions ---
@st.cache_data
def load_data():
    """Loads, cleans, and preprocesses the dataset from the DATA_URL."""
    try:
        df = pd.read_csv(DATA_URL)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.columns = df.columns.str.replace(' ', '_')
        if df['Date'].isnull().any():
            st.error("Error: Some date values could not be parsed.")
            st.stop()
        numeric_cols = [col for col in df.columns if col != 'Date']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        return df.sort_values('Date').dropna(how='all', axis=1)
    except Exception as e:
        st.error(f"Fatal Error: Data loading failed: {str(e)}")
        st.stop()

@st.cache_data
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
        st.error(f"Error: GeoJSON loading failed: {str(e)}")
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
                    if parameter not in groups: groups[parameter] = {}
                    groups[parameter][prefix] = col
    return {param: data for param, data in groups.items() if param and all(k in data for k in ['Max', 'Min', 'Mean'])}


def normalize_value(value, overall_min, overall_max):
    """Normalizes a value to a 0-1 range based on overall min/max."""
    if pd.isna(value) or pd.isna(overall_min) or pd.isna(overall_max) or overall_max == overall_min:
        return 0
    normalized = (np.clip(value, overall_min, overall_max) - overall_min) / (overall_max - overall_min)
    return normalized

# --- Statistical Functions ---
def run_ttest(data, variable, group_var):
    """Performs an independent two-sample t-test."""
    groups = data[group_var].dropna().unique()
    if len(groups) != 2: return None, f"T-Test requires 2 groups. Found {len(groups)}."
    group_data = [data.loc[data[group_var] == grp, variable].dropna() for grp in groups]
    if any(len(d) < 3 for d in group_data): return None, f"T-Test requires >= 3 valid samples per group."
    try:
        t_stat, p_value = stats.ttest_ind(*group_data, nan_policy='omit')
        return (t_stat, p_value), None
    except Exception as e: return None, f"T-Test failed: {str(e)}"

def run_anova(data, variable, group_var):
    """Performs a one-way ANOVA test."""
    if data[group_var].dropna().nunique() < 2: return None, f"ANOVA requires >= 2 groups."
    try:
        clean_data = data.dropna(subset=[variable, group_var])
        if clean_data[group_var].nunique() < 2: return None, f"Fewer than 2 groups remain after removing NaNs."
        if clean_data.groupby(group_var)[variable].nunique().min() < 2 and clean_data[variable].nunique() > 1 : st.warning(f"Warning: '{variable}' has no variation in >= 1 group.")
        elif clean_data[variable].nunique() < 2: return None, f"'{variable}' has insufficient variation overall."
        model = ols(f'`{variable}` ~ C(`{group_var}`)', data=clean_data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        return anova_table, None
    except ValueError as ve: return None, f"ANOVA failed: {str(ve)}"
    except Exception as e: return None, f"ANOVA failed: {str(e)}"

def run_regression(data, x_var, y_var):
    """Performs a simple linear regression."""
    try:
        data_clean = data[[x_var, y_var]].dropna()
        if len(data_clean) < 10: return None, f"Regression requires >= 10 samples. Found {len(data_clean)}."
        if data_clean[x_var].nunique() < 2: return None, f"'{x_var}' has insufficient variation."
        if data_clean[y_var].nunique() < 2: return None, f"'{y_var}' has insufficient variation."
        X = sm.add_constant(data_clean[x_var])
        model = sm.OLS(data_clean[y_var], X).fit()
        return model, None
    except Exception as e: return None, f"Regression failed: {str(e)}"

# --- Plotting Function for Gauge ---
# REMOVED: create_gauge function is no longer needed

# --- Load Data ---
df = load_data()
geojson, sharaan_boundary = load_geojson()
param_groups = get_parameter_groups(df)
if not param_groups:
    st.error("Error: Could not identify valid parameter groups.")
    st.stop()

# --- Sidebar Navigation ---
st.sidebar.title("EcoMonitor Navigation")
pages = ["Green Cover", "Dashboard", "Correlation", "Temporal", "Statistics"]
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
            st.caption("Animation illustrating fluctuations in vegetation indices.")
        except FileNotFoundError: st.error(f"Error: Video file not found at '{VIDEO_PATH}'.")
        except Exception as e: st.error(f"Error loading video: {str(e)}")

# --- Page 2: Climate Dashboard ---
elif selected_page == "Dashboard":
    st.title("📊 Climate & Environmental Dashboard")

    # --- Dashboard Controls ---
    if param_groups:
        groups_list = sorted(param_groups.keys())
        selected_group_key_dashboard = st.selectbox(
            "Select Parameter Group",
            groups_list,
            key="dashboard_group_select_main",
            index=0
        )
    else:
        st.warning("No parameter groups available.")
        selected_group_key_dashboard = None

    st.markdown("---", unsafe_allow_html=True)

    # --- Data Filtering and Display ---
    if selected_group_key_dashboard:
        group_cols_info = param_groups[selected_group_key_dashboard]
        dashboard_df = df

        if not dashboard_df.empty:

            # **MODIFICATION:** Plot first, then metrics below
            # Trend Plot
            st.subheader("Trend Over Time")
            fig_line, ax_line = plt.subplots(figsize=(12, 5))
            plot_title = f"{selected_group_key_dashboard.replace('_', ' ').title()} Trend (Overall)"
            ax_line.set_title(plot_title, fontsize=14)
            for prefix in ['Max', 'Mean', 'Min']:
                if prefix in group_cols_info:
                    col_name = group_cols_info[prefix]
                    # Ensure line plot without markers
                    sns.lineplot(data=dashboard_df, x='Date', y=col_name, label=prefix, ax=ax_line, linestyle='-', linewidth=1.5)

            ax_line.set_ylabel(selected_group_key_dashboard.replace('_', ' '), fontsize=12)
            ax_line.set_xlabel("Date", fontsize=12)
            ax_line.legend(title="Statistic")
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            st.pyplot(fig_line, use_container_width=True) # Plot takes full width

            st.markdown("---", unsafe_allow_html=True) # Separator

            # Key Statistics using st.metric
            st.subheader(f"Key Statistics: {selected_group_key_dashboard.replace('_', ' ').title()}")
            metric_cols = st.columns(3) # Arrange metrics horizontally

            # Calculate overall metrics
            overall_max_val = dashboard_df[group_cols_info['Max']].max()
            overall_min_val = dashboard_df[group_cols_info['Min']].min()
            overall_mean_val = dashboard_df[group_cols_info['Mean']].mean()

            with metric_cols[0]:
                st.metric(label="Overall Maximum", value=f"{overall_max_val:.2f}")
            with metric_cols[1]:
                 st.metric(label="Overall Average", value=f"{overall_mean_val:.2f}") # Swapped order for common layout
            with metric_cols[2]:
                st.metric(label="Overall Minimum", value=f"{overall_min_val:.2f}")

            # REMOVED Gauge creation and display logic

        else:
            st.warning("No data available for the selected parameter group.")
    else:
         st.warning("Please select a parameter group using the control above.")


# --- Page 3: Correlation Analysis ---
elif selected_page == "Correlation":
    st.title("🔗 Cross-Parameter Correlation Analysis")
    excluded_group = st.session_state.get("dashboard_group_select_main", None)
    available_groups = sorted([p for p in param_groups.keys() if p != excluded_group])

    if len(available_groups) >= 2:
        selected_corr_groups = st.multiselect(
            "Select parameters to correlate (select at least 2)",
            available_groups,
            default=available_groups[:min(len(available_groups), 3)],
            key="correlation_params_select_main"
        )
        st.markdown("---", unsafe_allow_html=True)

        if len(selected_corr_groups) >= 2:
            corr_vars_cols = []
            corr_labels = []
            valid_selection = True
            for p_group in selected_corr_groups:
                if p_group not in param_groups: st.warning(f"Invalid group '{p_group}'."); valid_selection = False; continue
                for stat_type in ['Max', 'Min', 'Mean']:
                    if stat_type in param_groups[p_group]:
                         col_name = param_groups[p_group][stat_type]; corr_vars_cols.append(col_name)
                         label_text = p_group.replace('_', ' ').title(); corr_labels.append(f"{label_text[:15]}\n({stat_type})")
                    else: st.warning(f"Missing '{stat_type}' for '{p_group}'."); valid_selection = False

            if corr_vars_cols and valid_selection and len(corr_vars_cols) > 1:
                correlation_matrix = df[corr_vars_cols].corr()
                fig_width = max(8, len(corr_vars_cols) * 0.9); fig_height = max(6, len(corr_vars_cols) * 0.8)
                fig_corr, ax_corr = plt.subplots(figsize=(fig_width, fig_height))
                sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5, linecolor='lightgray', ax=ax_corr, xticklabels=corr_labels, yticklabels=corr_labels, annot_kws={"size": 9})
                ax_corr.set_title("Correlation Matrix Heatmap", fontsize=14)
                plt.xticks(rotation=45, ha='right', fontsize=10); plt.yticks(rotation=0, fontsize=10)
                plt.tight_layout(pad=2.0); st.pyplot(fig_corr)
            elif len(corr_vars_cols) <= 1: st.warning("Need >= 2 valid columns for correlation.")
        else: st.info("Please select at least 2 parameters.")
    else: st.warning("Not enough parameters available for correlation.")

# --- Page 4: Temporal Analysis ---
elif selected_page == "Temporal":
    st.title("📈 Temporal Analysis with Rolling Averages")
    if param_groups:
        col1_temp, col2_temp = st.columns([1,1])
        with col1_temp: temporal_group_key = st.selectbox("Select Parameter Group", sorted(param_groups.keys()), key="temporal_group_select_main")
        with col2_temp: rolling_window_days = st.slider("Select Rolling Window Size (days)", 1, 90, 7, 1, key="temporal_window_slider_main")
        st.markdown("---", unsafe_allow_html=True)

        if temporal_group_key:
            if temporal_group_key not in param_groups: st.error(f"Invalid group '{temporal_group_key}'.")
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
                        if prefix in ['Max', 'Mean', 'Min']: sns.lineplot(x=ts_rolling_avg.index, y=ts_rolling_avg[col], label=f'{prefix} ({rolling_window_days}-day avg)', ax=ax_temporal, linewidth=1.5)
                    min_col = group_cols_info.get('Min'); max_col = group_cols_info.get('Max')
                    if min_col in ts_rolling_avg.columns and max_col in ts_rolling_avg.columns:
                         ax_temporal.fill_between(ts_rolling_avg.index, ts_rolling_avg[min_col], ts_rolling_avg[max_col], alpha=0.15, color='gray', label='Min-Max Range')
                    ax_temporal.set_ylabel(temporal_group_key.replace('_', ' '), fontsize=12); ax_temporal.set_xlabel("Date", fontsize=12)
                    ax_temporal.legend(loc='best'); plt.tight_layout(); st.pyplot(fig_temporal, use_container_width=True)
                else: st.warning(f"Missing required columns for '{temporal_group_key}'.")
        else: st.warning("Please select a parameter group.")
    else: st.warning("No parameter groups available.")

# --- Page 5: Statistics ---
elif selected_page == "Statistics":
    st.title("📉 Statistical Hypothesis Testing")
    test_type = st.selectbox("Select Analysis Type", ["T-Test", "ANOVA", "Regression"], key="stats_test_type_select_main")
    st.markdown("---", unsafe_allow_html=True)

    if "T-Test" in test_type:
        st.subheader("Independent Samples T-Test")
        col1, col2 = st.columns(2)
        with col1:
            numeric_vars_ttest = df.select_dtypes(include=np.number).columns.tolist()
            t_test_variable = st.selectbox("Variable (Numeric)", numeric_vars_ttest, key="ttest_variable_select_main") if numeric_vars_ttest else None
            if not t_test_variable: st.warning("No numeric variables found.")
        with col2:
            potential_group_vars = [c for c in df.columns if df[c].dropna().nunique() == 2]
            t_test_group_var = st.selectbox("Grouping Variable (2 Groups)", potential_group_vars, key="ttest_group_select_main") if potential_group_vars else None
            if not t_test_group_var: st.warning("No suitable grouping variables found.")

        if t_test_variable and t_test_group_var:
             if st.button("Run T-Test", key="ttest_run_button_main"):
                result, error_msg = run_ttest(df, t_test_variable, t_test_group_var)
                if error_msg: st.error(f"T-Test Error: {error_msg}")
                elif result:
                    t_stat, p_value = result; st.markdown("##### Results")
                    res_col1, res_col2 = st.columns(2)
                    with res_col1: st.metric("T-Statistic", f"{t_stat:.3f}")
                    with res_col2: st.metric("P-Value", f"{p_value:.4g}")
                    alpha = 0.05
                    if p_value < alpha: st.success(f"Significant difference (p < {alpha}).")
                    else: st.info(f"No significant difference (p >= {alpha}).")
                else: st.error("T-Test failed.")
        else: st.info("Select a numeric variable and a grouping variable.")

    elif "ANOVA" in test_type:
        st.subheader("One-Way ANOVA")
        col1, col2 = st.columns(2)
        with col1:
            numeric_vars_anova = df.select_dtypes(include=np.number).columns.tolist()
            anova_variable = st.selectbox("Variable (Numeric)", numeric_vars_anova, key="anova_variable_select_main") if numeric_vars_anova else None
            if not anova_variable: st.warning("No numeric variables found.")
        with col2:
            potential_anova_groups = [c for c in df.columns if c != anova_variable and df[c].dropna().nunique() > 1]
            anova_group_var = st.selectbox("Grouping Variable (2+ Groups)", potential_anova_groups, key="anova_group_select_main") if potential_anova_groups else None
            if not anova_group_var: st.warning("No suitable grouping variables found.")

        if anova_variable and anova_group_var:
            if st.button("Run ANOVA", key="anova_run_button_main"):
                anova_results, error_msg = run_anova(df, anova_variable, anova_group_var)
                if error_msg: st.error(f"ANOVA Error: {error_msg}")
                elif anova_results is not None and not anova_results.empty:
                    st.markdown("##### ANOVA Results Table")
                    st.dataframe(anova_results.style.format({'PR(>F)': '{:.4g}'}))
                    if 'PR(>F)' in anova_results.columns and not anova_results['PR(>F)'].empty:
                        p_value_anova = anova_results['PR(>F)'].iloc[0]; alpha = 0.05
                        if p_value_anova < alpha:
                            st.success(f"Significant difference across groups (p < {alpha}).")
                            if df[anova_group_var].dropna().nunique() > 2: st.info("Consider post-hoc tests.")
                        else: st.info(f"No significant difference across groups (p >= {alpha}).")
                    else: st.warning("Could not extract p-value.")
                else: st.error("ANOVA failed or produced empty results.")
        else: st.info("Select a numeric variable and a grouping variable.")

    elif "Regression" in test_type:
        st.subheader("Simple Linear Regression")
        col1, col2 = st.columns(2)
        numeric_cols_list = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols_list) < 2: st.warning("Regression requires >= 2 numeric variables.")
        else:
            with col1: reg_x_variable = st.selectbox("Independent Variable (X)", numeric_cols_list, key="regression_x_select_main", index = 0)
            with col2:
                available_y = [c for c in numeric_cols_list if c != reg_x_variable]
                reg_y_variable = st.selectbox("Dependent Variable (Y)", available_y, key="regression_y_select_main", index = 0 if available_y else -1) if available_y else None
                if not reg_y_variable: st.warning("Error selecting dependent variable.")

            if reg_x_variable and reg_y_variable:
                 if st.button("Run Regression", key="regression_run_button_main"):
                    model_fit, error_msg = run_regression(df, reg_x_variable, reg_y_variable)
                    if error_msg: st.error(f"Regression Error: {error_msg}")
                    elif model_fit:
                        st.markdown(f"##### Summary: {reg_y_variable} ~ {reg_x_variable}")
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
                            if p_val_coeff < alpha: st.success(f"Significant relationship (p < {alpha}).")
                            else: st.info(f"No significant relationship (p >= {alpha}).")
                        else: st.warning("Could not determine significance.")
                        st.markdown("---", unsafe_allow_html=True)
                        st.markdown("##### Regression Plot")
                        try:
                            fig_reg, ax_reg = plt.subplots(figsize=(8, 5))
                            sns.regplot(x=reg_x_variable, y=reg_y_variable, data=df, ax=ax_reg, line_kws={'color': 'red', 'linestyle': '--', 'linewidth': 2}, scatter_kws={'alpha': 0.5, 's': 50})
                            ax_reg.set_title(f"Regression: {reg_y_variable} vs {reg_x_variable}", fontsize=14)
                            ax_reg.set_xlabel(reg_x_variable.replace('_',' ').title(), fontsize=12); ax_reg.set_ylabel(reg_y_variable.replace('_',' ').title(), fontsize=12)
                            plt.tight_layout(); st.pyplot(fig_reg, use_container_width=True)
                        except Exception as plot_e: st.warning(f"Could not generate plot: {plot_e}")
                    else: st.error("Regression failed.")
            else: st.info("Select both independent (X) and dependent (Y) variables.")

# --- Footer ---
if selected_page:
    st.markdown("---", unsafe_allow_html=True)
    st.caption(f"EcoMonitor Dashboard | Data sourced from specified URLs | Last data point: {df['Date'].max().strftime('%Y-%m-%d')}")

