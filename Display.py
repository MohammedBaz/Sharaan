import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import random
import requests
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings

# Configuration
DATA_URL = "https://raw.githubusercontent.com/MohammedBaz/Sharaan/main/dataset.csv"
GEOJSON_URL = "https://raw.githubusercontent.com/MohammedBaz/Sharaan/main/Sharaan.geojson"
COLORMAP_NAME = 'viridis'

# --- Data Loading Functions ---
@st.cache_data
def load_data():
    """Load and validate climate dataset"""
    try:
        df = pd.read_csv(DATA_URL)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Standardize column names
        df.columns = df.columns.str.replace(' ', '_')
        
        if df['Date'].isnull().any():
            invalid_dates = df[df['Date'].isnull()]['Date'].unique()
            st.error(f"Invalid date formats found: {invalid_dates}")
            st.stop()
            
        # Convert numeric columns
        numeric_cols = [col for col in df.columns if col != 'Date']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df.sort_values('Date').dropna(how='all', axis=1)
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        st.stop()

@st.cache_data
def load_geojson():
    """Load and process geospatial data"""
    try:
        response = requests.get(GEOJSON_URL)
        response.raise_for_status()
        geojson = response.json()
        gdf = gpd.GeoDataFrame.from_features(geojson['features'])
        
        # Handle geometries
        if not gdf.geometry.is_valid.all():
            gdf.geometry = gdf.geometry.buffer(0)
        
        # Union polygons using updated method
        try:
            polygon = gdf.geometry.union_all()
        except AttributeError:
            polygon = gdf.geometry.unary_union
            
        return geojson, polygon
    except Exception as e:
        st.error(f"Geojson error: {str(e)}")
        st.stop()

# --- Helper Functions ---
def normalize_value(value, overall_min, overall_max):
    """Normalize value between 0-1 for colormap"""
    if pd.isna(value) or pd.isna(overall_min) or pd.isna(overall_max):
        return 0.5
    if overall_max == overall_min:
        return 0.5
    return (np.clip(value, overall_min, overall_max) - overall_min) / (overall_max - overall_min)

def get_parameter_groups(df):
    """Identify parameter groups from column names"""
    groups = {}
    for col in df.columns:
        if col == 'Date': continue
        if '_' in col:
            # Split into prefix and parameter name
            parts = col.split('_')
            if len(parts) < 2: continue
            prefix = parts[0]
            parameter = '_'.join(parts[1:])
            
            if parameter not in groups:
                groups[parameter] = {
                    'Max': f"Max_{parameter}",
                    'Min': f"Min_{parameter}",
                    'Mean': f"Mean_{parameter}"
                }
    return groups

# --- Statistical Functions ---
def run_ttest(data, variable, group_var):
    """Independent t-test with validation"""
    groups = data[group_var].dropna().unique()
    if len(groups) != 2:
        return None, "Exactly two groups required"
    
    group_data = [data[data[group_var] == grp][variable].dropna() for grp in groups]
    if len(group_data[0]) < 3 or len(group_data[1]) < 3:
        return None, "Minimum 3 samples per group required"
    
    t_stat, p_value = stats.ttest_ind(*group_data)
    return (t_stat, p_value), None

def run_anova(data, variable, group_var):
    """One-way ANOVA with validation"""
    if len(data[group_var].unique()) < 2:
        return None, "Minimum 2 groups required"
    
    try:
        model = ols(f'{variable} ~ C({group_var})', data=data).fit()
        return sm.stats.anova_lm(model, typ=2), None
    except Exception as e:
        return None, str(e)

def run_regression(data, x_var, y_var):
    """Linear regression with enhanced error handling"""
    try:
        # Clean data and validate inputs
        data_clean = data[[x_var, y_var]].dropna()
        if len(data_clean) < 10:
            return None, "At least 10 samples required"
            
        if data_clean[x_var].nunique() == 1:
            return None, "Independent variable must have variation"
            
        if data_clean[y_var].nunique() == 1:
            return None, "Dependent variable must have variation"
        
        # Add constant and fit model
        X = sm.add_constant(data_clean[x_var])
        y = data_clean[y_var]
        model = sm.OLS(y, X).fit()
        
        return model, None
        
    except Exception as e:
        return None, f"Regression failed: {str(e)}"

# --- App Setup ---
st.set_page_config(layout="wide")
df = load_data()
geojson, boundary_polygon = load_geojson()

# --- Main App Structure ---
tab1, tab2, tab3 = st.tabs(["Climate Dashboard", "Temporal Analysis", "Statistical Tests"])

with tab1:
    # --- Dashboard Tab ---
    st.title("🌦️ Sharaan Climate Dashboard")
    
    # Sidebar Controls
    with st.sidebar:
        st.header("Controls")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        selected_var = st.selectbox("Parameter", numeric_cols, 
                                  index=numeric_cols.index('Rainfall') if 'Rainfall' in numeric_cols else 0)
        date_range = st.date_input("Date Range", 
                                 value=(df.Date.min().date(), df.Date.max().date()),
                                 min_value=df.Date.min().date(),
                                 max_value=df.Date.max().date())
    
    # Data Filtering
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_df = df[(df.Date >= start_date) & (df.Date <= end_date)]
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average", f"{filtered_df[selected_var].mean():.2f}")
    with col2:
        st.metric("Maximum", f"{filtered_df[selected_var].max():.2f}")
    with col3:
        st.metric("Minimum", f"{filtered_df[selected_var].min():.2f}")
    
    # Main Content
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Temporal Plot
        st.subheader("Temporal Trends")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=filtered_df, x='Date', y=selected_var, color='#2ecc71')
        ax.set_ylabel(selected_var)
        st.pyplot(fig)
    
    with col_right:
        # Spatial Visualization
        st.subheader("Spatial Intensity")
        try:
            # Calculate normalized intensity
            overall_min = df[selected_var].min()
            overall_max = df[selected_var].max()
            current_avg = filtered_df[selected_var].mean()
            norm_value = normalize_value(current_avg, overall_min, overall_max)
            
            # Create map plot
            fig, ax = plt.subplots(figsize=(6, 6))
            gdf = gpd.GeoDataFrame.from_features(geojson['features'])
            gdf.plot(ax=ax, facecolor=plt.get_cmap(COLORMAP_NAME)(norm_value), edgecolor='black')
            ax.set_axis_off()
            ax.set_title(f"{selected_var} Intensity")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Map error: {str(e)}")

with tab2:
    # --- Temporal Analysis Tab ---
    st.title("⏳ Temporal Analysis")
    st.subheader("Parameter Group Trends")
    
    # Get parameter groups
    param_groups = get_parameter_groups(df)
    selected_group = st.selectbox("Select Parameter Group", sorted(param_groups.keys()))
    
    # Get relevant columns
    group_cols = param_groups[selected_group]
    cols_to_plot = [group_cols['Max'], group_cols['Min'], group_cols['Mean']]
    
    # Rolling Average Control
    window_size = st.slider("Rolling Average Window (Days)", 1, 90, 7)
    
    # Process data
    ts_data = filtered_df.set_index('Date')[cols_to_plot].rolling(window=window_size).mean()
    
    # Create plot with styling
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Style configuration
    style_map = {
        'Max': {'color': '#e74c3c', 'linestyle': '--', 'label': 'Max'},
        'Min': {'color': '#3498db', 'linestyle': '--', 'label': 'Min'},
        'Mean': {'color': '#2ecc71', 'linewidth': 2, 'label': 'Mean'}
    }
    
    # Plot lines
    for prefix in ['Max', 'Mean', 'Min']:
        col = group_cols[prefix]
        sns.lineplot(data=ts_data, x=ts_data.index, y=col, ax=ax,
                    **style_map[prefix])
    
    # Add shaded range
    ax.fill_between(ts_data.index,
                    ts_data[group_cols['Min']],
                    ts_data[group_cols['Max']],
                    color='#95a5a6', alpha=0.2,
                    label='Min-Max Range')
    
    # Formatting
    title = selected_group.replace('_', ' ').title()
    ax.set_title(f"{title} Trends with {window_size}-Day Rolling Average")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)

with tab3:
    # --- Statistical Testing Tab ---
    st.title("📊 Statistical Testing")
    
    test_type = st.selectbox("Select Test", ["T-Test", "ANOVA", "Regression"])
    numeric_vars = df.select_dtypes(include=np.number).columns.tolist()
    
    if test_type == "T-Test":
        col1, col2 = st.columns(2)
        with col1:
            variable = st.selectbox("Variable", numeric_vars)
        with col2:
            valid_group_vars = [col for col in df.columns if df[col].nunique() == 2]
            if not valid_group_vars:
                st.error("No valid group variables found. Requires a column with exactly two categories.")
                st.stop()
            group_var = st.selectbox("Group Variable", valid_group_vars)
        
        if st.button("Run T-Test"):
            result, error = run_ttest(df, variable, group_var)
            if error:
                st.error(error)
            else:
                t_stat, p_value = result
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("t-statistic", f"{t_stat:.2f}")
                with col2:
                    st.metric("p-value", f"{p_value:.4f}")
                
                # Visualization
                fig, ax = plt.subplots()
                sns.boxplot(x=group_var, y=variable, data=df)
                st.pyplot(fig)
    
    elif test_type == "ANOVA":
        variable = st.selectbox("Variable", numeric_vars)
        group_var = st.selectbox("Group Variable", df.columns)
        
        if st.button("Run ANOVA"):
            result, error = run_anova(df, variable, group_var)
            if error:
                st.error(error)
            else:
                st.dataframe(result.style.format("{:.4f}"))
                
                # Post-hoc analysis
                st.subheader("Post-hoc Analysis")
                try:
                    tukey = pairwise_tukeyhsd(df[variable], df[group_var])
                    st.text(str(tukey))
                except Exception as e:
                    st.error(f"Post-hoc error: {str(e)}")
    
    elif test_type == "Regression":
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("Independent Variable", numeric_vars)
        with col2:
            y_var = st.selectbox("Dependent Variable", numeric_vars)
        
        if st.button("Run Regression"):
            model, error = run_regression(df, x_var, y_var)
            if error:
                st.error(error)
            else:
                st.subheader("Regression Results")
                try:
                    st.text(model.summary().as_text())
                except Exception as summary_error:
                    st.error("Could not generate full summary. Key metrics:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("R-squared", f"{model.rsquared:.2f}")
                        st.metric("Coefficient", f"{model.params[1]:.2f}")
                    with col2:
                        st.metric("P-value", f"{model.pvalues[1]:.4f}")
                        st.metric("Observations", model.nobs)
                
                # Diagnostic plots
                st.subheader("Diagnostic Plots")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                sm.qqplot(model.resid, line='s', ax=ax1)
                sns.scatterplot(x=model.fittedvalues, y=model.resid, ax=ax2)
                ax2.axhline(0, color='red', linestyle='--')
                st.pyplot(fig)

# --- Footer ---
st.markdown("---")
st.caption(f"Data updated: {df.Date.max().strftime('%Y-%m-%d')} | Protected Area Monitoring System")
