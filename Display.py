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
        df.columns = df.columns.str.replace(' ', '_')
        
        if df['Date'].isnull().any():
            invalid_dates = df[df['Date'].isnull()]['Date'].unique()
            st.error(f"Invalid date formats found: {invalid_dates}")
            st.stop()
            
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
        
        if not gdf.geometry.is_valid.all():
            gdf.geometry = gdf.geometry.buffer(0)
        
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
        data_clean = data[[x_var, y_var]].dropna()
        if len(data_clean) < 10:
            return None, "At least 10 samples required"
            
        if data_clean[x_var].nunique() == 1:
            return None, "Independent variable must have variation"
            
        if data_clean[y_var].nunique() == 1:
            return None, "Dependent variable must have variation"
        
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
param_groups = get_parameter_groups(df)

# --- Main App Structure ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Climate Dashboard", 
    "Correlation Analysis",
    "Temporal Analysis", 
    "Statistical Tests"
])

# Shared group selection with unique keys
def group_selector(default_group='Air_temperature', key_suffix=""):
    groups = sorted(param_groups.keys())
    default_index = groups.index(default_group) if default_group in groups else 0
    return st.selectbox(
        "Parameter Group", 
        groups, 
        index=default_index,
        key=f"param_group_{key_suffix}"
    )

# --- Tab1: Climate Dashboard ---
with tab1:
    st.title("🌦️ Sharaan Climate Dashboard")
    
    with st.sidebar:
        st.header("Controls")
        selected_group = group_selector(key_suffix="dashboard")
        date_range = st.date_input("Date Range", 
                                 value=(df.Date.min().date(), df.Date.max().date()),
                                 min_value=df.Date.min().date(),
                                 max_value=df.Date.max().date())

    group_data = param_groups[selected_group]
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_df = df[(df.Date >= start_date) & (df.Date <= end_date)]
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Maximum", f"{filtered_df[group_data['Max']].max():.2f}")
    with col2:
        st.metric("Minimum", f"{filtered_df[group_data['Min']].min():.2f}")
    with col3:
        st.metric("Average", f"{filtered_df[group_data['Mean']].mean():.2f}")

    # Main content
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("Trend Analysis")
        fig, ax = plt.subplots(figsize=(12, 5))
        style_map = {
            'Max': {'color': '#e74c3c', 'linestyle': '--'},
            'Min': {'color': '#3498db', 'linestyle': '--'},
            'Mean': {'color': '#2ecc71', 'linewidth': 2}
        }
        
        for prefix in ['Max', 'Mean', 'Min']:
            col = group_data[prefix]
            sns.lineplot(data=filtered_df, x='Date', y=col, ax=ax, 
                        label=f"{prefix} {selected_group}", **style_map[prefix])
        
        ax.set_title(f"{selected_group.replace('_', ' ').title()} Trends")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col_right:
        st.subheader("Spatial Distribution")
        try:
            current_avg = filtered_df[group_data['Mean']].mean()
            overall_min = df[group_data['Mean']].min()
            overall_max = df[group_data['Mean']].max()
            norm_value = normalize_value(current_avg, overall_min, overall_max)
            
            fig, ax = plt.subplots(figsize=(6, 6))
            gdf = gpd.GeoDataFrame.from_features(geojson['features'])
            gdf.plot(ax=ax, facecolor=plt.get_cmap(COLORMAP_NAME)(norm_value), edgecolor='black')
            ax.set_axis_off()
            ax.set_title(f"Mean {selected_group} Intensity")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Map error: {str(e)}")

# --- Tab2: Correlation Analysis ---
with tab2:
    st.title("📊 Correlation Analysis")
    
    # Get selected group from Tab1's selection
    selected_group = st.session_state.get("param_group_dashboard", "Air_temperature")
    group_data = param_groups.get(selected_group)
    
    if not group_data:
        st.error("No valid parameter group selected")
        st.stop()
    
    # Extract parameters
    params = [group_data['Max'], group_data['Min'], group_data['Mean']]
    
    # Calculate correlations
    st.subheader(f"Correlation Matrix: {selected_group.replace('_', ' ').title()}")
    corr_matrix = df[params].corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        mask=np.triu(np.ones_like(corr_matrix)),
        ax=ax
    )
    ax.set_xticklabels(['Max', 'Min', 'Mean'], rotation=0)
    ax.set_yticklabels(['Max', 'Min', 'Mean'], rotation=0)
    st.pyplot(fig)

# --- Tab3: Temporal Analysis ---
with tab3:
    st.title("⏳ Temporal Analysis")
    selected_group = group_selector(key_suffix="temporal")
    group_data = param_groups[selected_group]
    
    st.subheader(f"Detailed Analysis: {selected_group.replace('_', ' ').title()}")
    window_size = st.slider("Rolling Average Window (Days)", 1, 90, 7, key='temporal_window')
    
    ts_data = df.set_index('Date')[list(group_data.values())].rolling(window=window_size).mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for prefix, col in group_data.items():
        sns.lineplot(data=ts_data, x=ts_data.index, y=col, ax=ax,
                    label=f"{prefix} {selected_group}")
    
    ax.fill_between(ts_data.index,
                    ts_data[group_data['Min']],
                    ts_data[group_data['Max']],
                    color='#95a5a6', alpha=0.2)
    
    ax.set_title(f"{selected_group.replace('_', ' ').title()} with {window_size}-Day Rolling Average")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# --- Tab4: Statistical Tests ---
with tab4:
    st.title("📈 Statistical Testing")
    selected_group = group_selector(key_suffix="stats")
    group_data = param_groups[selected_group]
    
    test_type = st.selectbox("Select Test", ["T-Test", "ANOVA", "Regression"], key='test_type')
    
    if test_type == "T-Test":
        col1, col2 = st.columns(2)
        with col1:
            variable = st.selectbox("Variable", list(group_data.values()), key='ttest_var')
        with col2:
            valid_group_vars = [col for col in df.columns if df[col].nunique() == 2]
            group_var = st.selectbox("Group Variable", valid_group_vars, key='ttest_group')
        
        if st.button("Run T-Test", key='ttest_btn'):
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
                
                fig, ax = plt.subplots()
                sns.boxplot(x=group_var, y=variable, data=df)
                st.pyplot(fig)
    
    elif test_type == "ANOVA":
        variable = st.selectbox("Variable", list(group_data.values()), key='anova_var')
        group_var = st.selectbox("Group Variable", df.columns, key='anova_group')
        
        if st.button("Run ANOVA", key='anova_btn'):
            result, error = run_anova(df, variable, group_var)
            if error:
                st.error(error)
            else:
                st.dataframe(result.style.format("{:.4f}"))
                
                st.subheader("Post-hoc Analysis")
                try:
                    tukey = pairwise_tukeyhsd(df[variable], df[group_var])
                    st.text(str(tukey))
                except Exception as e:
                    st.error(f"Post-hoc error: {str(e)}")
    
    elif test_type == "Regression":
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("Independent Variable", list(group_data.values()), key='reg_x')
        with col2:
            y_var = st.selectbox("Dependent Variable", list(group_data.values()), key='reg_y')
        
        if st.button("Run Regression", key='reg_btn'):
            model, error = run_regression(df, x_var, y_var)
            if error:
                st.error(error)
            else:
                st.subheader("Regression Results")
                try:
                    st.text(model.summary().as_text())
                except Exception:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("R-squared", f"{model.rsquared:.2f}")
                        st.metric("Coefficient", f"{model.params[1]:.2f}")
                    with col2:
                        st.metric("P-value", f"{model.pvalues[1]:.4f}")
                        st.metric("Observations", model.nobs)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                sm.qqplot(model.resid, line='s', ax=ax1)
                sns.scatterplot(x=model.fittedvalues, y=model.resid, ax=ax2)
                ax2.axhline(0, color='red', linestyle='--')
                st.pyplot(fig)

# --- Footer ---
st.markdown("---")
st.caption(f"Data updated: {df.Date.max().strftime('%Y-%m-%d')} | Protected Area Monitoring System")
