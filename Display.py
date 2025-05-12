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

# Configuration
DATA_URL = "https://raw.githubusercontent.com/MohammedBaz/Sharaan/main/dataset.csv"
GEOJSON_URL = "https://raw.githubusercontent.com/MohammedBaz/Sharaan/main/Sharaan.geojson"
VIDEO_PATH = "GreenCover.mp4"
VIDEO_CONFIG = {"autoplay": False, "muted": True, "loop": False}

# Custom CSS styling
st.markdown("""
    <style>
        .main {background-color: #f8f9fa;}
        .stVideo {border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
        .css-1aumxhk {background-color: #ffffff; border-radius: 10px; padding: 20px;}
        h1 {color: #2c3e50;}
        .metric-container {background: #ffffff; border-radius: 10px; padding: 15px;}
    </style>
""", unsafe_allow_html=True)

# --- Data Loading Functions ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_URL)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.columns = df.columns.str.replace(' ', '_')
        
        if df['Date'].isnull().any():
            invalid_dates = df[df['Date'].isnull()]['Date'].unique()
            st.error(f"Invalid date formats: {invalid_date}")
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
    try:
        response = requests.get(GEOJSON_URL)
        response.raise_for_status()
        geojson = response.json()
        gdf = gpd.GeoDataFrame.from_features(geojson['features'])
        gdf.geometry = gdf.geometry.buffer(0) if not gdf.geometry.is_valid.all() else gdf.geometry
        return geojson, gdf.geometry.unary_union
    except Exception as e:
        st.error(f"Geojson error: {str(e)}")
        st.stop()

# --- Helper Functions ---
def get_parameter_groups(df):
    groups = {}
    for col in df.columns:
        if col == 'Date': continue
        if '_' in col:
            parts = col.split('_')
            if len(parts) < 2: continue
            parameter = '_'.join(parts[1:])
            groups.setdefault(parameter, {
                'Max': f"Max_{parameter}",
                'Min': f"Min_{parameter}",
                'Mean': f"Mean_{parameter}"
            })
    return groups

def normalize_value(value, overall_min, overall_max):
    if pd.isna(value) or pd.isna(overall_min) or pd.isna(overall_max):
        return 0.5
    return (np.clip(value, overall_min, overall_max) - overall_min) / (overall_max - overall_min)

# --- Statistical Functions ---
def run_ttest(data, variable, group_var):
    groups = data[group_var].dropna().unique()
    if len(groups) != 2:
        return None, "Exactly two groups required"
    group_data = [data[data[group_var] == grp][variable].dropna() for grp in groups]
    if any(len(d) < 3 for d in group_data):
        return None, "Minimum 3 samples per group required"
    t_stat, p_value = stats.ttest_ind(*group_data)
    return (t_stat, p_value), None

def run_anova(data, variable, group_var):
    if len(data[group_var].unique()) < 2:
        return None, "Minimum 2 groups required"
    try:
        model = ols(f'{variable} ~ C({group_var})', data=data).fit()
        return sm.stats.anova_lm(model, typ=2), None
    except Exception as e:
        return None, str(e)

def run_regression(data, x_var, y_var):
    try:
        data_clean = data[[x_var, y_var]].dropna()
        if len(data_clean) < 10: return None, "At least 10 samples required"
        if data_clean[x_var].nunique() == 1: return None, "Independent variable needs variation"
        X = sm.add_constant(data_clean[x_var])
        return sm.OLS(data_clean[y_var], X).fit(), None
    except Exception as e:
        return None, f"Regression failed: {str(e)}"

# --- App Setup ---
st.set_page_config(layout="wide", page_title="EcoMonitor", page_icon="🌿")
df = load_data()
geojson, _ = load_geojson()
param_groups = get_parameter_groups(df)

# --- Main App Structure ---
tabs = ["🌿 Green Cover", "📊 Dashboard", "🔗 Correlation", "📈 Temporal", "📉 Statistics"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(tabs)

# --- Tab1: Green Cover ---
with tab1:
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.title("Vegetation Dynamics Monitor")
        try:
            with open(VIDEO_PATH, 'rb') as f:
                st.video(f.read(), format="video/mp4", **VIDEO_CONFIG)
        except FileNotFoundError:
            st.error("Video file not found in root directory")
        except Exception as e:
            st.error(f"Video error: {str(e)}")

# --- Tab2: Climate Dashboard ---
with tab2:
    st.title("Climate Dashboard")
    
    with st.sidebar:
        groups = sorted(param_groups.keys())
        selected_group = st.selectbox("Parameter Group", groups, key="dashboard_group")
        date_range = st.date_input("Date Range", value=(
            df.Date.min().date(), df.Date.max().date()),
            key="dashboard_dates"
        )
    
    group_data = param_groups[selected_group]
    filtered_df = df[(df.Date >= pd.to_datetime(date_range[0])) & 
                    (df.Date <= pd.to_datetime(date_range[1]))]
    
    # Metrics
    cols = st.columns(3)
    metrics = [
        (group_data['Max'], 'MAX', filtered_df[group_data['Max']].max()),
        (group_data['Min'], 'MIN', filtered_df[group_data['Min']].min()),
        (group_data['Mean'], 'AVG', filtered_df[group_data['Mean']].mean())
    ]
    for col, (_, label, value) in zip(cols, metrics):
        with col:
            st.markdown(f'<div class="metric-container">{label}<h2>{value:.2f}</h2></div>', 
                       unsafe_allow_html=True)
    
    # Visualizations
    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(10, 4))
        for prefix in ['Max', 'Mean', 'Min']:
            sns.lineplot(data=filtered_df, x='Date', y=group_data[prefix], 
                        label=prefix, ax=ax)
        ax.set_title(f"{selected_group.replace('_', ' ').title()} Trends")
        st.pyplot(fig)
    
    with col2:
        try:
            current = filtered_df[group_data['Mean']].mean()
            norm = normalize_value(current, df[group_data['Mean']].min(), df[group_data['Mean']].max())
            fig, ax = plt.subplots(figsize=(6, 6))
            gpd.GeoDataFrame.from_features(geojson['features']).plot(
                ax=ax, facecolor=plt.get_cmap('viridis')(norm), edgecolor='black')
            ax.set_axis_off()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Map error: {str(e)}")

# --- Tab3: Correlation Analysis ---
# --- Tab3: Correlation Analysis ---
with tab3:
    st.title("Cross-Parameter Correlation")
    
    excluded = st.session_state.get("dashboard_group", "")
    available = [p for p in param_groups.keys() if p != excluded]
    selected = st.multiselect("Select parameters", available, default=available[:2], key="corr_params")
    
    if len(selected) >= 2:
        corr_vars = []
        for p in selected:
            # CORRECTED LINE BELOW
            corr_vars.extend([param_groups[p][t] for t in ['Max', 'Min', 'Mean'])
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            df[corr_vars].corr(),
            annot=True, fmt=".2f", cmap="coolwarm", 
            xticklabels=[f"{p}\n({t})" for p in selected for t in ['Max', 'Min', 'Mean']],
            yticklabels=[f"{p}\n({t})" for p in selected for t in ['Max', 'Min', 'Mean']],
            ax=ax
        )
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
    else:
        st.warning("Select at least 2 parameters")

# --- Tab4: Temporal Analysis ---
with tab4:
    st.title("Temporal Analysis")
    group = st.selectbox("Parameter Group", sorted(param_groups.keys()), key="temporal_group")
    window = st.slider("Rolling Window (days)", 1, 90, 7, key="temporal_window")
    
    ts_data = df.set_index('Date')[[param_groups[group][t] for t in ['Max', 'Mean', 'Min']]
    ts_roll = ts_data.rolling(window).mean()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in ts_roll.columns:
        sns.lineplot(x=ts_roll.index, y=ts_roll[col], label=col.split('_')[0], ax=ax)
    ax.fill_between(ts_roll.index, ts_roll.iloc[:,1], ts_roll.iloc[:,2], alpha=0.1)
    ax.set_title(f"{window}-Day Rolling Average")
    st.pyplot(fig)

# --- Tab5: Statistics ---
with tab5:
    st.title("Statistical Analysis")
    test_type = st.selectbox("Analysis Type", ["T-Test", "ANOVA", "Regression"], key="stats_type")
    
    if test_type == "T-Test":
        col1, col2 = st.columns(2)
        with col1:
            var = st.selectbox("Variable", df.select_dtypes(include=np.number).columns, key="ttest_var")
        with col2:
            group_var = st.selectbox("Grouping Variable", 
                                   [c for c in df.columns if df[c].nunique() == 2], 
                                   key="ttest_group")
        if st.button("Run T-Test", key="ttest_btn"):
            result, err = run_ttest(df, var, group_var)
            if err:
                st.error(err)
            else:
                st.metric("t-statistic", f"{result[0]:.2f}")
                st.metric("p-value", f"{result[1]:.4f}")
    
    elif test_type == "ANOVA":
        var = st.selectbox("Variable", df.select_dtypes(include=np.number).columns, key="anova_var")
        group_var = st.selectbox("Grouping Variable", df.columns, key="anova_group")
        if st.button("Run ANOVA", key="anova_btn"):
            result, err = run_anova(df, var, group_var)
            if err:
                st.error(err)
            else:
                st.dataframe(result.style.format("{:.4f}"))
    
    elif test_type == "Regression":
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("Independent Variable", df.columns, key="reg_x")
        with col2:
            y_var = st.selectbox("Dependent Variable", df.columns, key="reg_y")
        if st.button("Run Regression", key="reg_btn"):
            model, err = run_regression(df, x_var, y_var)
            if err:
                st.error(err)
            else:
                st.write(f"**R²:** {model.rsquared:.2f}")
                st.write(f"**Coefficient:** {model.params[1]:.2f}")
                st.write(f"**p-value:** {model.pvalues[1]:.4f}")

# --- Footer ---
st.markdown("---")
st.caption(f"EcoMonitor • Data through {df.Date.max().strftime('%Y-%m-%d')}")
