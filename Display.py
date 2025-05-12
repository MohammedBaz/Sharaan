import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Import seaborn
import geopandas as gpd
import requests
from scipy import stats
import statsmodels.api as sm # For OLS regression
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import io # Needed for saving plot to memory
from fpdf import FPDF # Import FPDF for PDF generation
from sklearn.linear_model import LinearRegression # Alternative for simple regression

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
         /* Add margin below download button */
         [data-testid="stSidebar"] .stDownloadButton {
             margin-top: 1rem; /* Adjusted margin */
         }
         [data-testid="stSidebar"] .stDownloadButton button {
             width: 100%; /* Make button full width */
             margin-bottom: 0.5rem; /* Space between buttons */
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

# --- PDF Generation Function ---
def create_dashboard_pdf(param_name, fig, stats_dict):
    """Generates a PDF report for the dashboard."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"EcoMonitor Dashboard Report: {param_name.replace('_', ' ').title()}", 0, 1, "C")
    pdf.ln(10)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Overall Statistics:", 0, 1)
    pdf.set_font("Helvetica", "", 11)
    max_stat = f"{stats_dict['max']:.2f}" if not pd.isna(stats_dict['max']) else "N/A"
    mean_stat = f"{stats_dict['mean']:.2f}" if not pd.isna(stats_dict['mean']) else "N/A"
    min_stat = f"{stats_dict['min']:.2f}" if not pd.isna(stats_dict['min']) else "N/A"
    pdf.cell(0, 8, f"  - Maximum: {max_stat}", 0, 1)
    pdf.cell(0, 8, f"  - Average: {mean_stat}", 0, 1)
    pdf.cell(0, 8, f"  - Minimum: {min_stat}", 0, 1)
    pdf.ln(10)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Trend Over Time Plot:", 0, 1)
    pdf.ln(5)
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    page_width = pdf.w - 2 * pdf.l_margin
    pdf.image(img_buffer, x=pdf.l_margin, w=page_width)
    img_buffer.close()

    pdf_output = pdf.output(dest='S')
    if isinstance(pdf_output, bytearray):
        return bytes(pdf_output)
    return pdf_output


# --- Load Data ---
df = load_data()
geojson, sharaan_boundary = load_geojson()
param_groups = get_parameter_groups(df)
if not param_groups:
    st.error("Error: Could not identify valid parameter groups.")
    st.stop()

# --- Sidebar Navigation ---
st.sidebar.title("EcoMonitor Navigation")
pages = ["Green Cover", "Dashboard", "Correlation", "Temporal", "Statistics", "🔮 Prediction"]
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

    if selected_group_key_dashboard:
        group_cols_info = param_groups[selected_group_key_dashboard]
        dashboard_df = df

        if not dashboard_df.empty:
            st.subheader("Trend Over Time")
            fig_line, ax_line = plt.subplots(figsize=(12, 5))
            plot_title = f"{selected_group_key_dashboard.replace('_', ' ').title()} Trend (Overall)"
            ax_line.set_title(plot_title, fontsize=14)
            for prefix in ['Max', 'Mean', 'Min']:
                if prefix in group_cols_info:
                    col_name = group_cols_info[prefix]
                    sns.lineplot(data=dashboard_df, x='Date', y=col_name, label=prefix, ax=ax_line, linestyle='-', linewidth=1.5)
            ax_line.set_ylabel(selected_group_key_dashboard.replace('_', ' '), fontsize=12)
            ax_line.set_xlabel("Date", fontsize=12)
            ax_line.legend(title="Statistic")
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            st.pyplot(fig_line, use_container_width=True)

            st.markdown("---", unsafe_allow_html=True)
            st.subheader(f"Key Statistics: {selected_group_key_dashboard.replace('_', ' ').title()}")
            metric_cols = st.columns(3)
            overall_max_val = dashboard_df[group_cols_info['Max']].max()
            overall_min_val = dashboard_df[group_cols_info['Min']].min()
            overall_mean_val = dashboard_df[group_cols_info['Mean']].mean()
            with metric_cols[0]: st.metric(label="Overall Maximum", value=f"{overall_max_val:.2f}")
            with metric_cols[1]: st.metric(label="Overall Average", value=f"{overall_mean_val:.2f}")
            with metric_cols[2]: st.metric(label="Overall Minimum", value=f"{overall_min_val:.2f}")

            st.markdown("---", unsafe_allow_html=True)
            stats_for_pdf = {'max': overall_max_val, 'mean': overall_mean_val, 'min': overall_min_val}
            try:
                if 'fig_line' in locals() and fig_line is not None:
                    pdf_bytes = create_dashboard_pdf(selected_group_key_dashboard, fig_line, stats_for_pdf)
                    st.download_button(
                        label="📄 Download Dashboard as PDF",
                        data=pdf_bytes,
                        file_name=f"dashboard_{selected_group_key_dashboard}.pdf",
                        mime="application/pdf",
                        key='download_pdf_dashboard'
                    )
                else: st.error("Plot figure not available for PDF generation.")
            except Exception as pdf_e: st.error(f"Failed to generate PDF: {pdf_e}")
            finally:
                 if 'fig_line' in locals() and fig_line is not None: plt.close(fig_line)
        else: st.warning("No data available for the selected parameter group.")
    else: st.warning("Please select a parameter group using the control above.")


# --- Page 3: Correlation Analysis ---
elif selected_page == "Correlation":
    st.title("🔗 Cross-Parameter Correlation Analysis")
    excluded_group = st.session_state.get("dashboard_group_select_main", None)
    available_groups = sorted([p for p in param_groups.keys() if p != excluded_group])
    if len(available_groups) >= 2:
        selected_corr_groups = st.multiselect("Select parameters to correlate (select at least 2)", available_groups, default=available_groups[:min(len(available_groups), 3)], key="correlation_params_select_main")
        st.markdown("---", unsafe_allow_html=True)
        if len(selected_corr_groups) >= 2:
            corr_vars_cols = []; corr_labels = []; valid_selection = True
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
                plt.tight_layout(pad=2.0); st.pyplot(fig_corr); plt.close(fig_corr)
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
                    ax_temporal.legend(loc='best'); plt.tight_layout(); st.pyplot(fig_temporal, use_container_width=True); plt.close(fig_temporal)
                else: st.warning(f"Missing required columns for '{temporal_group_key}'.")
        else: st.warning("Please select a parameter group.")
    else: st.warning("No parameter groups available.")

# --- Page 5: Statistics ---
elif selected_page == "Statistics":
    st.title("📉 Data Distribution Overview")

    st.subheader("Box Plots for All Numerical Variables")
    st.markdown("Box plots visually summarize the distribution of each numerical variable, showing the median, quartiles, and potential outliers.")
    st.markdown("---", unsafe_allow_html=True)

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    
    if not numeric_columns:
        st.warning("No numeric columns found in the dataset to generate box plots.")
    else:
        num_plot_cols = min(len(numeric_columns), 2)
        plot_cols = st.columns(num_plot_cols)
        col_idx = 0

        for i, col_name in enumerate(numeric_columns):
            if col_name == 'Date': continue

            with plot_cols[col_idx]:
                try:
                    fig_box, ax_box = plt.subplots(figsize=(6, 4))
                    sns.boxplot(y=df[col_name], ax=ax_box, width=0.5)
                    ax_box.set_title(f"Distribution of {col_name.replace('_', ' ').title()}", fontsize=12)
                    ax_box.set_ylabel("")
                    plt.tight_layout()
                    st.pyplot(fig_box)
                    plt.close(fig_box)
                except Exception as e:
                    st.error(f"Error generating box plot for {col_name}: {e}")
            
            col_idx = (col_idx + 1) % num_plot_cols
            if col_idx == 0 and i < len(numeric_columns) -1 :
                 st.markdown("---", unsafe_allow_html=True)

# --- Page 6: Prediction ---
elif selected_page == "🔮 Prediction":
    st.title("🔮 Simple Time Series Prediction")
    st.markdown("""
    This section provides a basic forecast using linear regression. 
    It fits a line to the historical data based on time and extends this line into the future.
    **Note:** This is a very simple model. Its accuracy is limited, especially if the data has complex patterns (like seasonality) or no clear linear trend. Interpret with caution.
    """)
    st.markdown("---", unsafe_allow_html=True)

    # **MODIFICATION:** Set a fixed prediction horizon
    FIXED_DAYS_TO_PREDICT = 30 

    if param_groups:
        pred_param_group_key = st.selectbox(
            "Select Parameter Group to Predict",
            sorted(param_groups.keys()),
            key="prediction_param_group_select"
        )
        
        if pred_param_group_key and 'Mean' in param_groups[pred_param_group_key]:
            target_variable = param_groups[pred_param_group_key]['Mean']

            # **REMOVED:** days_to_predict number input

            if st.button("Generate Prediction Plot", key="prediction_run_button"): # Changed button label
                if target_variable not in df.columns:
                    st.error(f"Target variable '{target_variable}' not found in the dataset.")
                else:
                    try:
                        predict_df = df[['Date', target_variable]].copy().dropna()
                        if len(predict_df) < 2:
                             st.error(f"Not enough data points for '{target_variable}' to make a prediction.")
                        else:
                            predict_df['Time_Step'] = (predict_df['Date'] - predict_df['Date'].min()).dt.days
                            
                            X_train = predict_df[['Time_Step']]
                            y_train = predict_df[target_variable]

                            model = LinearRegression()
                            model.fit(X_train, y_train)
                            
                            last_time_step = predict_df['Time_Step'].max()
                            last_date = predict_df['Date'].max()
                            
                            # Use FIXED_DAYS_TO_PREDICT
                            future_time_steps = np.array([last_time_step + i + 1 for i in range(FIXED_DAYS_TO_PREDICT)]).reshape(-1, 1)
                            future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(FIXED_DAYS_TO_PREDICT)]
                            
                            future_predictions = model.predict(future_time_steps)
                            
                            predictions_df = pd.DataFrame({
                                'Date': future_dates,
                                f'Predicted_{target_variable}': future_predictions # More specific column name
                            })

                            st.subheader("Prediction Plot")
                            fig_pred, ax_pred = plt.subplots(figsize=(12, 6))
                            
                            sns.lineplot(x='Date', y=target_variable, data=predict_df, ax=ax_pred, label='Historical Data', linestyle='-', linewidth=1.5, color='blue')
                            
                            historical_fit = model.predict(X_train)
                            ax_pred.plot(predict_df['Date'], historical_fit, color='red', linestyle='--', label='Fitted Regression Line')

                            # Plot forecasted data
                            ax_pred.plot(predictions_df['Date'], predictions_df[f'Predicted_{target_variable}'], color='green', linestyle='-', linewidth=2, label=f'Forecast ({FIXED_DAYS_TO_PREDICT} days)')
                            
                            ax_pred.set_title(f"Linear Trend & Forecast for {target_variable.replace('_', ' ').title()}", fontsize=14)
                            ax_pred.set_xlabel("Date", fontsize=12)
                            ax_pred.set_ylabel(target_variable.replace('_', ' ').title(), fontsize=12)
                            ax_pred.legend()
                            plt.xticks(rotation=30, ha='right')
                            plt.tight_layout()
                            st.pyplot(fig_pred)
                            plt.close(fig_pred)

                            # Optionally, display the predicted values in a table
                            st.subheader(f"Forecasted Values for the Next {FIXED_DAYS_TO_PREDICT} Days")
                            st.dataframe(predictions_df[['Date', f'Predicted_{target_variable}']].style.format({f'Predicted_{target_variable}': "{:.2f}", 'Date': '{:%Y-%m-%d}'}))


                    except Exception as e:
                        st.error(f"An error occurred during prediction: {e}")
        else:
            st.warning("Selected parameter group does not have a 'Mean' column or is invalid.")
    else:
        st.warning("No parameter groups available for prediction.")


# --- Footer ---
if selected_page:
    st.markdown("---", unsafe_allow_html=True)
    st.caption(f"EcoMonitor Dashboard | Data sourced from specified URLs | Last data point: {df['Date'].max().strftime('%Y-%m-%d')}")

