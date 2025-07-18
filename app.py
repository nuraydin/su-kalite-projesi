import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
from PIL import Image
import numpy as np
import os
import matplotlib.font_manager as fm
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Attempt to import scikit-learn
try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError:
    st.error("scikit-learn library is not installed. Please run 'pip install scikit-learn==1.5.2'.")
    RandomForestRegressor = None

# Preload Matplotlib font cache to avoid delays
fm._load_fontmanager(try_read_cache=True)

st.set_page_config(page_title="Water Quality Test", layout="wide")

CSV_FILE = "water_quality_standards.txt"
LOGO_PATH = "mar_logo.png"

@st.cache_data
def fetch_limits():
    if not os.path.exists(CSV_FILE):
        st.error(f"{CSV_FILE} file not found!")
        return pd.DataFrame()
    try:
        df = pd.read_csv(CSV_FILE)
        df.rename(columns={df.columns[0]: "Parameter",
                           df.columns[1]: "TSE",
                           df.columns[2]: "EC",
                           df.columns[3]: "WHO"}, inplace=True)
        df = df.dropna(subset=["Parameter"]).reset_index(drop=True)

        drop_keywords = ["Acceptable", "STANDARDS", "Physical and Sensory",
                         "EMS/100", "Organoleptic", "Color", "Turbidity", "Odor", "Taste",
                         "Cyanide (CN)", "Selenium (Se)", "Antimony (Sb)",
                         "C.perfringers", "Pseudomonas Aeruginosa"]
        mask = ~df["Parameter"].str.contains("|".join(drop_keywords), na=False, regex=False)
        df = df[mask].reset_index(drop=True)
        logger.debug(f"Loaded {len(df)} parameters from {CSV_FILE}")
        return df
    except Exception as e:
        st.error(f"Error reading {CSV_FILE}: {str(e)}")
        return pd.DataFrame()

def parse_range(r):
    if pd.isna(r) or r == "":
        return (None, None)
    r = str(r).strip()
    try:
        if "-" in r:
            low, high = r.split("-")
            low = float(low.replace(",", "."))
            high = float(high.replace(",", "."))
            return (low, high) if low <= high else (None, None)
        val = float(r.replace(",", "."))
        return (0.0, val)
    except:
        return (None, None)

def judge(value, limit_range):
    if value is None or limit_range == (None, None):
        return "No Data"
    low, high = limit_range
    if low is None or high is None:
        return "No Data"
    if low <= value <= high:
        if (value - low) < 0.05 * (high - low) or (high - value) < 0.05 * (high - low):
            return "Borderline"
        return "Compliant"
    return "Non-Compliant"

def create_results(df, column_name, input_values):
    results = []
    for _, row in df.iterrows():
        param = row["Parameter"]
        user_val = input_values.get(param)
        limit_range = parse_range(row[column_name])
        status = judge(user_val, limit_range)
        results.append({"Parameter": param, "Value": user_val, "Status": status})
    return pd.DataFrame(results)

def color_code(val):
    if val == "Compliant":
        return "background-color: #d4edda"
    elif val == "Borderline":
        return "background-color: #fff3cd"
    elif val == "Non-Compliant":
        return "background-color: #f8d7da"
    else:
        return ""

def generate_pdf(tse_df, ec_df, who_df, ai_df=None):
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        for title, df in [("TSE Results", tse_df), ("EC Results", ec_df), ("WHO Results", who_df)] + ([("AI Predictions", ai_df)] if ai_df is not None else []):
            fig, ax = plt.subplots(figsize=(12, 9))  
            ax.axis('off')

            try:
                if os.path.exists(LOGO_PATH):
                    logo = Image.open(LOGO_PATH)
                    fig.figimage(logo, xo=40, yo=fig.bbox.ymax - 100, zoom=0.15)
            except:
                pass

            fig.text(0.5, 0.95, "💧 DRINKING WATER QUALITY REPORT", fontsize=20, ha="center", weight='bold')
            fig.text(0.5, 0.91, title, fontsize=16, ha="center", weight='bold')

            table = ax.table(cellText=df.values,
                             colLabels=df.columns,
                             cellLoc='center',
                             loc='center',
                             colWidths=[0.35, 0.2, 0.25] if title != "AI Predictions" else [0.35, 0.2, 0.2, 0.25])

            table.auto_set_font_size(False)
            table.set_fontsize(11)

            for key, cell in table.get_celld().items():
                cell.set_linewidth(0.5)
                if key[0] == 0:
                    cell.set_fontsize(12)
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor("#f2f2f2")
                else:
                    status = df.iloc[key[0]-1, -1]
                    if status == "Compliant":
                        cell.set_facecolor("#d4edda")
                    elif status == "Borderline":
                        cell.set_facecolor("#fff3cd")
                    elif status == "Non-Compliant":
                        cell.set_facecolor("#f8d7da")

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    buf.seek(0)
    return buf

def generate_ai_comment(tse_df):
    try:
        if tse_df.empty:
            return "TSE data is empty. Analysis could not be performed."
        non_compliant = len(tse_df[tse_df["Status"] == "Non-Compliant"])
        borderline = len(tse_df[tse_df["Status"] == "Borderline"])
        if non_compliant > 0:
            return f"Water quality is non-compliant with TSE standards in {non_compliant} parameters. Immediate action is recommended."
        elif borderline > 0:
            return f"Water quality is generally compliant, but {borderline} parameters are borderline. Careful monitoring is recommended."
        else:
            return "Water quality is fully compliant with TSE standards. No issues detected."
    except Exception as e:
        return f"Error generating AI comment: {str(e)}"

def random_forest_prediction(input_values, df_limits):
    if RandomForestRegressor is None:
        return pd.DataFrame(), 0.0, "Random Forest model not loaded: scikit-learn missing."

    try:
        # Log parameters and ranges
        logger.debug(f"Parameters: {df_limits['Parameter'].tolist()}")
        for _, row in df_limits.iterrows():
            logger.debug(f"Param: {row['Parameter']}, TSE Range: {row['TSE']}, Parsed: {parse_range(row['TSE'])}")

        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 100
        X_train = []
        y_train = []
        params = df_limits["Parameter"]
        
        for i in range(n_samples):
            sample = []
            for param, tse_range in zip(params, df_limits["TSE"]):
                low, high = parse_range(tse_range)
                if low is None or high is None:
                    sample.append(0)
                else:
                    sample.append(np.random.uniform(max(0, low * 0.9), high * 1.1))  # Narrower range
            X_train.append(sample)
            # Score: start at 100, deduct penalties
            score = 100
            penalties = []
            for val, param, tse_range in zip(sample, params, df_limits["TSE"]):
                low, high = parse_range(tse_range)
                if low is None or high is None:
                    continue
                if val < low or val > high:
                    penalties.append((param, val, "outside", -10))  # Reduced penalty
                    score -= 10
                elif (val - low) < 0.1 * (high - low) or (high - val) < 0.1 * (high - low):
                    penalties.append((param, val, "borderline", -2))  # Reduced penalty
                    score -= 2
            y_train.append(max(0, score))
            if i < 2:  # Log first two samples
                logger.debug(f"Sample {i}: Values={sample[:5]}, Penalties={penalties[:5]}, Score={score}")

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Prepare input
        input_vector = [input_values.get(param, 0) for param in params]
        logger.debug(f"Input vector: {input_vector}")
        prediction = model.predict([input_vector])[0]
        logger.debug(f"Predicted score: {prediction}")
        
        # Results DataFrame
        results = []
        for param, val in zip(params, input_vector):
            tse_range = df_limits[df_limits["Parameter"] == param]["TSE"].iloc[0]
            results.append({
                "Parameter": param,
                "Value": val,
                "Predicted Value": val,  # Placeholder
                "Status": judge(val, parse_range(tse_range))
            })
        return pd.DataFrame(results), prediction, None
    except Exception as e:
        logger.error(f"Random Forest error: {str(e)}")
        return pd.DataFrame(), 0.0, f"Error during Random Forest prediction: {str(e)}"

def create_comparison_chart(df_limits, input_values):
    try:
        # Select top 5 valid parameters
        params = []
        tse_limits = []
        user_values = []
        for _, row in df_limits.iloc[:5].iterrows():
            param = row["Parameter"]
            tse_range = parse_range(row["TSE"])
            if tse_range[1] is not None:
                params.append(param)
                tse_limits.append(float(tse_range[1]))
                user_values.append(float(input_values.get(param, 0)))
        
        if not params:
            return None, "No valid parameters for chart."
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.35
        x = range(len(params))
        ax.bar([i - bar_width/2 for i in x], user_values, bar_width, label="User Values", color="#36A2EB")
        ax.bar([i + bar_width/2 for i in x], tse_limits, bar_width, label="TSE Upper Limit", color="#FF6384")
        
        ax.set_xlabel("Parameters")
        ax.set_ylabel("Values")
        ax.set_title("User Values vs TSE Upper Limits")
        ax.set_xticks(x)
        ax.set_xticklabels(params, rotation=45)
        ax.legend()
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return buf, None
    except Exception as e:
        return None, f"Error generating chart: {str(e)}"

st.title("💧 Drinking Water Quality Test")
st.caption("📌 Please enter only numerical values. You can leave fields blank.")

df_limits = fetch_limits()
input_values = {}

cols = st.columns(2)
for idx, row in df_limits.iterrows():
    param = row["Parameter"]
    with cols[idx % 2]:
        input_values[param] = st.number_input(param, format="%.4f", key=param)

if st.button("Generate Report"):
    df_tse = create_results(df_limits, "TSE", input_values)
    df_ec = create_results(df_limits, "EC", input_values)
    df_who = create_results(df_limits, "WHO", input_values)
    
    # AI Analysis
    ai_df, quality_score, rf_error = random_forest_prediction(input_values, df_limits)
    ai_comment = generate_ai_comment(df_tse) if not rf_error else "AI comment could not be generated."
    chart_buf, chart_error = create_comparison_chart(df_limits, input_values)

    tabs = st.tabs(["📘 TSE", "📗 EC", "📕 WHO", "🤖 AI"])
    with tabs[0]:
        st.subheader("TSE Results")
        st.dataframe(df_tse.style.map(color_code, subset=["Status"]))
    with tabs[1]:
        st.subheader("EC Results")
        st.dataframe(df_ec.style.map(color_code, subset=["Status"]))
    with tabs[2]:
        st.subheader("WHO Results")
        st.dataframe(df_who.style.map(color_code, subset=["Status"]))
    with tabs[3]:
        st.subheader("AI Analysis")
        # Display input values for debugging
        st.write("**Entered Values:**")
        input_summary = {k: v for k, v in input_values.items() if v != 0}
        st.write(input_summary if input_summary else "No values entered.")
        if rf_error:
            st.error(rf_error)
        else:
            st.write(f"**Water Quality Score (Random Forest Prediction):** {quality_score:.2f}/100")
            st.write(f"**AI Comment:** {ai_comment}")
            st.dataframe(ai_df.style.map(color_code, subset=["Status"]))
        if chart_error:
            st.error(chart_error)
        elif chart_buf:
            st.image(chart_buf, caption="User Values vs TSE Upper Limits (First 5 Parameters)")

    st.markdown("---")
    st.success("Report is ready! You can download it in PDF format.")

    pdf_file = generate_pdf(df_tse, df_ec, df_who, ai_df if not rf_error else None)
    st.download_button("📥 Download PDF Report", data=pdf_file,
                       file_name="water_quality_report.pdf",
                       mime="application/pdf")
