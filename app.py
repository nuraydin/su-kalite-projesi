```python
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
    st.error("scikit-learn kütüphanesi yüklü değil. Lütfen 'pip install scikit-learn==1.5.2' komutunu çalıştırın.")
    RandomForestRegressor = None

# Preload Matplotlib font cache to avoid delays
fm._load_fontmanager(try_read_cache=True)

st.set_page_config(page_title="Su Kalite Testi", layout="wide")

CSV_FILE = "su_kalite_standartlari.txt"
LOGO_PATH = "mar_logo.png"

@st.cache_data
def fetch_limits():
    if not os.path.exists(CSV_FILE):
        st.error(f"{CSV_FILE} dosyası bulunamadı!")
        return pd.DataFrame()
    try:
        df = pd.read_csv(CSV_FILE)
        df.rename(columns={df.columns[0]: "Parametre",
                           df.columns[1]: "TSE",
                           df.columns[2]: "EC",
                           df.columns[3]: "WHO"}, inplace=True)
        df = df.dropna(subset=["Parametre"]).reset_index(drop=True)

        drop_keywords = ["Kabul Edilebilir", "STANDARTLAR", "Fiziksel ve Duyusal",
                         "EMS/100", "Organoleptik", "Renk", "Bulanıklık", "Koku", "Tat",
                         "Siyanür (CN)", "Selenyum (Se)", "Antimon (Sb)",
                         "C.perfringers", "Pseudomonas Aeruginosa"]
        mask = ~df["Parametre"].str.contains("|".join(drop_keywords), na=False, regex=False)
        df = df[mask].reset_index(drop=True)
        logger.debug(f"Loaded {len(df)} parameters from {CSV_FILE}")
        return df
    except Exception as e:
        st.error(f"{CSV_FILE} okunurken hata oluştu: {str(e)}")
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
        return "Veri Yok"
    low, high = limit_range
    if low is None or high is None:
        return "Veri Yok"
    if low <= value <= high:
        if (value - low) < 0.05 * (high - low) or (high - value) < 0.05 * (high - low):
            return "Sınırda"
        return "Uygun"
    return "Uygun Değil"

def create_results(df, column_name, input_values):
    results = []
    for _, row in df.iterrows():
        param = row["Parametre"]
        user_val = input_values.get(param)
        limit_range = parse_range(row[column_name])
        durum = judge(user_val, limit_range)
        results.append({"Parametre": param, "Değer": user_val, "Durum": durum})
    return pd.DataFrame(results)

def color_code(val):
    if val == "Uygun":
        return "background-color: #d4edda"
    elif val == "Sınırda":
        return "background-color: #fff3cd"
    elif val == "Uygun Değil":
        return "background-color: #f8d7da"
    else:
        return ""

def generate_pdf(tse_df, ec_df, who_df, ai_df=None):
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        for title, df in [("TSE Sonuçları", tse_df), ("EC Sonuçları", ec_df), ("WHO Sonuçları", who_df)] + ([("AI Tahminleri", ai_df)] if ai_df is not None else []):
            fig, ax = plt.subplots(figsize=(12, 9))  
            ax.axis('off')

            try:
                if os.path.exists(LOGO_PATH):
                    logo = Image.open(LOGO_PATH)
                    fig.figimage(logo, xo=40, yo=fig.bbox.ymax - 100, zoom=0.15)
            except:
                pass

            fig.text(0.5, 0.95, "💧 İÇME SUYU KALİTE RAPORU", fontsize=20, ha="center", weight='bold')
            fig.text(0.5, 0.91, title, fontsize=16, ha="center", weight='bold')

            table = ax.table(cellText=df.values,
                             colLabels=df.columns,
                             cellLoc='center',
                             loc='center',
                             colWidths=[0.35, 0.2, 0.25] if title != "AI Tahminleri" else [0.35, 0.2, 0.2, 0.25])

            table.auto_set_font_size(False)
            table.set_fontsize(11)

            for key, cell in table.get_celld().items():
                cell.set_linewidth(0.5)
                if key[0] == 0:
                    cell.set_fontsize(12)
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor("#f2f2f2")
                else:
                    durum = df.iloc[key[0]-1, -1]
                    if durum == "Uygun":
                        cell.set_facecolor("#d4edda")
                    elif durum == "Sınırda":
                        cell.set_facecolor("#fff3cd")
                    elif durum == "Uygun Değil":
                        cell.set_facecolor("#f8d7da")

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    buf.seek(0)
    return buf

def generate_ai_comment(tse_df):
    try:
        if tse_df.empty:
            return "TSE verileri boş. Analiz yapılamadı."
        non_compliant = len(tse_df[tse_df["Durum"] == "Uygun Değil"])
        borderline = len(tse_df[tse_df["Durum"] == "Sınırda"])
        if non_compliant > 0:
            return f"Su kalitesi TSE standartlarına göre {non_compliant} parametrede uygun değil. Derhal önlem alınması önerilir."
        elif borderline > 0:
            return f"Su kalitesi genel olarak uygun, ancak {borderline} parametrede sınırda değerler tespit edildi. Dikkatli izleme önerilir."
        else:
            return "Su kalitesi TSE standartlarına göre tamamen uygun. Herhangi bir sorun tespit edilmedi."
    except Exception as e:
        return f"AI yorumu oluşturulurken hata: {str(e)}"

def random_forest_prediction(input_values, df_limits):
    if RandomForestRegressor is None:
        return pd.DataFrame(), 0.0, "Random Forest modeli yüklü değil: scikit-learn eksik."

    try:
        # Log parameters and ranges
        logger.debug(f"Parameters: {df_limits['Parametre'].tolist()}")
        for _, row in df_limits.iterrows():
            logger.debug(f"Param: {row['Parametre']}, TSE Range: {row['TSE']}, Parsed: {parse_range(row['TSE'])}")

        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 100
        X_train = []
        y_train = []
        params = df_limits["Parametre"]
        
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
            tse_range = df_limits[df_limits["Parametre"] == param]["TSE"].iloc[0]
            results.append({
                "Parametre": param,
                "Değer": val,
                "Tahmini Değer": val,  # Placeholder
                "Durum": judge(val, parse_range(tse_range))
            })
        return pd.DataFrame(results), prediction, None
    except Exception as e:
        logger.error(f"Random Forest error: {str(e)}")
        return pd.DataFrame(), 0.0, f"Random Forest tahmini sırasında hata: {str(e)}"

def create_comparison_chart(df_limits, input_values):
    try:
        # Select top 5 valid parameters
        params = []
        tse_limits = []
        user_values = []
        for _, row in df_limits.iloc[:5].iterrows():
            param = row["Parametre"]
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
        ax.bar([i - bar_width/2 for i in x], user_values, bar_width, label="Kullanıcı Değerleri", color="#36A2EB")
        ax.bar([i + bar_width/2 for i in x], tse_limits, bar_width, label="TSE Üst Sınır", color="#FF6384")
        
        ax.set_xlabel("Parametreler")
        ax.set_ylabel("Değerler")
        ax.set_title("Kullanıcı Değerleri vs TSE Üst Sınırları")
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
        return None, f"Grafik oluşturulurken hata: {str(e)}"

st.title("💧 İçme Suyu Kalite Testi")
st.caption("📌 Lütfen sadece sayısal değer giriniz. Boş bırakabilirsiniz.")

df_limits = fetch_limits()
input_values = {}

cols = st.columns(2)
for idx, row in df_limits.iterrows():
    param = row["Parametre"]
    with cols[idx % 2]:
        input_values[param] = st.number_input(param, format="%.4f", key=param)

if st.button("Generate Report"):
    df_tse = create_results(df_limits, "TSE", input_values)
    df_ec = create_results(df_limits, "EC", input_values)
    df_who = create_results(df_limits, "WHO", input_values)
    
    # AI Analysis
    ai_df, quality_score, rf_error = random_forest_prediction(input_values, df_limits)
    ai_comment = generate_ai_comment(df_tse) if not rf_error else "AI yorumu oluşturulamadı."
    chart_buf, chart_error = create_comparison_chart(df_limits, input_values)

    tabs = st.tabs(["📘 TSE", "📗 EC", "📕 WO", "🤖 AI"])
    with tabs[0]:
        st.subheader("TSE Results")
        st.dataframe(df_tse.style.map(color_code, subset=["Durum"]))
    with tabs[1]:
        st.subheader("EC Results")
        st.dataframe(df_ec.style.map(color_code, subset=["Durum"]))
    with tabs[2]:
        st.subheader("WHO Results")
        st.dataframe(df_who.style.map(color_code, subset=["Durum"]))
    with tabs[3]:
        st.subheader("AI Analysis")
        # Display input values for debugging
        st.write("**Girilen Değerler:**")
        input_summary = {k: v for k, v in input_values.items() if v != 0}
        st.write(input_summary if input_summary else "Hiçbir değer girilmedi.")
        if rf_error:
            st.error(rf_error)
        else:
            st.write(f"**Su Kalite Skoru (Random Forest Tahmini):** {quality_score:.2f}/100")
            st.write(f"**AI Yorumu:** {ai_comment}")
            st.dataframe(ai_df.style.map(color_code, subset=["Durum"]))
        if chart_error:
            st.error(chart_error)
        elif chart_buf:
            st.image(chart_buf, caption="Kullanıcı Değerleri vs TSE Üst Sınırları (İlk 5 Parametre)")

    st.markdown("---")
    st.success("Rapor hazır! PDF formatında indirebilirsin.")

    pdf_file = generate_pdf(df_tse, df_ec, df_who, ai_df if not rf_error else None)
    st.download_button("📥 PDF Raporu İndir", data=pdf_file,
                       file_name="su_kalite_raporu.pdf",
                       mime="application/pdf")
```
