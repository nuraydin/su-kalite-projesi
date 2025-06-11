import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
from PIL import Image
import numpy as np
import os
import matplotlib.font_manager as fm

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
        mask = ~df["Parametre"].str.contains("|".join(drop_keywords), na=False)
        df = df[mask].reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"{CSV_FILE} okunurken hata oluştu: {str(e)}")
        return pd.DataFrame()

def parse_range(r):
    if pd.isna(r) or r == "":
        return (None, None)
    r = str(r).strip()
    if "-" in r:
        low, high = r.split("-")
        return (float(low.replace(",", ".")), float(high.replace(",", ".")))
    try:
        val = float(r.replace(",", "."))
        return (0.0, val)
    except:
        return (None, None)

def judge(value, limit_range):
    if value is None or limit_range == (None, None):
        return "Veri Yok"
    low, high = limit_range
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
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 100
        X_train = []
        y_train = []
        params = df_limits["Parametre"].tolist()
        
        for _ in range(n_samples):
            sample = []
            for param, tse_range in zip(df_limits["Parametre"], df_limits["TSE"]):
                low, high = parse_range(tse_range)
                if low is None or high is None:
                    sample.append(0)
                else:
                    sample.append(np.random.uniform(max(0, low * 0.5), high * 1.5))
            X_train.append(sample)
            # Quality score: higher if within TSE limits, lower if outside
            score = 100
            for val, tse_range in zip(sample, df_limits["TSE"]):
                low, high = parse_range(tse_range)
                if low is not None and high is not None:
                    if val < low or val > high:
                        score -= 20
                    elif (val - low) < 0.1 * (high - low) or (high - val) < 0.1 * (high - low):
                        score -= 5
            y_train.append(max(0, score))
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Prepare input for prediction
        input_vector = [input_values.get(param, 0) for param in params]
        prediction = model.predict([input_vector])[0]
        
        # Create results DataFrame
        results = []
        for param, val in zip(params, input_vector):
            results.append({
                "Parametre": param,
                "Değer": val,
                "Tahmini Değer": val,  # Placeholder, as RF predicts quality score
                "Durum": judge(val, parse_range(df_limits[df_limits["Parametre"] == param]["TSE"].iloc[0]))
            })
        return pd.DataFrame(results), prediction, None
    except Exception as e:
        return pd.DataFrame(), 0.0, f"Random Forest tahmini sırasında hata: {str(e)}"

def create_comparison_chart(df_limits, input_values):
    try:
        # Select top 5 parameters for readability
        params = df_limits["Parametre"].tolist()[:5]
        user_values = [input_values.get(p, 0) for p in params]
        tse_limits = [parse_range(row["TSE"])[1] if parse_range(row["TSE"])[1] is not None else 0 for _, row in df_limits.iloc[:5].iterrows()]
        
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
        return None, f"Grafik oluşturulurken hata: {str(e)}
