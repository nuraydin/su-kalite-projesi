import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
from PIL import Image
import numpy as np
import os
import matplotlib.font_manager as fm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Preload Matplotlib font cache
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

def generate_synthetic_data(df_limits):
    try:
        np.random.seed(42)
        n_samples = 1000
        params = df_limits["Parametre"].tolist()
        data = []
        labels = []
        for _ in range(n_samples):
            sample = {}
            score = 100
            for param, tse_range in zip(params, df_limits["TSE"]):
                low, high = parse_range(tse_range)
                if low is None or high is None:
                    val = 0
                else:
                    val = np.random.uniform(max(0, low * 0.5), high * 1.5)
                sample[param] = val
                if low is not None and high is not None:
                    if val < low or val > high:
                        score -= 20
                    elif (val - low) < 0.1 * (high - low) or (high - val) < 0.1 * (high - low):
                        score -= 5
            sample["etiket"] = 1 if score >= 80 else 0
            data.append(sample)
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Sintetik veri oluşturulurken hata: {str(e)}")
        return pd.DataFrame()

@st.cache_resource
def train_rf_model(df_limits):
    try:
        df_synthetic = generate_synthetic_data(df_limits)
        X = df_synthetic.drop("etiket", axis=1)
        y = df_synthetic["etiket"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        return model, scaler, accuracy, X.columns
    except Exception as e:
        st.error(f"Random Forest modeli eğitimi sırasında hata: {str(e)}")
        return None, None, 0.0, []

@st.cache_resource
def train_ann_model(df_limits):
    try:
        df_synthetic = generate_synthetic_data(df_limits)
        X = df_synthetic.drop("etiket", axis=1)
        y = df_synthetic["etiket"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        model = Sequential()
        model.add(Dense(units=16, activation='relu', input_dim=X_train.shape[1]))
        model.add(Dense(units=8, activation='relu'))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        return model, scaler, accuracy, X.columns
    except Exception as e:
        st.error(f"ANN modeli eğitimi sırasında hata: {str(e)}")
        return None, None, 0.0, []

def generate_pdf(tse_df, ec_df, who_df, rf_prediction, ann_prediction, feature_importance):
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        for title, df in [("TSE Sonuçları", tse_df), ("EC Sonuçları", ec_df), ("WHO Sonuçları", who_df)]:
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
            table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', colWidths=[0.35, 0.2, 0.25])
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
        # AI Results
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.axis('off')
        fig.text(0.5, 0.95, "💧 İÇME SUYU KALİTE RAPORU", fontsize=20, ha="center", weight='bold')
        fig.text(0.5, 0.91, "Yapay Zeka Tahminleri", fontsize=16, ha="center", weight='bold')
        text = (f"Random Forest Tahmini: {'Evet' if rf_prediction == 1 else 'Hayır'}\n"
                f"ANN Tahmini: {'Evet' if ann_prediction == 1 else 'Hayır'}\n"
                f"Özellik Önem Sıralaması:\n{feature_importance.sort_values(ascending=False).head().to_string()}")
        ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=12)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    buf.seek(0)
    return buf

def create_comparison_chart(df_limits, input_values):
    try:
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
        return None, f"Grafik oluşturulurken hata: {str(e)}"

st.title("💧 İçme Suyu Kalite Testi")
st.caption("📌 Lütfen sadece sayısal değer giriniz. Boş bırakabilirsiniz.")

df_limits = fetch_limits()
input_values = {}

cols = st.columns(4)
for idx, row in df_limits.iterrows():
    param = row["Parametre"]
    with cols[idx % 4]:
        input_values[param] = st.number_input(param, format="%.4f", key=param)

# Train models
rf_model, rf_scaler, rf_accuracy, feature_names = train_rf_model(df_limits)
ann_model, ann_scaler, ann_accuracy, _ = train_ann_model(df_limits)

if st.button("💡 Hesapla"):
    if not any(input_values.values()):
        st.warning("Lütfen en az bir parametre için değer girin.")
    else:
        df_tse = create_results(df_limits, "TSE", input_values)
        df_ec = create_results(df_limits, "EC", input_values)
        df_who = create_results(df_limits, "WHO", input_values)
        
        # AI Predictions
        input_data = pd.DataFrame([input_values])
        input_data = input_data.reindex(columns=feature_names, fill_value=0)
        
        # Random Forest
        if rf_model is not None:
            input_data_scaled_rf = rf_scaler.transform(input_data)
            rf_prediction = rf_model.predict(input_data_scaled_rf)[0]
            feature_importance = pd.Series(rf_model.feature_importances_, index=feature_names)
        else:
            rf_prediction = None
            feature_importance = pd.Series()
            st.error("Random Forest modeli yüklenemedi.")
        
        # ANN
        if ann_model is not None:
            input_data_scaled_ann = ann_scaler.transform(input_data)
            ann_prediction = (ann_model.predict(input_data_scaled_ann, verbose=0) > 0.5).astype(int)[0][0]
        else:
            ann_prediction = None
            st.error("ANN modeli yüklenemedi.")
        
        chart_buf, chart_error = create_comparison_chart(df_limits, input_values)
        
        tabs = st.tabs(["📘 TSE", "📗 EC", "📕 WHO", "🤖 Yapay Zeka"])
        with tabs[0]:
            st.subheader("TSE Sonuçları")
            st.dataframe(df_tse.style.map(color_code, subset=["Durum"]), use_container_width=True)
        with tabs[1]:
            st.subheader("EC Sonuçları")
            st.dataframe(df_ec.style.map(color_code, subset=["Durum"]), use_container_width=True)
        with tabs[2]:
            st.subheader("WHO Sonuçları")
            st.dataframe(df_who.style.map(color_code, subset=["Durum"]), use_container_width=True)
        with tabs[3]:
            st.subheader("Yapay Zeka Tahmini")
            if rf_prediction is not None:
                st.write(f"*Random Forest Tahmini*: {'Evet' if rf_prediction == 1 else 'Hayır'}")
                st.write(f"*Random Forest Doğruluğu*: {rf_accuracy:.2%}")
            if ann_prediction is not None:
                st.write(f"*ANN Tahmini*: {'Evet' if ann_prediction == 1 else 'Hayır'}")
                st.write(f"*ANN Doğruluğu*: {ann_accuracy:.2%}")
            if not feature_importance.empty:
                st.write("**Özellik Önem Sıralaması**")
                st.bar_chart(feature_importance.sort_values(ascending=False))
            if chart_error:
                st.error(chart_error)
            elif chart_buf:
                st.image(chart_buf, caption="Kullanıcı Değerleri vs TSE Üst Sınırları (İlk 5 Parametre)")
        
        st.markdown("---")
        st.success("Rapor hazır! PDF formatında indirebilirsin.")
        
        pdf_file = generate_pdf(df_tse, df_ec, df_who, rf_prediction, ann_prediction, feature_importance)
        st.download_button("📥 PDF Raporu İndir", data=pdf_file,
                           file_name="su_kalite_raporu.pdf",
                           mime="application/pdf")
