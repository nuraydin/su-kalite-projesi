import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Su Kalite Testi", layout="wide")

CSV_FILE = "su_kalite_standartlari.txt"
LOGO_PATH = "mar_logo.png"

@st.cache_data
def fetch_limits():
    df = pd.read_csv(CSV_FILE)
    df.rename(columns={df.columns[0]: "Parametre",
                       df.columns[1]: "TSE",
                       df.columns[2]: "EC",
                       df.columns[3]: "WHO"}, inplace=True)
    df = df.dropna(subset=["Parametre"]).reset_index(drop=True)

    drop_keywords = ["Kabul Edilebilir", "STANDARTLAR", "Fiziksel ve Duyusal",
                     "EMS/100", "Organoleptik", "Renk", "Bulanıklık", "Koku", "Tat","Siyanür (CN)","Selenyum (Se)","Antimon (Sb)",
                     "C.perfringers","Pseudomonas Aeruginosa"]
    mask = ~df["Parametre"].str.contains("|".join(drop_keywords), na=False)
    df = df[mask].reset_index(drop=True)
    return df

def parse_range(r):
    if pd.isna(r):
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
def generate_pdf(tse_df, ec_df, who_df):
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        for title, df in [("TSE Sonuçları", tse_df), ("EC Sonuçları", ec_df), ("WHO Sonuçları", who_df)]:
            fig, ax = plt.subplots(figsize=(12, 9))  
            ax.axis('off')

            try:
                logo = Image.open("mar_logo.png")
                fig.figimage(logo, xo=40, yo=fig.bbox.ymax - 100, zoom=0.15)
            except:
                pass

            
            fig.text(0.5, 0.95, "💧 İÇME SUYU KALİTE RAPORU", fontsize=20, ha="center", weight='bold')
            fig.text(0.5, 0.91, title, fontsize=16, ha="center", weight='bold')

            
            table = ax.table(cellText=df.values,
                             colLabels=df.columns,
                             cellLoc='center',
                             loc='center',
                             colWidths=[0.35, 0.2, 0.25])

            table.auto_set_font_size(False)
            table.set_fontsize(11)

            for key, cell in table.get_celld().items():
                cell.set_linewidth(0.5)
                if key[0] == 0:
                    cell.set_fontsize(12)
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor("#f2f2f2")
                else:
                    durum = df.iloc[key[0]-1, 2]
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
st.title("💧 İçme Suyu Kalite Testi")
st.caption("📌 Lütfen sadece sayısal değer giriniz. Boş bırakabilirsiniz.")

df_limits = fetch_limits()
input_values = {}

cols = st.columns(4)
for idx, row in df_limits.iterrows():
    param = row["Parametre"]
    with cols[idx % 4]:
        input_values[param] = st.number_input(param, format="%.4f", key=param)

if st.button("💡 Hesapla"):
    df_tse = create_results(df_limits, "TSE", input_values)
    df_ec = create_results(df_limits, "EC", input_values)
    df_who = create_results(df_limits, "WHO", input_values)

    tabs = st.tabs(["📘 TSE", "📗 EC", "📕 WHO"])
    with tabs[0]:
        st.subheader("TSE Sonuçları")
        st.dataframe(df_tse.style.applymap(color_code, subset=["Durum"]))
    with tabs[1]:
        st.subheader("EC Sonuçları")
        st.dataframe(df_ec.style.applymap(color_code, subset=["Durum"]))
    with tabs[2]:
        st.subheader("WHO Sonuçları")
        st.dataframe(df_who.style.applymap(color_code, subset=["Durum"]))

    st.markdown("---")
    st.success("Rapor hazır! PDF formatında indirebilirsin.")

    pdf_file = generate_pdf(df_tse, df_ec, df_who)
    st.download_button("📥 PDF Raporu İndir", data=pdf_file,
                       file_name="su_kalite_raporu.pdf",
                       mime="application/pdf")