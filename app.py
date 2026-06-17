import streamlit as st
import joblib
import pandas as pd
import os

# Set wide layout and page config
st.set_page_config(
    page_title="Prediksi Distribusi Pupuk Jawa Timur",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS — earth-toned "field bulletin" theme, dark text for contrast on pastel backgrounds
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .stApp {
        background-color: #F7F4EC;
    }

    [data-testid="stSidebar"] {
        background-color: #E3EAD9;
        border-right: 1px solid #C8D2BC;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #1E1C16;
    }

    h1, h2, h3 {
        color: #1F3D28;
        font-family: 'IBM Plex Sans', sans-serif;
    }

    p, li {
        color: #1E1C16;
    }

    .stButton>button {
        background-color: #1F3D28;
        color: #F7F4EC;
        border-radius: 4px;
        border: none;
        padding: 0.55rem 1.1rem;
        font-weight: 600;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .stButton>button:hover {
        background-color: #14301F;
        color: #F7F4EC;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Sistem Prediksi Pupuk")
st.sidebar.markdown("---")

st.sidebar.markdown("""
<div style='background:#F7F4EC; border-left:3px solid #1F3D28; padding:12px 14px; border-radius:4px;'>
<p style='font-family:"IBM Plex Sans", sans-serif; font-size:13px; color:#1E1C16; margin:0; line-height:1.5;'>
Memprediksi risiko dan realisasi distribusi pupuk di Jawa Timur, berdasarkan data iklim dan produksi pertanian.
</p>
</div>
""", unsafe_allow_html=True)

# Load pretrained artifacts on startup, store in session state for all pages!
@st.cache_resource
def load_artifacts():
    """Load all required artifacts for inference"""
    # Ensure models directory exists
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)

    artifacts = {}

    # Try to load models; if not present, we'll handle that later
    try:
        artifacts["model_klasifikasi"] = joblib.load("models/classification_model.pkl")
    except:
        artifacts["model_klasifikasi"] = None

    try:
        artifacts["model_regresi"] = joblib.load("models/regression_model.pkl")
    except:
        artifacts["model_regresi"] = None

    try:
        artifacts["scaler"] = joblib.load("models/scaler.pkl")
    except:
        artifacts["scaler"] = None

    try:
        artifacts["kabupaten_mapping"] = joblib.load("models/freq_map.pkl")
    except:
        artifacts["kabupaten_mapping"] = None

    # Load and preprocess dataset for Dashboard
    try:
        df = pd.read_csv("data/Distribusi_Pupuk_Jatim_2023-2025.csv")
        # Clean numeric columns with commas
        for col in ["luas_panen_kedelai", "produksi_kedelai_ton"]:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        artifacts["df"] = df
    except Exception as e:
        st.warning(f"Data tidak ditemukan: {e}")
        artifacts["df"] = None

    return artifacts

# Initialize artifacts in session state ONCE!
if "artifacts" not in st.session_state:
    st.session_state.artifacts = load_artifacts()

# ---- Letterhead-style header ----
st.markdown("""
<div style='border-bottom: 3px solid #1F3D28; padding-bottom: 18px; margin-bottom: 10px;'>
  <p style='font-family: "IBM Plex Mono", monospace; font-size: 13px; letter-spacing: 2px; text-transform: uppercase; color: #6B4A33; margin: 0 0 8px 0;'>Data Pertanian &middot; Jawa Timur</p>
  <h1 style='font-family: "Source Serif 4", Georgia, serif; font-size: 40px; color: #1F3D28; margin: 0 0 10px 0; font-weight: 600;'>Sistem Prediksi Distribusi Pupuk</h1>
  <p style='font-family: "IBM Plex Sans", sans-serif; font-size: 16px; color: #1E1C16; max-width: 680px; margin: 0; line-height: 1.6;'>
  Memprediksi realisasi distribusi pupuk NPK dan tingkat risiko kekurangan pasokan di tiap kabupaten/kota, berdasarkan data iklim dan produksi padi, jagung, dan kedelai.
  </p>
</div>
<p style='font-family: "IBM Plex Mono", monospace; font-size: 13px; color: #6B4A33; letter-spacing: 0.5px; margin: 0 0 36px 0;'>
38 KABUPATEN/KOTA &nbsp;&middot;&nbsp; 2023&ndash;2025 &nbsp;&middot;&nbsp; PUPUK, IKLIM &amp; PRODUKSI
</p>
""", unsafe_allow_html=True)

# ---- Feature index (table-of-contents style, not a card grid) ----
features = [
    ("Dashboard", "Ringkasan data dan visualisasi distribusi pupuk per kabupaten/kota, dari waktu ke waktu.", "#1F3D28"),
    ("Prediksi", "Coba prediksi tingkat risiko dan total realisasi distribusi pupuk dari data yang Anda masukkan sendiri.", "#8C4A2F"),
    ("Model Info", "Penjelasan arsitektur model, alur pipeline data, dan fitur-fitur yang digunakan.", "#3B4A5A"),
]

rows_html = ""
for name, desc, color in features:
    rows_html += f"""
    <div style='display:flex; align-items:flex-start; gap:18px; padding:18px 0; border-bottom:1px solid #D8D3C4;'>
      <div style='width:4px; min-width:4px; background:{color}; align-self:stretch; border-radius:2px;'></div>
      <div>
        <h3 style='font-family:"IBM Plex Sans", sans-serif; font-size:18px; font-weight:600; color:#1E1C16; margin:0 0 4px 0;'>{name}</h3>
        <p style='font-family:"IBM Plex Sans", sans-serif; font-size:15px; color:#3D3A33; margin:0; line-height:1.5;'>{desc}</p>
      </div>
    </div>
    """

st.markdown(rows_html, unsafe_allow_html=True)