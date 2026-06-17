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

# Custom CSS for agricultural green theme
st.markdown("""
    <style>
    /* Sidebar background */
    [data-testid=stSidebar] {
        background-color: #e8f5e9;
    }
    
    /* Main background */
    .stApp {
        background-color: #f5f5f5;
    }
    
    /* Card styling */
    div[data-testid="stMetricLabel"] {
        font-weight: 500;
    }
    
    /* Header colors */
    h1, h2, h3 {
        color: #2e7d32;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #4caf50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #388e3c;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar header
st.sidebar.title("🌾 Sistem Prediksi Pupuk")
st.sidebar.markdown("---")

# Load pretrained artifacts on startup
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

artifacts = load_artifacts()
st.session_state.artifacts = artifacts

# Show welcome message in sidebar
st.sidebar.subheader("Selamat Datang!")
st.sidebar.info("Sistem ini untuk memprediksi tingkat kebutuhan pupuk dan distribusi pupuk di Jawa Timur.")

# Initialize session state for page navigation
pages = ["Dashboard", "Prediksi", "Model Info"]
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# Sidebar navigation
st.session_state.page = st.sidebar.radio(
    "Navigasi Halaman",
    pages,
    index=pages.index(st.session_state.page)
)

# Now handle routing
page = st.session_state.page

if page == "Dashboard":
    # Execute Dashboard page
    import sys
    from streamlit import runtime
    from streamlit import _is_running_in_script_mode as is_script_mode
    exec(open("pages/01_Dashboard.py", encoding='utf-8').read())
elif page == "Prediksi":
    # Check if prediction page exists
    import os
    if os.path.exists("pages/02_Prediksi.py"):
        exec(open("pages/02_Prediksi.py", encoding='utf-8').read())
    else:
        st.info("Halaman Prediksi sedang dalam pengembangan!")
elif page == "Model Info":
    if os.path.exists("pages/03_Model_Info.py"):
        exec(open("pages/03_Model_Info.py", encoding='utf-8').read())
    else:
        st.info("Halaman Model Info sedang dalam pengembangan!")
