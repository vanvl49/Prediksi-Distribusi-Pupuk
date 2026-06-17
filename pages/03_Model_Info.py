import streamlit as st
import pandas as pd

st.markdown("# 📚 Model Info")

st.markdown("---")
st.markdown("## Alur Pipeline")

# Pipeline flow
st.markdown("""
```
Input Data 
    ↓
[Feature Engineering]
    ↓
[Scaling Data]
    ↓
[Klasifikasi Tingkat Kebutuhan] → Prediksi Kelas + Probabilitas
    ↓
[Regresi Distribusi Pupuk] (Input: Feature + Hasil Klasifikasi)
    ↓
Output: Tingkat Kebutuhan + Distribusi Pupuk
```
""")

st.markdown("---")
st.markdown("## Deskripsi Model")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Model Klasifikasi")
    st.success("Menggunakan model **Random Forest Classifier** atau **XGBoost Classifier** (tergantung performa terbaik). Model ini memprediksi tingkat kebutuhan pupuk menjadi 3 kategori: **Rendah**, **Sedang**, atau **Tinggi**.")

with col2:
    st.markdown("### Model Regresi")
    st.success("Menggunakan model **Random Forest Regressor** atau **XGBoost Regressor** (tergantung performa terbaik). Model ini memprediksi total distribusi pupuk (urea + NPK) dalam satuan ton.")


st.markdown("---")
st.markdown("## Daftar Fitur")

fitur_klasifikasi = [
    "alokasi_urea",
    "alokasi_NPK",
    "suhu",
    "produktivitas_padi",
    "produktivitas_jagung",
    "produktivitas_kedelai",
    "produksi_total_ton",
    "tipe_wilayah_encoded",
    "kabupaten_encoded"
]

fitur_regresi = fitur_klasifikasi + [
    "curah_hujan",
    "luas_panen_total",
    "pred_kelas_kebutuhan",
    "proba_rendah",
    "proba_sedang",
    "proba_tinggi"
]

# Show feature tables
st.markdown("### Fitur untuk Klasifikasi")
st.table(pd.DataFrame({
    "Fitur": fitur_klasifikasi,
    "Deskripsi": [
        "Alokasi pupuk urea (ton)",
        "Alokasi pupuk NPK (ton)",
        "Suhu rata-rata (°C)",
        "Produktivitas tanaman padi (ton/ha)",
        "Produktivitas tanaman jagung (ton/ha)",
        "Produktivitas tanaman kedelai (ton/ha)",
        "Total produksi tanaman (ton)",
        "Tipe wilayah (0 = Kabupaten, 1 = Kota)",
        "Encoded nilai kabupaten berdasarkan rata-rata produksi"
    ]
}))

st.markdown("### Fitur Tambahan untuk Regresi")
st.table(pd.DataFrame({
    "Fitur": ["curah_hujan", "luas_panen_total", "pred_kelas_kebutuhan", "proba_rendah", "proba_sedang", "proba_tinggi"],
    "Deskripsi": [
        "Curah hujan (mm)",
        "Total luas panen (ha)",
        "Kelas kebutuhan prediksi (0=Rendah, 1=Sedang, 2=Tinggi)",
        "Probabilitas kelas rendah",
        "Probabilitas kelas sedang",
        "Probabilitas kelas tinggi"
    ]
}))
