import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.markdown("# 📊 Prediksi Distribusi Pupuk")

# Get artifacts from session state
artifacts = st.session_state.artifacts
df = artifacts.get("df")

if df is not None:
    # Get list of kabupaten
    list_kabupaten = sorted(df['kabupaten'].unique())
else:
    list_kabupaten = []

# --- Form Input ---
with st.form("prediction_form"):
    st.markdown("### Masukkan Data")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        kabupaten = st.selectbox("Kabupaten/Kota", list_kabupaten)
    with col2:
        bulan = st.selectbox("Bulan", [
            "Januari", "Februari", "Maret", "April",
            "Mei", "Juni", "Juli", "Agustus",
            "September", "Oktober", "November", "Desember"
        ])
    with col3:
        tahun = st.selectbox("Tahun", [2023,2024,2025,2026,2027])
    
    st.markdown("---")
    st.markdown("### Alokasi Pupuk")
    col4, col5 = st.columns(2)
    with col4:
        alokasi_urea = st.number_input("Alokasi Urea (ton)", min_value=0.0, value=10000.0, step=100.0)
    with col5:
        alokasi_NPK = st.number_input("Alokasi NPK (ton)", min_value=0.0, value=5000.0, step=100.0)
    
    st.markdown("---")
    st.markdown("### Data Iklim")
    col6, col7 = st.columns(2)
    with col6:
        curah_hujan = st.number_input("Curah Hujan (mm)", min_value=0.0, value=10.0, step=0.1)
    with col7:
        suhu = st.number_input("Suhu (°C)", min_value=0.0, value=27.0, step=0.1)
    
    st.markdown("---")
    st.markdown("### Data Tanaman")
    
    st.subheader("Padi")
    col8, col9 = st.columns(2)
    with col8:
        luas_panen_padi = st.number_input("Luas Panen Padi (ha)", min_value=0.0, value=5000.0, step=100.0)
    with col9:
        produksi_padi_ton = st.number_input("Produksi Padi (ton)", min_value=0.0, value=25000.0, step=100.0)
    
    st.subheader("Jagung")
    col10, col11 = st.columns(2)
    with col10:
        luas_panen_jagung = st.number_input("Luas Panen Jagung (ha)", min_value=0.0, value=2000.0, step=100.0)
    with col11:
        produksi_jagung_ton = st.number_input("Produksi Jagung (ton)", min_value=0.0, value=10000.0, step=100.0)
    
    st.subheader("Kedelai")
    col12, col13 = st.columns(2)
    with col12:
        luas_panen_kedelai = st.number_input("Luas Panen Kedelai (ha)", min_value=0.0, value=500.0, step=10.0)
    with col13:
        produksi_kedelai_ton = st.number_input("Produksi Kedelai (ton)", min_value=0.0, value=200.0, step=10.0)

    submit = st.form_submit_button("Lakukan Prediksi", type="primary")

# --- Processing ---
if submit:
    # Check if models are loaded
    model_klasifikasi = artifacts.get("model_klasifikasi")
    model_regresi = artifacts.get("model_regresi")
    scaler = artifacts.get("scaler")
    kabupaten_mapping = artifacts.get("kabupaten_mapping")

    # Check if all required artifacts are not None
    if any([model_klasifikasi is None, 
            model_regresi is None, 
            scaler is None, 
            kabupaten_mapping is None]):
        st.error("Model belum dilatih! Silakan jalankan training terlebih dahulu!")
    else:
        try:
            # --- Step 1: Feature Engineering ---
            bulan_map = {
                "Januari": 1, "Februari": 2, "Maret":3, "April":4, 
                "Mei":5, "Juni":6, "Juli":7, "Agustus":8, 
                "September":9, "Oktober":10, "November":11, "Desember":12
            }
            bulan_num = bulan_map[bulan]
            
            # Productivity features
            produktivitas_padi = (produksi_padi_ton / luas_panen_padi) if luas_panen_padi > 0 else 0.0
            produktivitas_jagung = (produksi_jagung_ton / luas_panen_jagung) if luas_panen_jagung > 0 else 0.0
            produktivitas_kedelai = (produksi_kedelai_ton / luas_panen_kedelai) if luas_panen_kedelai > 0 else 0.0
            
            # Total area and production
            luas_panen_total = luas_panen_padi + luas_panen_jagung + luas_panen_kedelai
            produksi_total_ton = produksi_padi_ton + produksi_jagung_ton + produksi_kedelai_ton
            
            # Region encoding
            tipe_wilayah_encoded = 1 if "Kota" in kabupaten else 0
            kabupaten_encoded = kabupaten_mapping.get(kabupaten, kabupaten_mapping.mean())
            
            # --- Step 2: Build features for classification ---
            fitur_klasifikasi = [
                "alokasi_urea", "alokasi_NPK",
                "suhu",
                "produktivitas_padi", "produktivitas_jagung", "produktivitas_kedelai",
                "produksi_total_ton",
                "tipe_wilayah_encoded", "kabupaten_encoded"
            ]
            
            # Numeric features (exact same as data_preprocessing.py)
            fitur_numerik = [
                "curah_hujan", "suhu",
                "produktivitas_padi", "produktivitas_jagung",
                "produktivitas_kedelai",
                "luas_panen_total", "produksi_total_ton",
                "alokasi_urea", "alokasi_NPK",
                "kabupaten_encoded"
            ]
            
            # Build data for scaling (exact columns in scaler's fit order!)
            data_full = pd.DataFrame([{
                "curah_hujan": curah_hujan,
                "suhu": suhu,
                "produktivitas_padi": produktivitas_padi,
                "produktivitas_jagung": produktivitas_jagung,
                "produktivitas_kedelai": produktivitas_kedelai,
                "luas_panen_total": luas_panen_total,
                "produksi_total_ton": produksi_total_ton,
                "alokasi_urea": alokasi_urea,
                "alokasi_NPK": alokasi_NPK,
                "kabupaten_encoded": kabupaten_encoded
            }])
            
            # Apply scaling (fill missing with median if needed, but our inputs are all provided)
            # But since data_full may not have median values, we can just scale directly!
            scaled_data = scaler.transform(data_full[fitur_numerik])
            data_scaled = pd.DataFrame(scaled_data, columns=fitur_numerik)
            
            # Add other features (tipe_wilayah_encoded - it wasn't scaled!)
            data_scaled["tipe_wilayah_encoded"] = tipe_wilayah_encoded
            
            # --- Step 3: Classification ---
            # Prepare classification input
            X_clf = data_scaled[fitur_klasifikasi]
            pred_class = model_klasifikasi.predict(X_clf)[0]
            pred_proba = model_klasifikasi.predict_proba(X_clf)[0]
            
            # Get class labels (in correct order: rendah, sedang, tinggi)
            if hasattr(model_klasifikasi, 'classes_'):
                class_labels = model_klasifikasi.classes_
            else:
                class_labels = ['rendah', 'sedang', 'tinggi']
            
            # --- Step 4: Prepare regression features ---
            # Encode predicted class to ordinal
            kelas_ordinal = {'rendah':0, 'sedang':1, 'tinggi':2}
            pred_kelas_kebutuhan = kelas_ordinal.get(pred_class, 1)
            
            # Create probability columns - ensure order matches class labels
            probas = {}
            for i, c in enumerate(['rendah', 'sedang', 'tinggi']):
                # Find index of class c in model_klasifikasi.classes_
                try:
                    idx = list(class_labels).index(c)
                    probas[f"proba_{c}"] = pred_proba[idx]
                except ValueError:
                    probas[f"proba_{c}"] = 0.0
            
            # Build regression features dict
            data_reg_dict = {}
            
            # Add classification features (from scaled data)
            for col in fitur_klasifikasi:
                data_reg_dict[col] = data_scaled[col].values[0]
            
            # Add additional regression features
            data_reg_dict["curah_hujan"] = data_scaled["curah_hujan"].values[0]
            data_reg_dict["luas_panen_total"] = data_scaled["luas_panen_total"].values[0]
            data_reg_dict["pred_kelas_kebutuhan"] = pred_kelas_kebutuhan
            
            # Add probabilities
            data_reg_dict.update(probas)
            
            # Convert to DataFrame
            data_reg = pd.DataFrame([data_reg_dict])
            
            # Ensure order matches training
            if hasattr(model_regresi, 'feature_names_in_'):
                fitur_regresi = list(model_regresi.feature_names_in_)
                # Reindex and fill any missing with 0
                data_reg = data_reg.reindex(columns=fitur_regresi, fill_value=0.0)
            else:
                # Fallback: use fixed order
                fitur_regresi_default = [
                    "alokasi_urea", "alokasi_NPK", "suhu",
                    "produktivitas_padi", "produktivitas_jagung",
                    "produktivitas_kedelai", "produksi_total_ton",
                    "tipe_wilayah_encoded", "kabupaten_encoded",
                    "curah_hujan", "luas_panen_total",
                    "pred_kelas_kebutuhan", "proba_rendah",
                    "proba_sedang", "proba_tinggi"
                ]
                data_reg = data_reg.reindex(columns=fitur_regresi_default, fill_value=0.0)
            
            # --- Step 5: Regression ---
            pred_distribusi = model_regresi.predict(data_reg)[0]
            
            # --- Display Results ---
            st.markdown("---")
            st.markdown("## Hasil Prediksi")
            
            # Card layout
            col_feat, col_class, col_reg = st.columns([1,1,1])
            
            # --- Feature Summary Card ---
            with col_feat:
                st.markdown("### Ringkasan Fitur")
                st.write(f"**Produktivitas Padi**: {produktivitas_padi:.2f} ton/ha")
                st.write(f"**Produktivitas Jagung**: {produktivitas_jagung:.2f} ton/ha")
                st.write(f"**Produktivitas Kedelai**: {produktivitas_kedelai:.2f} ton/ha")
                st.write(f"**Total Luas Panen**: {luas_panen_total:,.2f} ha")
                st.write(f"**Total Produksi**: {produksi_total_ton:,.2f} ton")
            
            # --- Classification Result ---
            with col_class:
                st.markdown("### Tingkat Kebutuhan Pupuk")
                # Big badge
                if pred_class == 'rendah':
                    st.markdown(f"<div style='background-color: #e8f5e9; color: #2e7d32; font-size:24px; font-weight:bold; padding:20px; border-radius:12px; text-align:center;'>RENDAH</div>", unsafe_allow_html=True)
                elif pred_class == 'sedang':
                    st.markdown(f"<div style='background-color: #fff8e1; color: #f57c00; font-size:24px; font-weight:bold; padding:20px; border-radius:12px; text-align:center;'>SEDANG</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='background-color: #ffebee; color: #c62828; font-size:24px; font-weight:bold; padding:20px; border-radius:12px; text-align:center;'>TINGGI</div>", unsafe_allow_html=True)
                    
                # Probability bars
                st.write("**Probabilitas**:")
                for c, p in zip(class_labels, pred_proba):
                    st.progress(p, text=f"{c}: {p:.2%}")
            
            # --- Regression Result ---
            with col_reg:
                st.markdown("### Prediksi Distribusi Pupuk")
                st.metric("Total Distribusi Pupuk", f"{pred_distribusi:,.0f} Ton")
                
                # Interpretation text
                st.markdown("**Interpretasi**:")
                if pred_class == 'tinggi':
                    st.info(f"Kebutuhan pupuk di {kabupaten} untuk {bulan} {tahun} berada pada tingkat **TINGGI**. Disarankan untuk meningkatkan distribusi pupuk sekitar {pred_distribusi:,.0f} ton.")
                elif pred_class == 'sedang':
                    st.info(f"Kebutuhan pupuk di {kabupaten} untuk {bulan} {tahun} berada pada tingkat **SEDANG**. Distribusi pupuk sebaiknya dipertahankan sekitar {pred_distribusi:,.0f} ton.")
                else:
                    st.info(f"Kebutuhan pupuk di {kabupaten} untuk {bulan} {tahun} berada pada tingkat **RENDAH**. Distribusi pupuk dapat diatur sekitar {pred_distribusi:,.0f} ton.")
            
        except Exception as e:
            st.error(f"Terjadi kesalahan dalam prediksi: {e}")
