# Prediksi Distribusi Pupuk Jawa Timur

## Deskripsi Proyek
Sistem untuk memprediksi **tingkat kebutuhan pupuk** (klasifikasi) dan **total distribusi pupuk** (regresi) di Kabupaten/Kota Jawa Timur menggunakan Machine Learning.


## Struktur Proyek
```
Prediksi Distribusi Pupuk/
├── app.py                      # Entry point Streamlit app
├── test_training.py            # Script untuk test training pipeline
├── requirements.txt            # Daftar paket yang dibutuhkan
├── README.md                   # Dokumentasi ini
├── data/
│   └── Distribusi_Pupuk_Jatim_2023-2025.csv    # Dataset asli
├── pages/                      # Halaman-halaman Streamlit
│   ├── 01_Dashboard.py         # Dashboard visualisasi
│   ├── 02_Prediksi.py          # Halaman prediksi
│   └── 03_Model_Info.py        # Informasi model
├── src/                        # Kode untuk training & preprocessing
│   ├── data_preprocessing.py   # Pipeline preprocessing
│   ├── modeling.py             # Pipeline training
│   ├── evaluation.py           # Fungsi evaluasi model
│   └── data_understanding_dan_EDA.ipynb    # Original notebook EDA
└── models/                     # Folder untuk menyimpan artifacts (auto-created)
    ├── scaler.pkl
    ├── freq_map.pkl
    ├── classification_model.pkl
    ├── regression_model.pkl
    ├── classification_results.pkl
    └── regression_results.pkl
```


## Langkah-langkah Penggunaan

### 1. Instalasi Dependencies
Pastikan Anda menggunakan Python 3.8+, lalu jalankan:
```bash
pip install -r requirements.txt
```


### 2. Training Model (Jika belum ada model)
Jalankan script training dari folder proyek root:
```bash
cd "Prediksi Distribusi Pupuk"
python src/modeling.py
```
Ini akan:
- Menjalankan preprocessing data
- Melatih model klasifikasi dan regresi
- Menyimpan semua artifacts di folder `models/`


### 3. Menjalankan Aplikasi Streamlit
```bash
streamlit run app.py
```

### 4. Menggunakan Aplikasi
1. **Dashboard**: Melihat visualisasi data dan metrik utama
2. **Prediksi**: Masukan data untuk memprediksi tingkat kebutuhan dan distribusi pupuk
3. **Model Info**: Melihat informasi pipeline dan fitur-fitur yang digunakan


## Catatan Penting
- Semua kode sumber di folder `src/` tetap original dengan penambahan modularisasi
- Model dan artifact disimpan di `models/`
- Dataset berada di `data/Distribusi_Pupuk_Jatim_2023-2025.csv`
