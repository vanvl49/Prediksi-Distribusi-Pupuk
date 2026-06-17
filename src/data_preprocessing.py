import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    print("=" * 60)
    print("STEP 1: DATA LOADING & INITIAL CLEANING")
    print("=" * 60)

    df = pd.read_csv(file_path)
    print(f"Initial shape: {df.shape}")

    # Clean comma-separated numeric columns
    kolom_koma = ["luas_panen_kedelai", "produksi_kedelai_ton"]
    for col in kolom_koma:
        df[col] = df[col].astype(str).str.replace(",", ".", regex=False).astype(float)

    # Convert month to numeric
    bulan_map = {
        "Januari": 1, "Februari": 2, "Maret": 3, "April": 4,
        "Mei": 5, "Juni": 6, "Juli": 7, "Agustus": 8,
        "September": 9, "Oktober": 10, "November": 11, "Desember": 12,
    }
    df["bulan_num"] = df["bulan"].map(bulan_map)
    df = df.sort_values(["kabupaten", "tahun", "bulan_num"]).reset_index(drop=True)

    n_dup = df.duplicated(subset=["kabupaten", "tahun", "bulan_num"]).sum()
    print(f"Duplicate rows: {n_dup}")
    print(f"Total missing values: {df.isna().sum().sum()}")

    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 2: FEATURE SELECTION")
    print("=" * 60)

    kolom_za_organik = [
        "alokasi_za",
        "distribusi_za",
        "alokasi_organik",
        "distribusi_organik",
    ]
    df = df.drop(columns=kolom_za_organik)
    print(f"Dropped ZA & organic features: {len(kolom_za_organik)} columns removed.")

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 60)

    # Realization ratios
    for jp in ["urea", "NPK"]:
        df[f"rasio_realisasi_{jp}"] = np.where(
            df[f"alokasi_{jp}"] > 0, df[f"distribusi_{jp}"] / df[f"alokasi_{jp}"], np.nan
        )
    print("rasio_realisasi_urea, rasio_realisasi_NPK created.")

    # Productivity features
    df["produktivitas_padi"] = np.where(
        df["luas_panen_padi"] > 0, df["produksi_padi_ton"] / df["luas_panen_padi"], np.nan
    )
    df["produktivitas_jagung"] = np.where(
        df["luas_panen_jagung"] > 0, df["produksi_jagung_ton"] / df["luas_panen_jagung"], np.nan
    )
    df["produktivitas_kedelai"] = np.where(
        df["luas_panen_kedelai"] > 0, df["produksi_kedelai_ton"] / df["luas_panen_kedelai"], np.nan
    )
    print("produktivitas_padi, produktivitas_jagung, produktivitas_kedelai created.")

    # Total features
    df["luas_panen_total"] = df["luas_panen_padi"] + df["luas_panen_jagung"] + df["luas_panen_kedelai"]
    df["produksi_total_ton"] = df["produksi_padi_ton"] + df["produksi_jagung_ton"] + df["produksi_kedelai_ton"]
    print("luas_panen_total, produksi_total_ton created.")

    # Region type
    df["tipe_wilayah_encoded"] = df["kabupaten"].str.startswith("Kota").astype(int)
    print("tipe_wilayah_encoded created (0=kabupaten, 1=kota).")

    # Quarter
    df["kuartal"] = df["bulan_num"].apply(lambda m: (m - 1) // 3 + 1)
    print("kuartal created.")

    print(f"\nShape after feature engineering: {df.shape}")
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 4: OUTLIER HANDLING")
    print("=" * 60)

    for jp in ["urea", "NPK"]:
        col = f"rasio_realisasi_{jp}"
        n_ekstrem = (df[col] > 3).sum()
        if n_ekstrem > 0:
            df.loc[df[col] > 3, col] = np.nan
        print(f"{col}: {n_ekstrem} extreme values (>3) set to NaN")

    return df


def split_and_encode(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("STEP 5: TRAIN-TEST SPLIT & FREQUENCY ENCODING")
    print("=" * 60)

    df_train = df[df["tahun"].isin([2023, 2024])].copy()
    df_test = df[df["tahun"] == 2025].copy()
    print(f"Training (2023-2024): {df_train.shape[0]} rows")
    print(f"Testing (2025): {df_test.shape[0]} rows")

    # Frequency encoding for kabupaten (based on production)
    freq_map = df_train.groupby("kabupaten")["produksi_total_ton"].mean()
    df_train["kabupaten_encoded"] = df_train["kabupaten"].map(freq_map)
    df_test["kabupaten_encoded"] = df_test["kabupaten"].map(freq_map)
    print("kabupaten_encoded created (based on average production_total_ton from training).")

    return df_train, df_test, freq_map


def scale_features(df_train, df_test, fitur_numerik):
    print("\n" + "=" * 60)
    print("STEP 6: SCALING")
    print("=" * 60)

    scaler = StandardScaler()
    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()

    for col in fitur_numerik:
        median_train = df_train[col].median()
        df_train_scaled[col] = df_train[col].fillna(median_train)
        df_test_scaled[col] = df_test[col].fillna(median_train)

    df_train_scaled[fitur_numerik] = scaler.fit_transform(df_train_scaled[fitur_numerik])
    df_test_scaled[fitur_numerik] = scaler.transform(df_test_scaled[fitur_numerik])
    print(f"Scaling applied to {len(fitur_numerik)} numeric features (fit only on training).")

    return df_train_scaled, df_test_scaled, scaler


def preprocess_data(input_path: str, output_dir: str = "data"):
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 150)

    # Execute pipeline
    df = load_and_clean_data(input_path)
    df = select_features(df)
    df = feature_engineering(df)
    df = handle_outliers(df)
    df_train, df_test, freq_map = split_and_encode(df)

    # Define numeric features for scaling
    fitur_numerik = [
        "curah_hujan", "suhu",
        "produktivitas_padi", "produktivitas_jagung",
        "produktivitas_kedelai",
        "luas_panen_total", "produksi_total_ton",
        "alokasi_urea", "alokasi_NPK",
        "kabupaten_encoded",
    ]

    df_train_scaled, df_test_scaled, scaler = scale_features(df_train, df_test, fitur_numerik)

    # Final feature summary
    print("\n" + "=" * 60)
    print("STEP 7: FINAL FEATURE SET SUMMARY")
    print("=" * 60)

    fitur_final = [
        "kabupaten",
        "tahun",
        "bulan_num",
        "alokasi_urea",
        "alokasi_NPK",
        "curah_hujan",
        "suhu",
        "luas_panen_total",
        "produksi_total_ton",
        "produktivitas_padi",
        "produktivitas_jagung",
        "produktivitas_kedelai",
        "kabupaten_encoded",
        "tipe_wilayah_encoded",
        "rasio_realisasi_urea",
        "rasio_realisasi_NPK",
        "distribusi_urea",
        "distribusi_NPK"
    ]

    print(f"Total final features: {len(fitur_final)} columns")
    print("\nFinal feature list:")
    for f in fitur_final:
        print(f"  - {f}")

    # Export data
    import os
    os.makedirs(output_dir, exist_ok=True)

    df_train.to_csv(os.path.join(output_dir, "training_unscaled.csv"), index=False)
    df_test.to_csv(os.path.join(output_dir, "testing_unscaled.csv"), index=False)
    df_train_scaled.to_csv(os.path.join(output_dir, "distribusi_pupuk_training.csv"), index=False)
    df_test_scaled.to_csv(os.path.join(output_dir, "distribusi_pupuk_test.csv"), index=False)

    print(f"\nFinal train_scaled shape: {df_train_scaled[fitur_final].shape}")
    print(f"Final test_scaled shape: {df_test_scaled[fitur_final].shape}")

    return df_train_scaled, df_test_scaled, scaler, freq_map


if __name__ == "__main__":
    preprocess_data("data/Distribusi_Pupuk_Jatim_2023-2025.csv")
