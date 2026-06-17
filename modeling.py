import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
)
from xgboost import XGBClassifier, XGBRegressor

RANDOM_STATE = 42


def normalisasi_minmax(series, ref_min, ref_max):
    return (series - ref_min) / (ref_max - ref_min)


def create_composite_index(train_unscaled, test_unscaled):
    print("=" * 60)
    print("TAHAP 1: COMPOSITE INDEX & LABEL KLASIFIKASI")
    print("=" * 60)

    # Hitung rasio rata-rata
    train_unscaled["rasio_avg"] = (
        train_unscaled["rasio_realisasi_urea"] + train_unscaled["rasio_realisasi_NPK"]
    ) / 2
    test_unscaled["rasio_avg"] = (
        test_unscaled["rasio_realisasi_urea"] + test_unscaled["rasio_realisasi_NPK"]
    ) / 2

    # Imputasi NaN
    median_rasio_train = train_unscaled["rasio_avg"].median()
    train_unscaled["rasio_avg"] = train_unscaled["rasio_avg"].fillna(median_rasio_train)
    test_unscaled["rasio_avg"] = test_unscaled["rasio_avg"].fillna(median_rasio_train)
    print(f"Median rasio_avg training (untuk imputasi NaN): {median_rasio_train:.4f}")

    # Komponen composite index dengan bobot sama
    bobot_sama = 1/3
    komponen = {
        "rasio_avg": bobot_sama,
        "luas_panen_total": bobot_sama,
        "curah_hujan": bobot_sama,
    }

    composite_train = pd.Series(0.0, index=train_unscaled.index)
    composite_test = pd.Series(0.0, index=test_unscaled.index)

    print(f"\nKomponen composite index (BOBOT SAMA = {bobot_sama:.4f}):")
    print("-" * 60)
    for col, bobot in komponen.items():
        ref_min = train_unscaled[col].min()
        ref_max = train_unscaled[col].max()

        norm_train = normalisasi_minmax(train_unscaled[col], ref_min, ref_max)
        norm_test = normalisasi_minmax(test_unscaled[col], ref_min, ref_max).clip(0, 1)

        composite_train += bobot * norm_train
        composite_test += bobot * norm_test

        print(f"  {col:<25} | bobot={bobot:.4f} | range training=[{ref_min:.3f}, {ref_max:.3f}]")

    train_unscaled["composite_index"] = composite_train
    test_unscaled["composite_index"] = composite_test

    print(f"\nStatistik composite_index (training):")
    print(train_unscaled["composite_index"].describe().round(4))

    # Binning jadi 3 kelas
    q33, q67 = train_unscaled["composite_index"].quantile([1/3, 2/3])
    print(f"\nThreshold tercile (dari training):")
    print(f"  Q33 (33.3%): {q33:.4f}")
    print(f"  Q67 (66.7%): {q67:.4f}")

    def bin_kelas(x):
        if pd.isna(x):
            return np.nan
        elif x <= q33:
            return "rendah"
        elif x <= q67:
            return "sedang"
        else:
            return "tinggi"

    train_unscaled["tingkat_kebutuhan"] = train_unscaled["composite_index"].apply(bin_kelas)
    test_unscaled["tingkat_kebutuhan"] = test_unscaled["composite_index"].apply(bin_kelas)

    print(f"\nDistribusi kelas (training):")
    print(train_unscaled["tingkat_kebutuhan"].value_counts())
    print(f"\nDistribusi kelas (testing):")
    print(test_unscaled["tingkat_kebutuhan"].value_counts())

    # Plot distribusi composite index
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=train_unscaled, x="tingkat_kebutuhan", y="composite_index",
        order=["rendah", "sedang", "tinggi"], ax=ax,
        palette={"rendah": "#2ecc71", "sedang": "#f39c12", "tinggi": "#e74c3c"}
    )
    ax.set_title("Distribusi Composite Index per Kelas Tingkat Kebutuhan (Training)\n"
                 "(Bobot Sama: rasio_avg + luas_panen + curah_hujan)", fontsize=13)
    ax.set_xlabel("Tingkat Kebutuhan")
    ax.set_ylabel("Composite Index (0-1)")
    plt.tight_layout()
    plt.savefig("06_distribusi_composite_index.png", dpi=120)
    plt.close()
    print("\n[Saved] 06_distribusi_composite_index.png")

    return train_unscaled, test_unscaled


def train_classification_model(train_scaled, test_scaled, train_unscaled, test_unscaled):
    print("\n" + "=" * 60)
    print("TAHAP 2: MODEL KLASIFIKASI (DATA SCALED)")
    print("=" * 60)

    fitur_klasifikasi = [
        "alokasi_urea", "alokasi_NPK",
        "suhu",
        "produktivitas_padi", "produktivitas_jagung", "produktivitas_kedelai",
        "produksi_total_ton",
        "tipe_wilayah_encoded", "kabupaten_encoded",
    ]
    print(f"Fitur input klasifikasi ({len(fitur_klasifikasi)}):")
    for f in fitur_klasifikasi:
        print(f"  - {f}")
    print("\n(YANG DIKECUALIKAN: rasio_realisasi_*, luas_panen_total, curah_hujan)")
    print("(Alasan: ketiganya adalah komponen pembentuk composite index)")

    # Gabungkan label dari unscaled ke scaled
    train_scaled["tingkat_kebutuhan"] = train_unscaled["tingkat_kebutuhan"].values
    test_scaled["tingkat_kebutuhan"] = test_unscaled["tingkat_kebutuhan"].values

    # Drop baris dengan NaN
    train_clf = train_scaled.dropna(subset=fitur_klasifikasi + ["tingkat_kebutuhan"]).copy()
    test_clf = test_scaled.dropna(subset=fitur_klasifikasi + ["tingkat_kebutuhan"]).copy()
    print(f"\nBaris dipakai - train: {train_clf.shape[0]}, test: {test_clf.shape[0]}")

    X_train_clf = train_clf[fitur_klasifikasi]
    y_train_clf = train_clf["tingkat_kebutuhan"]
    X_test_clf = test_clf[fitur_klasifikasi]
    y_test_clf = test_clf["tingkat_kebutuhan"]

    label_order = ["rendah", "sedang", "tinggi"]

    # Random Forest Classifier
    print("\n>>> Training Random Forest Classifier...")
    rf_clf = RandomForestClassifier(
        n_estimators=200, max_depth=8,
        random_state=RANDOM_STATE, class_weight="balanced"
    )
    rf_clf.fit(X_train_clf, y_train_clf)
    pred_rf = rf_clf.predict(X_test_clf)
    acc_rf = accuracy_score(y_test_clf, pred_rf)
    print(f"[Random Forest] Accuracy: {acc_rf:.4f}")
    print(classification_report(y_test_clf, pred_rf, labels=label_order))

    # XGBoost Classifier
    print("\n>>> Training XGBoost Classifier...")
    label_map = {"rendah": 0, "sedang": 1, "tinggi": 2}
    y_train_clf_num = y_train_clf.map(label_map)
    y_test_clf_num = y_test_clf.map(label_map)

    xgb_clf = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=RANDOM_STATE, eval_metric="mlogloss",
    )
    xgb_clf.fit(X_train_clf, y_train_clf_num)
    pred_xgb_num = xgb_clf.predict(X_test_clf)
    acc_xgb = accuracy_score(y_test_clf_num, pred_xgb_num)
    print(f"[XGBoost] Accuracy: {acc_xgb:.4f}")
    print(classification_report(y_test_clf_num, pred_xgb_num, target_names=label_order))

    # Pilih model terbaik
    if acc_rf >= acc_xgb:
        model_klasifikasi_terbaik = "RandomForest"
        pred_final_clf_train = rf_clf.predict(X_train_clf)
        pred_final_clf_test = pred_rf
        proba_train = rf_clf.predict_proba(X_train_clf)
        proba_test = rf_clf.predict_proba(X_test_clf)
        proba_classes = list(rf_clf.classes_)
    else:
        model_klasifikasi_terbaik = "XGBoost"
        pred_final_clf_train_num = xgb_clf.predict(X_train_clf)
        pred_final_clf_train = pd.Series(pred_final_clf_train_num).map(
            {v: k for k, v in label_map.items()}
        ).values
        pred_final_clf_test = pd.Series(pred_xgb_num).map(
            {v: k for k, v in label_map.items()}
        ).values
        proba_train = xgb_clf.predict_proba(X_train_clf)
        proba_test = xgb_clf.predict_proba(X_test_clf)
        proba_classes = [label_order[i] for i in range(3)]

    print(f"\n>>> Model klasifikasi terbaik: {model_klasifikasi_terbaik} "
          f"(Accuracy={max(acc_rf, acc_xgb):.4f})")

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_test_clf, pred_final_clf_test, labels=label_order)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_order, yticklabels=label_order, ax=ax)
    ax.set_title(f"Confusion Matrix - {model_klasifikasi_terbaik}\n"
                 f"(Accuracy: {max(acc_rf, acc_xgb):.4f})")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    plt.tight_layout()
    plt.savefig("07_confusion_matrix.png", dpi=120)
    plt.close()
    print("[Saved] 07_confusion_matrix.png")

    # Feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    if model_klasifikasi_terbaik == "RandomForest":
        importances = rf_clf.feature_importances_
    else:
        importances = xgb_clf.feature_importances_
    feat_imp = pd.Series(importances, index=fitur_klasifikasi).sort_values()
    feat_imp.plot(kind="barh", ax=ax, color=['#3498db' if v > feat_imp.median() else '#95a5a6' for v in feat_imp.values])
    ax.set_title(f"Feature Importance - Klasifikasi Tingkat Kebutuhan\n"
                 f"({model_klasifikasi_terbaik})", fontsize=13)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("08_feature_importance_klasifikasi.png", dpi=120)
    plt.close()
    print("[Saved] 08_feature_importance_klasifikasi.png")

    # Simpan data untuk regresi
    train_clf["distribusi_total"] = (
        train_unscaled.loc[train_clf.index, "distribusi_urea"] +
        train_unscaled.loc[train_clf.index, "distribusi_NPK"]
    )
    test_clf["distribusi_total"] = (
        test_unscaled.loc[test_clf.index, "distribusi_urea"] +
        test_unscaled.loc[test_clf.index, "distribusi_NPK"]
    )

    return {
        "model_klasifikasi": rf_clf if model_klasifikasi_terbaik == "RandomForest" else xgb_clf,
        "model_klasifikasi_nama": model_klasifikasi_terbaik,
        "X_train_clf": X_train_clf,
        "y_train_clf": y_train_clf,
        "X_test_clf": X_test_clf,
        "y_test_clf": y_test_clf,
        "pred_final_clf_train": pred_final_clf_train,
        "pred_final_clf_test": pred_final_clf_test,
        "proba_train": proba_train,
        "proba_test": proba_test,
        "proba_classes": proba_classes,
        "train_clf": train_clf,
        "test_clf": test_clf,
        "label_order": label_order,
        "fitur_klasifikasi": fitur_klasifikasi
    }


def train_regression_model(classification_results):
    print("\n" + "=" * 60)
    print("TAHAP 3: MODEL REGRESI - PREDIKSI TOTAL DISTRIBUSI PUPUK (DATA SCALED)")
    print("=" * 60)

    train_clf = classification_results["train_clf"]
    test_clf = classification_results["test_clf"]
    pred_final_clf_train = classification_results["pred_final_clf_train"]
    pred_final_clf_test = classification_results["pred_final_clf_test"]
    proba_train = classification_results["proba_train"]
    proba_test = classification_results["proba_test"]
    proba_classes = classification_results["proba_classes"]
    fitur_klasifikasi = classification_results["fitur_klasifikasi"]

    print(f"Target regresi: distribusi_total = distribusi_urea + distribusi_NPK (ton/bulan)")
    print(f"Statistik target (training):")
    print(f"  Mean  : {train_clf['distribusi_total'].mean():,.2f} ton")
    print(f"  Median: {train_clf['distribusi_total'].median():,.2f} ton")
    print(f"  Min   : {train_clf['distribusi_total'].min():,.2f} ton")
    print(f"  Max   : {train_clf['distribusi_total'].max():,.2f} ton")

    # Tambahkan fitur dari hasil klasifikasi
    kelas_ordinal_map = {"rendah": 0, "sedang": 1, "tinggi": 2}
    train_clf["pred_kelas_kebutuhan"] = pd.Series(
        pred_final_clf_train, index=train_clf.index
    ).map(kelas_ordinal_map)
    test_clf["pred_kelas_kebutuhan"] = pd.Series(
        pred_final_clf_test, index=test_clf.index
    ).map(kelas_ordinal_map)

    for i, kelas in enumerate(proba_classes):
        train_clf[f"proba_{kelas}"] = proba_train[:, i]
        test_clf[f"proba_{kelas}"] = proba_test[:, i]

    print("\nFitur dari hasil klasifikasi ditambahkan:")
    print("  - pred_kelas_kebutuhan (ordinal: 0=rendah, 1=sedang, 2=tinggi)")
    print(f"  - proba_rendah, proba_sedang, proba_tinggi")

    fitur_regresi = fitur_klasifikasi + [
        "curah_hujan", "luas_panen_total",
        "pred_kelas_kebutuhan",
        "proba_rendah", "proba_sedang", "proba_tinggi",
    ]
    print(f"\nFitur input regresi ({len(fitur_regresi)} kolom):")
    for f in fitur_regresi:
        print(f"  - {f}")

    train_reg = train_clf.dropna(subset=fitur_regresi + ["distribusi_total"]).copy()
    test_reg = test_clf.dropna(subset=fitur_regresi + ["distribusi_total"]).copy()
    print(f"\nBaris dipakai - train: {train_reg.shape[0]}, test: {test_reg.shape[0]}")

    X_train_reg = train_reg[fitur_regresi]
    y_train_reg = train_reg["distribusi_total"]
    X_test_reg = test_reg[fitur_regresi]
    y_test_reg = test_reg["distribusi_total"]

    # Random Forest Regressor
    print("\n>>> Training Random Forest Regressor...")
    rf_reg = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=RANDOM_STATE
    )
    rf_reg.fit(X_train_reg, y_train_reg)
    pred_rf_reg = rf_reg.predict(X_test_reg)
    mae_rf = mean_absolute_error(y_test_reg, pred_rf_reg)
    rmse_rf = np.sqrt(mean_squared_error(y_test_reg, pred_rf_reg))
    r2_rf = r2_score(y_test_reg, pred_rf_reg)
    print(f"[Random Forest] MAE={mae_rf:,.2f} ton | RMSE={rmse_rf:,.2f} ton | R²={r2_rf:.4f}")

    # XGBoost Regressor
    print("\n>>> Training XGBoost Regressor...")
    xgb_reg = XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=RANDOM_STATE
    )
    xgb_reg.fit(X_train_reg, y_train_reg)
    pred_xgb_reg = xgb_reg.predict(X_test_reg)
    mae_xgb = mean_absolute_error(y_test_reg, pred_xgb_reg)
    rmse_xgb = np.sqrt(mean_squared_error(y_test_reg, pred_xgb_reg))
    r2_xgb = r2_score(y_test_reg, pred_xgb_reg)
    print(f"[XGBoost]  MAE={mae_xgb:,.2f} ton | RMSE={rmse_xgb:,.2f} ton | R²={r2_xgb:.4f}")

    # Pilih model terbaik
    if r2_rf >= r2_xgb:
        model_regresi_terbaik = "RandomForest"
        pred_final_reg = pred_rf_reg
        mae_final = mae_rf
        rmse_final = rmse_rf
        r2_final = r2_rf
        model_reg_obj = rf_reg
    else:
        model_regresi_terbaik = "XGBoost"
        pred_final_reg = pred_xgb_reg
        mae_final = mae_xgb
        rmse_final = rmse_xgb
        r2_final = r2_xgb
        model_reg_obj = xgb_reg

    print(f"\n>>> Model regresi terbaik: {model_regresi_terbaik}")
    print(f"    MAE  = {mae_final:,.2f} ton")
    print(f"    RMSE = {rmse_final:,.2f} ton")
    print(f"    R²   = {r2_final:.4f}")

    # Plot actual vs predicted
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test_reg, pred_final_reg, alpha=0.5, color="steelblue", edgecolors='white', linewidth=0.5)
    batas_min = min(y_test_reg.min(), pred_final_reg.min())
    batas_max = max(y_test_reg.max(), pred_final_reg.max())
    ax.plot([batas_min, batas_max], [batas_min, batas_max], "r--", linewidth=2, label="Prediksi = Aktual")
    ax.set_xlabel("Distribusi Total Aktual (ton)", fontsize=12)
    ax.set_ylabel("Distribusi Total Prediksi (ton)", fontsize=12)
    ax.set_title(f"Aktual vs Prediksi - {model_regresi_terbaik}\n"
                 f"(MAE={mae_final:,.0f} ton, R²={r2_final:.4f})", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("09_aktual_vs_prediksi.png", dpi=120)
    plt.close()
    print("[Saved] 09_aktual_vs_prediksi.png")

    # Feature importance
    fig, ax = plt.subplots(figsize=(10, 7))
    if model_regresi_terbaik == "RandomForest":
        importances_reg = rf_reg.feature_importances_
    else:
        importances_reg = xgb_reg.feature_importances_
    feat_imp_reg = pd.Series(importances_reg, index=fitur_regresi).sort_values()
    colors_reg = ['#27ae60' if 'proba' in f or 'pred_kelas' in f else '#3498db' for f in feat_imp_reg.index]
    feat_imp_reg.plot(kind="barh", ax=ax, color=colors_reg)
    ax.set_title(f"Feature Importance - Regresi Distribusi Pupuk\n"
                 f"({model_regresi_terbaik}) | Hijau = fitur hasil klasifikasi", fontsize=13)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("10_feature_importance_regresi.png", dpi=120)
    plt.close()
    print("[Saved] 10_feature_importance_regresi.png")

    # Ringkasan hasil
    print("\n" + "=" * 60)
    print("RINGKASAN HASIL PIPELINE")
    print("=" * 60)
    print(f"Data yang digunakan:")
    print(f"  - Composite Index : data UNSCALED (nilai asli)")
    print(f"  - Klasifikasi     : data SCALED")
    print(f"  - Regresi         : data SCALED")
    print(f"")
    print(f"Tahap 1 (Composite Index):")
    print(f"  Komponen : rasio_avg + luas_panen_total + curah_hujan")
    print(f"  Bobot    : SAMA RATA (1/3 ≈ 0.333 each)")
    print(f"  Binning  : Tercile (33% / 67%)")
    print(f"")
    print(f"Tahap 2 (Klasifikasi):")
    print(f"  Model    : {classification_results['model_klasifikasi_nama']}")
    print(f"  Accuracy : {accuracy_score(classification_results['y_test_clf'], classification_results['pred_final_clf_test']):.4f}")
    print(f"  Fitur    : {len(fitur_klasifikasi)} kolom (exclude komponen label)")
    print(f"")
    print(f"Tahap 3 (Regresi):")
    print(f"  Model    : {model_regresi_terbaik}")
    print(f"  R²       : {r2_final:.4f}")
    print(f"  MAE      : {mae_final:,.2f} ton")
    print(f"  RMSE     : {rmse_final:,.2f} ton")
    print(f"  Fitur    : {len(fitur_regresi)} kolom (termasuk hasil klasifikasi)")

    if r2_final > 0.7:
        interpretasi = "BAIK (model menjelaskan >70% varians)"
    elif r2_final > 0.5:
        interpretasi = "CUKUP (model menjelaskan >50% varians)"
    else:
        interpretasi = "PERLU IMPROVEMENT (model <50% varians)"
    print(f"\nInterpretasi R²: {interpretasi}")
    print(f"Rata-rata kesalahan prediksi: ±{mae_final:,.0f} ton dari total distribusi")
    print("\n=== MODELING SELESAI ===")

    return {
        "model_regresi": model_reg_obj,
        "model_regresi_nama": model_regresi_terbaik,
        "X_train_reg": X_train_reg,
        "y_train_reg": y_train_reg,
        "X_test_reg": X_test_reg,
        "y_test_reg": y_test_reg,
        "pred_final_reg": pred_final_reg,
        "mae_final": mae_final,
        "rmse_final": rmse_final,
        "r2_final": r2_final,
        "fitur_regresi": fitur_regresi
    }
