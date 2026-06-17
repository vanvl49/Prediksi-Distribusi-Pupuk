import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
import os

RANDOM_STATE = 42


def evaluate_classification(classification_results):
    print("\n" + "=" * 60)
    print("EVALUASI KLASIFIKASI - DETAIL")
    print("=" * 60)

    # Unpack results
    model_klasifikasi = classification_results["model_klasifikasi"]
    model_klasifikasi_nama = classification_results["model_klasifikasi_nama"]
    X_train_clf = classification_results["X_train_clf"]
    y_train_clf = classification_results["y_train_clf"]
    X_test_clf = classification_results["X_test_clf"]
    y_test_clf = classification_results["y_test_clf"]
    pred_final_clf_test = classification_results["pred_final_clf_test"]
    proba_test = classification_results["proba_test"]
    label_order = classification_results["label_order"]
    fitur_klasifikasi = classification_results["fitur_klasifikasi"]

    # Classification Report
    from sklearn.metrics import classification_report
    report_dict = classification_report(
        y_test_clf, pred_final_clf_test,
        labels=label_order, output_dict=True
    )
    report_df = pd.DataFrame(report_dict).T
    print("\nClassification Report (Test Set):")
    print(report_df.round(3))

    # Plot Precision/Recall/F1
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_df = report_df.loc[label_order, ['precision', 'recall', 'f1-score']]
    metrics_df.plot(kind='bar', ax=ax, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax.set_title('Precision, Recall, F1-Score per Kelas (Test Set)', fontsize=14)
    ax.set_xlabel('Kelas')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=9)
    plt.tight_layout()
    plt.savefig("11_classification_metrics.png", dpi=120)
    plt.close()
    print("[Saved] 11_classification_metrics.png")

    # ROC AUC Multiclass (One-vs-Rest)
    try:
        y_test_bin = label_binarize(y_test_clf, classes=label_order)
        y_score = proba_test

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['#2ecc71', '#f39c12', '#e74c3c']

        roc_auc_scores = {}
        for i, kelas in enumerate(label_order):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            roc_auc_scores[kelas] = roc_auc
            ax.plot(fpr, tpr, color=colors[i], lw=2,
                    label=f'{kelas} (AUC = {roc_auc:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (0.5)')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve - Klasifikasi Tingkat Kebutuhan\n({model_klasifikasi_nama})', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("12_roc_auc_multiclass.png", dpi=120)
        plt.close()
        print("[Saved] 12_roc_auc_multiclass.png")

        macro_auc = roc_auc_score(y_test_bin, y_score, multi_class='ovr', average='macro')
        weighted_auc = roc_auc_score(y_test_bin, y_score, multi_class='ovr', average='weighted')
        print(f"\nROC AUC Scores:")
        for kelas, score in roc_auc_scores.items():
            print(f"  {kelas:<10}: {score:.4f}")
        print(f"  Macro Avg  : {macro_auc:.4f}")
        print(f"  Weighted Avg: {weighted_auc:.4f}")
    except Exception as e:
        print(f"\n⚠️ ROC AUC tidak dapat dihitung: {e}")

    # Cross-Validation
    print("\n>>> Cross-Validation Klasifikasi (5-fold)...")
    if model_klasifikasi_nama == "RandomForest":
        cv_model_clf = RandomForestClassifier(
            n_estimators=200, max_depth=8,
            random_state=RANDOM_STATE, class_weight="balanced"
        )
        cv_scores_clf = cross_val_score(cv_model_clf, X_train_clf, y_train_clf, cv=5, scoring='accuracy')
    else:
        cv_model_clf = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=RANDOM_STATE, eval_metric="mlogloss",
        )
        label_map = {"rendah": 0, "sedang": 1, "tinggi": 2}
        y_train_clf_num = y_train_clf.map(label_map)
        cv_scores_clf = cross_val_score(cv_model_clf, X_train_clf, y_train_clf_num, cv=5, scoring='accuracy')

    acc_test = classification_report(
        y_test_clf, pred_final_clf_test, labels=label_order, output_dict=True
    )['accuracy']

    print(f"  CV Accuracy (mean ± std): {cv_scores_clf.mean():.4f} ± {cv_scores_clf.std():.4f}")
    print(f"  Test Accuracy           : {acc_test:.4f}")
    gap_clf = abs(cv_scores_clf.mean() - acc_test)
    if gap_clf > 0.05:
        print(f"  ⚠️ Gap CV-Test: {gap_clf:.4f} (indikasi overfitting)")
    else:
        print(f"  ✓ Gap CV-Test: {gap_clf:.4f} (model stabil)")


def evaluate_regression(regression_results, classification_results):
    print("\n" + "=" * 60)
    print("EVALUASI REGRESI - DETAIL")
    print("=" * 60)

    # Unpack results
    model_regresi_nama = regression_results["model_regresi_nama"]
    X_train_reg = regression_results["X_train_reg"]
    y_train_reg = regression_results["y_train_reg"]
    X_test_reg = regression_results["X_test_reg"]
    y_test_reg = regression_results["y_test_reg"]
    pred_final_reg = regression_results["pred_final_reg"]
    mae_final = regression_results["mae_final"]
    rmse_final = regression_results["rmse_final"]
    r2_final = regression_results["r2_final"]
    fitur_regresi = regression_results["fitur_regresi"]

    # Residuals
    residuals = y_test_reg - pred_final_reg
    residuals_std = np.std(residuals)

    # Additional metrics
    mape = np.mean(np.abs(residuals / y_test_reg.replace(0, np.nan))) * 100
    mape_median = np.median(np.abs(residuals / y_test_reg.replace(0, np.nan))) * 100

    n_reg = len(y_test_reg)
    p_reg = len(fitur_regresi)
    r2_adj = 1 - (1 - r2_final) * (n_reg - 1) / (n_reg - p_reg - 1)

    print(f"\nMetrik Regresi Lengkap:")
    print(f"  MAE          : {mae_final:,.2f} ton")
    print(f"  RMSE         : {rmse_final:,.2f} ton")
    print(f"  R²           : {r2_final:.4f}")
    print(f"  R² Adjusted  : {r2_adj:.4f}")
    print(f"  MAPE         : {mape:.2f}% (mean)")
    print(f"  MAPE Median  : {mape_median:.2f}% (median)")
    print(f"  Std Residual : {residuals_std:,.2f} ton")

    if mape < 10:
        mape_interpretasi = "SANGAT BAIK (<10%)"
    elif mape < 20:
        mape_interpretasi = "BAIK (10-20%)"
    elif mape < 50:
        mape_interpretasi = "CUKUP (20-50%)"
    else:
        mape_interpretasi = "BURUK (>50%)"
    print(f"  Interpretasi MAPE: {mape_interpretasi}")

    # Residual Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].scatter(pred_final_reg, residuals, alpha=0.5, color='steelblue', edgecolors='white')
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0].axhline(y=2*residuals_std, color='orange', linestyle='--', alpha=0.7, label=f'+2σ ({2*residuals_std:,.0f})')
    axes[0].axhline(y=-2*residuals_std, color='orange', linestyle='--', alpha=0.7, label=f'-2σ ({-2*residuals_std:,.0f})')
    axes[0].set_xlabel('Prediksi (ton)', fontsize=12)
    axes[0].set_ylabel('Residual (Aktual - Prediksi)', fontsize=12)
    axes[0].set_title(f'Residual Plot - {model_regresi_nama}', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    sns.histplot(residuals, kde=True, ax=axes[1], color='steelblue', stat='density', bins=20)
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    axes[1].plot(x_norm, stats.norm.pdf(x_norm, 0, residuals_std),
                 'r--', linewidth=2, label=f'Normal (μ=0, σ={residuals_std:,.0f})')
    axes[1].set_xlabel('Residual (ton)', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title(f'Distribusi Residual\n(Shapiro-Wilk p={stats.shapiro(residuals[:min(5000, len(residuals))]).pvalue:.4f})', fontsize=12)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("13_residual_analysis.png", dpi=120)
    plt.close()
    print("[Saved] 13_residual_analysis.png")

    # Cross-Validation
    print("\n>>> Cross-Validation Regresi (5-fold)...")
    if model_regresi_nama == "RandomForest":
        cv_model_reg = RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=RANDOM_STATE
        )
    else:
        cv_model_reg = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE
        )

    cv_r2 = cross_val_score(cv_model_reg, X_train_reg, y_train_reg, cv=5, scoring='r2')
    cv_mae = -cross_val_score(cv_model_reg, X_train_reg, y_train_reg, cv=5, scoring='neg_mean_absolute_error')

    print(f"  CV R²  (mean ± std): {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    print(f"  CV MAE (mean ± std): {cv_mae.mean():,.2f} ± {cv_mae.std():,.2f} ton")
    print(f"  Test R²            : {r2_final:.4f}")
    print(f"  Test MAE           : {mae_final:,.2f} ton")

    gap_reg = abs(cv_r2.mean() - r2_final)
    if gap_reg > 0.1:
        print(f"  ⚠️ Gap CV-Test R²: {gap_reg:.4f} (indikasi overfitting)")
    else:
        print(f"  ✓ Gap CV-Test R²: {gap_reg:.4f} (model stabil)")

    # Error per kelas kebutuhan
    print("\n>>> Analisis Error per Kelas Kebutuhan...")
    test_clf = classification_results["test_clf"]
    test_reg_eval = test_clf.copy()
    test_reg_eval["residual"] = residuals.values
    test_reg_eval["abs_error"] = np.abs(residuals.values)
    test_reg_eval["actual_kelas"] = classification_results["y_test_clf"].loc[test_reg_eval.index].values
    error_per_kelas = test_reg_eval.groupby("actual_kelas").agg(
        mean_abs_error=("abs_error", "mean"),
        std_abs_error=("abs_error", "std"),
        count=("abs_error", "count")
    )
    print("\nMean Absolute Error per Kelas Kebutuhan:")
    print(error_per_kelas.round(2))


def evaluate_all(classification_results, regression_results):
    print("\n" + "=" * 60)
    print("EVALUASI KOMPREHENSIF")
    print("=" * 60)
    evaluate_classification(classification_results)
    evaluate_regression(regression_results, classification_results)


if __name__ == "__main__":
    import joblib
    print("Evaluasi membutuhkan hasil training. Jalankan training.py terlebih dahulu.")
    print("Atau import fungsi evaluate_all() dengan parameter classification_results dan regression_results.")
