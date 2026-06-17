import pandas as pd
import os
from data_preprocessing import preprocess_data
from modeling import create_composite_index, train_classification_model, train_regression_model
from evaluation import evaluate_all


def main():
    # Preprocessing data
    print("=" * 60)
    print("MEMULAI PREPROCESSING DATA")
    print("=" * 60)

    # Preprocess the data
    train_scaled, test_scaled, scaler, freq_map = preprocess_data(
        "data/Distribusi_Pupuk_Jatim_2023-2025.csv"
    )

    # Load unscaled data for composite index
    train_unscaled = pd.read_csv("data/training_unscaled.csv")
    test_unscaled = pd.read_csv("data/testing_unscaled.csv")

    # Create composite index
    train_unscaled, test_unscaled = create_composite_index(train_unscaled, test_unscaled)

    # Train classification model
    classification_results = train_classification_model(
        train_scaled, test_scaled, train_unscaled, test_unscaled
    )

    # Train regression model
    regression_results = train_regression_model(classification_results)

    # Save models
    import joblib
    os.makedirs("models", exist_ok=True)
    joblib.dump(classification_results["model_klasifikasi"], "models/classification_model.pkl")
    joblib.dump(regression_results["model_regresi"], "models/regression_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(freq_map, "models/freq_map.pkl")
    joblib.dump(classification_results, "models/classification_results.pkl")
    joblib.dump(regression_results, "models/regression_results.pkl")

    # Evaluate models
    evaluate_all(classification_results, regression_results)

    print("\n" + "=" * 60)
    print("TRAINING & EVALUASI SELESAI - MODEL DIPERSEMBAHKAN DI FOLDER models/")
    print("=" * 60)


if __name__ == "__main__":
    main()
