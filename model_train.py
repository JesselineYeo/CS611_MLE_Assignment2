
### 3 models are being trained, all artefacts saved and champion model saved.
### Under MASH: python model_train.py 
### Dataset: training-data datamart/gold/training_dataset/merged_tb.parquet
### Outputs: 1) model_bank/credit_model_<model>_<run_date>.pkl, 2) model_bank/champion_model.pkl, 3) model_bank/model_metrics.csv, 4)model_bank/model_registry.csv

import argparse
import os
import pickle
from datetime import datetime

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# using the training dataset created in gold/training_dataset folder
def main(
    training_data="datamart/gold/training_dataset/merged_tb.parquet",
    model_bank_directory="model_bank",
    test_size=0.2,
    random_state=42
):
    print("--- start model training job ---")

    # 1. Set model training date
    model_train_date_str = datetime.today().strftime("%Y-%m-%d")

    # 2. Load training data
    merged_tb = pd.read_parquet(training_data)

    print("Loaded training data:", merged_tb.shape)

    # 3. Prepare X and y
    drop_cols = [
        "Customer_ID",
        "snapshot_date",
        "loan_id",
        "Type_of_Loan",
        "label"
    ]

    feature_cols = [
        col for col in merged_tb.columns
        if col not in drop_cols
    ]

    X = merged_tb[feature_cols]
    y = merged_tb["label"].astype(int)

    # Keep numeric features only
    X = X.select_dtypes(include=["number"])

    # Fill any remaining missing values
    X = X.fillna(X.median())

    feature_cols = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("Train default rate:", round(y_train.mean(), 4))
    print("Test default rate:", round(y_test.mean(), 4))

    # 4. Train models

    # logistic regression with standard scaler
    log_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0
        ))
    ])

    # random forest classifier
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    )

    # class imbalance weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    # xgboost classifier
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    )

    trained_models = {
        "logistic_regression": log_model,
        "random_forest": rf_model,
        "xgboost": xgb_model
    }

    # 5. Evaluate all models
    model_results = []

    for model_name, model in trained_models.items():
        print("\nTraining:", model_name)

        model.fit(X_train, y_train)

        train_proba = model.predict_proba(X_train)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_auc = roc_auc_score(y_train, train_proba)
        test_auc = roc_auc_score(y_test, test_proba)

        model_results.append({
            "model_name": model_name,
            "auc_train": train_auc,
            "auc_test": test_auc,
            "gini_train": round(2 * train_auc - 1, 3),
            "gini_test": round(2 * test_auc - 1, 3),
            "accuracy_test": accuracy_score(y_test, test_pred),
            "precision_test": precision_score(y_test, test_pred, zero_division=0),
            "recall_test": recall_score(y_test, test_pred, zero_division=0),
            "f1_test": f1_score(y_test, test_pred, zero_division=0),
            "train_rows": X_train.shape[0],
            "test_rows": X_test.shape[0],
            "train_default_rate": round(y_train.mean(), 2),
            "test_default_rate": round(y_test.mean(), 2)
        })

        print("Test AUC:", round(test_auc, 4))
        print(classification_report(y_test, test_pred, zero_division=0))

    model_metrics = pd.DataFrame(model_results).sort_values(
        "auc_test",
        ascending=False
    )

    print("\nModel metrics:")
    print(model_metrics)

    # 6. Select champion model
    champion_model_name = model_metrics.iloc[0]["model_name"]
    champion_model = trained_models[champion_model_name]

    print("Champion model:", champion_model_name)

    # 7. Create model bank folder
    os.makedirs(model_bank_directory, exist_ok=True)

    # 8. Save all 3 model artefacts
    saved_artefacts = []

    for model_name, model in trained_models.items():
        row = model_metrics[
            model_metrics["model_name"] == model_name
        ].iloc[0]

        model_version = (
            "credit_model_"
            + model_name
            + "_"
            + model_train_date_str.replace("-", "_")
        )

        model_artefact = {
            "model": model,
            "model_name": model_name,
            "model_version": model_version,
            "is_champion": model_name == champion_model_name,
            "feature_cols": feature_cols,
            "data_dates": {
                "model_train_date": model_train_date_str
            },
            "data_stats": {
                "X_train_rows": X_train.shape[0],
                "X_test_rows": X_test.shape[0],
                "y_train_default_rate": round(y_train.mean(), 2),
                "y_test_default_rate": round(y_test.mean(), 2)
            },
            "results": {
                "auc_train": row["auc_train"],
                "auc_test": row["auc_test"],
                "gini_train": row["gini_train"],
                "gini_test": row["gini_test"],
                "accuracy_test": row["accuracy_test"],
                "precision_test": row["precision_test"],
                "recall_test": row["recall_test"],
                "f1_test": row["f1_test"]
            },
            "hp_params": model.get_params()
        }

        file_path = os.path.join(
            model_bank_directory,
            model_version + ".pkl"
        )

        with open(file_path, "wb") as file:
            pickle.dump(model_artefact, file)

        saved_artefacts.append({
            "model_name": model_name,
            "model_version": model_version,
            "is_champion": model_name == champion_model_name,
            "auc_test": row["auc_test"],
            "file_path": file_path
        })

        print(f"Saved {model_name} to {file_path}")

    # 9. Save champion model separately
    champion_version = (
        "credit_model_"
        + champion_model_name
        + "_"
        + model_train_date_str.replace("-", "_")
    )

    champion_source_path = os.path.join(
        model_bank_directory,
        champion_version + ".pkl"
    )

    champion_target_path = os.path.join(
        model_bank_directory,
        "champion_model.pkl"
    )

    with open(champion_source_path, "rb") as file:
        champion_artefact = pickle.load(file)

    with open(champion_target_path, "wb") as file:
        pickle.dump(champion_artefact, file)

    print("Champion model saved to:", champion_target_path)

    # 10. Save registry and metrics
    model_registry = pd.DataFrame(saved_artefacts)

    model_registry.to_csv(
        os.path.join(model_bank_directory, "model_registry.csv"),
        index=False
    )

    model_metrics.to_csv(
        os.path.join(model_bank_directory, "model_metrics.csv"),
        index=False
    )

    print("\nModel registry:")
    print(model_registry)

    # 11. Test load champion pickle
    with open(champion_target_path, "rb") as file:
        loaded_champion_artefact = pickle.load(file)

    loaded_model = loaded_champion_artefact["model"]
    loaded_feature_cols = loaded_champion_artefact["feature_cols"]

    loaded_test_proba = loaded_model.predict_proba(
        X_test[loaded_feature_cols]
    )[:, 1]

    loaded_test_auc = roc_auc_score(
        y_test,
        loaded_test_proba
    )

    print("Loaded champion model:", loaded_champion_artefact["model_name"])
    print("Loaded champion version:", loaded_champion_artefact["model_version"])
    print("Loaded champion test AUC:", loaded_test_auc)
    print("Model loaded successfully!")
    print("--- completed model training job ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--training-data",
        default="datamart/gold/training_dataset/merged_tb.parquet"
    )

    parser.add_argument(
        "--model-bank",
        default="model_bank"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42
    )

    args = parser.parse_args()

    main(
        training_data=args.training_data,
        model_bank_directory=args.model_bank,
        test_size=args.test_size,
        random_state=args.random_state
    )