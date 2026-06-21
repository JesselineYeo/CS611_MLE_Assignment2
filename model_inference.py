import argparse
import os
import pickle
import pprint
from datetime import datetime

import numpy as np
import pandas as pd
import pyspark
from pyspark.sql.functions import col, to_date, lit


# Using the following command to run the script:
# python model_inference_champion.py --snapshotdate "2024-07-01"
# python model_inference_champion.py --snapshotdate "2024-07-01" --modelname "champion_model.pkl"


def main(snapshotdate, modelname="champion_model.pkl", feature_store_path="datamart/gold/feature_store/"):
    print("\n\n--- starting inference job ---\n\n")

    spark = (
        pyspark.sql.SparkSession.builder
        .appName("model_inference")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    config = {
        "snapshot_date_str": snapshotdate,
        "snapshot_date": datetime.strptime(snapshotdate, "%Y-%m-%d"),
        "model_name": modelname,
        "model_bank_directory": "model_bank/",
        "feature_store_path": feature_store_path,
    }
    config["model_artefact_filepath"] = os.path.join(
        config["model_bank_directory"],
        config["model_name"]
    )

    pprint.pprint(config)

    # load model artefact
    with open(config["model_artefact_filepath"], "rb") as file:
        model_artefact = pickle.load(file)

    model = model_artefact["model"]
    feature_cols = model_artefact["feature_cols"]

    print("Model loaded successfully:", config["model_artefact_filepath"])
    print("Champion model:", model_artefact.get("model_name", "unknown"))
    print("Model version:", model_artefact.get("model_version", "unknown"))
    print("Number of features:", len(feature_cols))

    # load gold feature store for the given snapshot_date
    feature_file = (
    config["feature_store_path"]
    + "gold_feature_store_"
    + config["snapshot_date_str"].replace("-", "_")
    + ".parquet"
)

    print("Loading feature store:", feature_file)

    features_store_sdf = spark.read.parquet(feature_file)

    # Make date comparison robust whether snapshot_date is string/date/timestamp.
    features_sdf = (
        features_store_sdf
        .withColumn("snapshot_date_tmp", to_date(col("snapshot_date")))
        .filter(col("snapshot_date_tmp") == lit(config["snapshot_date_str"]))
        .drop("snapshot_date_tmp")
    )

    row_count = features_sdf.count()
    print("Extracted feature rows:", row_count, "for", config["snapshot_date_str"])

    if row_count == 0:
        spark.stop()
        raise ValueError(
            f"No feature rows found for snapshot_date={config['snapshot_date_str']} "
            f"in {config['feature_store_path']}"
        )

    features_pdf = features_sdf.toPandas()

    # prepare infernce data by selecting only the required feature columns
    missing_cols = [c for c in feature_cols if c not in features_pdf.columns]
    if missing_cols:
        spark.stop()
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    X_inference = features_pdf[feature_cols].copy()

    X_inference = X_inference.replace([np.inf, -np.inf], np.nan)

    print("Missing values before fill:", X_inference.isna().sum().sum())
    print(X_inference.isna().sum().sort_values(ascending=False).head(10))

    X_inference = X_inference.fillna(X_inference.median(numeric_only=True))
    X_inference = X_inference.fillna(0)

    print("Missing values after fill:", X_inference.isna().sum().sum())
    print("X_inference shape:", X_inference.shape)

    # model inference
    pred_proba = model.predict_proba(X_inference)[:, 1]
    pred_label = model.predict(X_inference)

    prediction_df = features_pdf[["Customer_ID", "snapshot_date"]].copy()
    prediction_df["model_name"] = model_artefact.get("model_name", config["model_name"].replace(".pkl", ""))
    prediction_df["model_version"] = model_artefact.get("model_version", config["model_name"].replace(".pkl", ""))
    prediction_df["pred_proba"] = pred_proba
    prediction_df["pred_label"] = pred_label

    print(prediction_df.head())

    # save predictions to gold directory
    gold_directory = "datamart/gold/model_predictions/champion_model/"
    os.makedirs(gold_directory, exist_ok=True)

    partition_name = (
        "champion_model_predictions_"
        + config["snapshot_date_str"].replace("-", "_")
        + ".parquet"
    )
    filepath = os.path.join(gold_directory, partition_name)

    spark.createDataFrame(prediction_df).write.mode("overwrite").parquet(filepath)
    print("Saved predictions to:", filepath)

    spark.stop()
    print("\n\n--- completed inference job ---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run champion model inference")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--modelname",
        type=str,
        default="champion_model.pkl",
        help="Model artefact filename. Default: champion_model.pkl"
    )
    parser.add_argument(
        "--featurestorepath",
        type=str,
        default="datamart/gold/feature_store/",
        help="Path to gold feature store parquet directory"
    )

    args = parser.parse_args()
    main(args.snapshotdate, args.modelname, args.featurestorepath)
