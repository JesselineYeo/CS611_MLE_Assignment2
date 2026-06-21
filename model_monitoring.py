import argparse
import os
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import pyspark
from pyspark.sql.functions import col
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def read_parquet_any(spark, path):
    """Read a parquet path or folder with Spark."""
    return spark.read.parquet(path)


def calculate_psi(expected, actual, buckets=10):
    """
    Population Stability Index.
    expected = baseline distribution, usually first monitoring month.
    actual = comparison distribution.
    """
    expected = pd.Series(expected).dropna().astype(float)
    actual = pd.Series(actual).dropna().astype(float)

    if expected.empty or actual.empty:
        return np.nan

    breakpoints = np.linspace(0, 1, buckets + 1)
    cuts = np.quantile(expected, breakpoints)
    cuts = np.unique(cuts)

    # If too few unique cut points, fall back to fixed probability bins.
    if len(cuts) <= 2:
        cuts = np.linspace(0, 1, buckets + 1)

    expected_counts = pd.cut(expected, bins=cuts, include_lowest=True).value_counts(sort=False)
    actual_counts = pd.cut(actual, bins=cuts, include_lowest=True).value_counts(sort=False)

    expected_pct = expected_counts / max(expected_counts.sum(), 1)
    actual_pct = actual_counts / max(actual_counts.sum(), 1)

    # Avoid divide-by-zero.
    expected_pct = expected_pct.replace(0, 0.0001)
    actual_pct = actual_pct.replace(0, 0.0001)

    psi = ((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)).sum()
    return float(psi)


def main(
    prediction_path,
    label_store_path,
    output_path,
    monitoring_start,
    monitoring_end,
):
    print("\n\n--- starting model monitoring job ---\n\n")

    spark = (
        pyspark.sql.SparkSession.builder
        .appName("model_monitoring")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    monitoring_start = pd.to_datetime(monitoring_start)
    monitoring_end = pd.to_datetime(monitoring_end)

    # load gold predictions for the given monitoring period
    prediction_files = []

    for d in pd.date_range(monitoring_start, monitoring_end, freq="MS"):
        date_str = d.strftime("%Y_%m_%d")
        prediction_files.append(
            os.path.join(
                prediction_path,
                f"champion_model_predictions_{date_str}.parquet"
            )
        )

    print("Loading prediction files:")
    for p in prediction_files:
        print(p)

    pred_sdf = spark.read.parquet(*prediction_files)

    pred_df = pred_sdf.toPandas()

    pred_df["snapshot_date"] = pd.to_datetime(pred_df["snapshot_date"])
    pred_df["Customer_ID"] = pred_df["Customer_ID"].astype(str).str.strip()

    pred_df = pred_df[
        (pred_df["snapshot_date"] >= monitoring_start)
        & (pred_df["snapshot_date"] <= monitoring_end)
    ].copy()

    print("Prediction rows:", pred_df.shape)
    print("Prediction months:", sorted(pred_df["snapshot_date"].dt.strftime("%Y-%m-%d").unique()))

    # load actual labels files for the given monitoring period
    label_files = []

    for d in pd.date_range(monitoring_start, monitoring_end, freq="MS"):
        label_date = d + pd.DateOffset(months=6)
        date_str = label_date.strftime("%Y_%m_%d")

        label_files.append(
            os.path.join(
                label_store_path,
                f"gold_risk_performance_{date_str}.parquet"
            )
        )

    print("Loading label files:")
    for p in label_files:
        print(p)

    label_sdf = spark.read.parquet(*label_files)

    label_df = label_sdf.toPandas()

    label_df["snapshot_date"] = pd.to_datetime(label_df["snapshot_date"])
    label_df["Customer_ID"] = label_df["Customer_ID"].astype(str).str.strip()

    # Your label snapshot_date is 6 months after the feature snapshot date.
    # Therefore, to monitor Jul24 predictions, actual label date is Jan25.
    label_df["feature_snapshot_date"] = label_df["snapshot_date"] - pd.DateOffset(months=6)

    actual_df = (
        label_df
        .groupby(["Customer_ID", "feature_snapshot_date"], as_index=False)
        .agg(actual_label=("label", "max"))
    )

    print("Actual label rows:", actual_df.shape)

    # join predictions with actual labels to create a monitoring base table
    monitor_base = pred_df.merge(
        actual_df,
        left_on=["Customer_ID", "snapshot_date"],
        right_on=["Customer_ID", "feature_snapshot_date"],
        how="left",
    )

    monitor_base = monitor_base.drop(columns=["feature_snapshot_date"], errors="ignore")

    print("Rows after prediction-actual join:", monitor_base.shape)
    print("Actual label distribution including missing:")
    print(monitor_base["actual_label"].value_counts(dropna=False))

    # Keep only rows where actual outcome is available.
    monitor_eval = monitor_base.dropna(subset=["actual_label"]).copy()
    monitor_eval["actual_label"] = monitor_eval["actual_label"].astype(int)
    monitor_eval["pred_label"] = monitor_eval["pred_label"].astype(int)

    print("Rows available for performance monitoring:", monitor_eval.shape)

    # monthly performance metrics
    monthly_rows = []

    for snapshot_date, group in monitor_eval.groupby("snapshot_date"):
        y_true = group["actual_label"]
        y_pred = group["pred_label"]
        y_score = group["pred_proba"]

        # ROC-AUC requires both classes to exist in that month.
        if y_true.nunique() == 2:
            auc = roc_auc_score(y_true, y_score)
        else:
            auc = np.nan

        monthly_rows.append({
            "snapshot_date": snapshot_date,
            "row_count": len(group),
            "actual_default_rate": y_true.mean(),
            "predicted_default_rate": y_pred.mean(),
            "avg_pred_proba": y_score.mean(),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": auc,
        })

    performance_df = pd.DataFrame(monthly_rows).sort_values("snapshot_date")


    # 5. Calculate stability metrics (PSI) vs baseline month
    stability_rows = []

    months = sorted(pred_df["snapshot_date"].dropna().unique())
    if len(months) > 0:
        baseline_month = months[0]
        baseline_scores = pred_df.loc[pred_df["snapshot_date"] == baseline_month, "pred_proba"]

        for snapshot_date, group in pred_df.groupby("snapshot_date"):
            scores = group["pred_proba"]
            stability_rows.append({
                "snapshot_date": snapshot_date,
                "prediction_count": len(group),
                "pred_proba_mean": scores.mean(),
                "pred_proba_std": scores.std(),
                "pred_proba_p10": scores.quantile(0.10),
                "pred_proba_p50": scores.quantile(0.50),
                "pred_proba_p90": scores.quantile(0.90),
                "prediction_psi_vs_baseline": calculate_psi(baseline_scores, scores),
                "baseline_month": baseline_month,
            })

    stability_df = pd.DataFrame(stability_rows).sort_values("snapshot_date")

    # 6. Combine monitoring results
    monitoring_df = performance_df.merge(
        stability_df,
        on="snapshot_date",
        how="outer",
    ).sort_values("snapshot_date")

    monitoring_df["monitoring_run_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Governance Rules
    monitoring_df["retrain_flag"] = 0

    monitoring_df.loc[
        (monitoring_df["roc_auc"] < 0.70) |
        (monitoring_df["prediction_psi_vs_baseline"] > 0.25),
        "retrain_flag"
    ] = 1

    monitoring_df["governance_reason"] = np.where(
        monitoring_df["retrain_flag"] == 1,
        "Performance degradation or drift detected",
        "Model healthy"
    )

    print("\nMonitoring results:")
    print(monitoring_df)

    # 7. Save gold monitoring table
    os.makedirs(output_path, exist_ok=True)

    output_parquet = os.path.join(output_path, "model_monitoring.parquet")
    output_csv = os.path.join(output_path, "model_monitoring.csv")

    monitoring_sdf = spark.createDataFrame(monitoring_df)
    monitoring_sdf.write.mode("overwrite").parquet(output_parquet)
    monitoring_df.to_csv(output_csv, index=False)

    print("Saved monitoring parquet to:", output_parquet)
    print("Saved monitoring csv to:", output_csv)

    spark.stop()

    print("\n\n--- completed model monitoring job ---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run model monitoring")

    parser.add_argument(
        "--predictionpath",
        type=str,
        default="datamart/gold/model_predictions/champion_model/",
        help="Path to gold model predictions parquet folder",
    )

    parser.add_argument(
        "--labelstorepath",
        type=str,
        default="datamart/gold/label_store/",
        help="Path to gold label store parquet folder",
    )

    parser.add_argument(
        "--outputpath",
        type=str,
        default="datamart/gold/model_monitoring/champion_model/",
        help="Output path for gold monitoring table",
    )

    parser.add_argument(
        "--monitoringstart",
        type=str,
        default="2024-07-01",
        help="Monitoring start snapshot date, YYYY-MM-DD",
    )

    parser.add_argument(
        "--monitoringend",
        type=str,
        default="2024-12-01",
        help="Monitoring end snapshot date, YYYY-MM-DD",
    )

    args = parser.parse_args()

    main(
        prediction_path=args.predictionpath,
        label_store_path=args.labelstorepath,
        output_path=args.outputpath,
        monitoring_start=args.monitoringstart,
        monitoring_end=args.monitoringend,
    )
