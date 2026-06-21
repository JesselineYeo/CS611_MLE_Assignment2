import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from pyspark.sql.functions import col, lit, when, size, split, regexp_extract, coalesce, log1p
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline


##### Processing #####

## label_store
def process_gold_loan_daily(snapshot_date_str, silver_loan_daily_directory, gold_risk_performance_directory, spark, dpd=30, mob=6):
    # connect to silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    # # label_def column acts as documentation and auditability. It is a string representation of the labeling rule used to create the label, e.g. "30dpd_6mob" means "label=1 if dpd>=30 at 6 mob, else 0".
    # df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    # df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")
    df = df.select("loan_id", "Customer_ID", "label", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_risk_performance_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_risk_performance_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


## feature store

# def process_gold_risk_profile(snapshot_date_str, silver_profile_directory, gold_risk_profile_directory, spark):
#     # connect to silver table
#     partition_name = "silver_customer_profile_" + snapshot_date_str.replace("-", "_") + ".parquet"
#     filepath = silver_profile_directory + partition_name
#     df = spark.read.parquet(filepath)
#     print('loaded from:', filepath, 'row count:', df.count())

#     # save gold table - IRL connect to database to write
#     partition_name = "gold_risk_profile_" + snapshot_date_str.replace('-','_') + '.parquet'
#     filepath = gold_risk_profile_directory + partition_name
#     df.write.mode("overwrite").parquet(filepath)
#     # df.toPandas().to_parquet(filepath,
#     #           compression='gzip')
#     print('saved to:', filepath)
    
#     return df

# def process_gold_risk_behaviour(snapshot_date_str, silver_clickstream_directory, gold_risk_behaviour_directory, spark):
#     # connect to silver table
#     partition_name = "silver_features_clickstream_" + snapshot_date_str.replace("-", "_") + ".parquet"
#     filepath = silver_clickstream_directory + partition_name
#     df = spark.read.parquet(filepath)
#     print('loaded from:', filepath, 'row count:', df.count())


#     # save gold table - IRL connect to database to write
#     partition_name = "gold_risk_behaviour_" + snapshot_date_str.replace('-','_') + '.parquet'
#     filepath = gold_risk_behaviour_directory + partition_name
#     df.write.mode("overwrite").parquet(filepath)
#     # df.toPandas().to_parquet(filepath,
#     #           compression='gzip')
#     print('saved to:', filepath)
    
#     return df


def cap_by_quantile(df, colname, lower_q=0.01, upper_q=0.95):
    q_low, q_high = df.approxQuantile(colname, [lower_q, upper_q], 0.01)

    return df.withColumn(
        colname,
        when(col(colname) < q_low, q_low)
        .when(col(colname) > q_high, q_high)
        .otherwise(col(colname))
    )


def process_gold_feature_store(
    snapshot_date_str,
    silver_profile_directory,
    silver_clickstream_directory,
    gold_feature_store_directory,
    spark
):

    # read silver profile
    partition_name = "silver_customer_profile_" + snapshot_date_str.replace("-", "_") + ".parquet"
    filepath = silver_profile_directory + partition_name
    profile = spark.read.parquet(filepath)
    print("loaded profile from:", filepath, "row count:", profile.count())

    # read silver clickstream / behaviour
    partition_name = "silver_features_clickstream_" + snapshot_date_str.replace("-", "_") + ".parquet"
    filepath = silver_clickstream_directory + partition_name
    behaviour = spark.read.parquet(filepath)
    print("loaded behaviour from:", filepath, "row count:", behaviour.count())

    # clean / impute categorical values
    profile = (
        profile
        .withColumn("Credit_Mix", coalesce(col("Credit_Mix"), lit("Unknown")))
        .withColumn("Payment_Behaviour", coalesce(col("Payment_Behaviour"), lit("Unknown")))
        .withColumn("Payment_of_Min_Amount", coalesce(col("Payment_of_Min_Amount"), lit("Unknown")))
        .withColumn("Occupation", coalesce(col("Occupation"), lit("Unknown")))
    )

    # add engineered features
    profile = (
        profile
        # Count how many different loan types a customer has.
        .withColumn(
            "num_loan_types", size(col("Type_of_Loan"))
        )
        # Key creditworthiness ratio 
        .withColumn(
            "debt_to_income",
            col("Outstanding_Debt") / (col("Annual_Income") + lit(1.0))
        )
        # Spending/investing habits normalized by income → reflect financial discipline
        .withColumn(
            "emi_to_salary",
            col("Total_EMI_per_month") / (col("Monthly_Inhand_Salary") + lit(1.0))
        )
        .withColumn(
            "investment_rate",
            col("Amount_invested_monthly") / (col("Monthly_Inhand_Salary") + lit(1.0))
        )
        # Binary flag for behavioral signal → indicates financial volatility
        .withColumn(
            "has_credit_limit_change",
            (col("Changed_Credit_Limit") != 0).cast(IntegerType())
        )
        # Encodes liquidity vs. debt and credit-seeking behavior
        .withColumn(
            "balance_to_debt",
            (col("Monthly_Balance") + lit(1.0)) / (col("Outstanding_Debt") + lit(1.0))
        )
        .withColumn(
            "inq_per_loan",
            col("Num_Credit_Inquiries") / (col("Num_of_Loan") + lit(1.0))
        )
        # bin age
        .withColumn(
            "age_group",
            when(col("Age") < 25, "<25")
            .when(col("Age") < 40, "25-39")
            .when(col("Age") < 60, "40-59")
            .otherwise("60+")
        )
    )

    # one-hot encode categorical columns
    cats = [
        "Credit_Mix",
        "Payment_Behaviour",
        "Payment_of_Min_Amount",
        "Occupation",
        "age_group"
    ]

    cats = [c for c in cats if c in profile.columns]

    idxs = [
        StringIndexer(
            inputCol=c,
            outputCol=c + "_idx",
            handleInvalid="keep"
        )
        for c in cats
    ]

    ohs = [
        OneHotEncoder(
            inputCol=c + "_idx",
            outputCol=c + "_vec"
        )
        for c in cats
    ]

    pipe = Pipeline(stages=idxs + ohs)
    profile = pipe.fit(profile).transform(profile)

    profile = profile.drop(*cats, *[c + "_idx" for c in cats])

    # cap clickstream outliers
    for i in range(1, 21):
        c = f"fe_{i}"
        if c in behaviour.columns:
            behaviour = cap_by_quantile(behaviour, c)

    # combine profile + behaviour into one feature store
    feature_store = profile.join(
        behaviour,
        on=["Customer_ID", "snapshot_date"],
        how="left"
    )

    # drop unnecessary / risky columns if present
    drop_cols = [
        "Name",
        "SSN",
        "history_years",
        "history_months"
    ]

    drop_cols = [c for c in drop_cols if c in feature_store.columns]
    feature_store = feature_store.drop(*drop_cols)

    # save gold feature store
    os.makedirs(gold_feature_store_directory, exist_ok=True)

    partition_name = "gold_feature_store_" + snapshot_date_str.replace("-", "_") + ".parquet"
    filepath = gold_feature_store_directory + partition_name

    feature_store.write.mode("overwrite").parquet(filepath)
    print("saved feature store:", filepath, "row count:", feature_store.count())

    return feature_store

def process_gold_table(
    snapshot_date_str,
    silver_loan_daily_directory,
    silver_profile_directory,
    silver_clickstream_directory,
    # gold_risk_profile_directory,
    # gold_risk_behaviour_directory,
    gold_feature_store_directory,
    gold_risk_performance_directory,
    spark,
    dpd=30,
    mob=6
):
    process_gold_loan_daily(
        snapshot_date_str,
        silver_loan_daily_directory,
        gold_risk_performance_directory,
        spark,
        dpd=dpd,
        mob=mob
    )

    # process_gold_risk_profile(
    #     snapshot_date_str,
    #     silver_profile_directory,
    #     gold_risk_profile_directory,
    #     spark
    # )

    # process_gold_risk_behaviour(
    #     snapshot_date_str,
    #     silver_clickstream_directory,
    #     gold_risk_behaviour_directory,
    #     spark
    # )

    process_gold_feature_store(
        snapshot_date_str,
        silver_profile_directory,
        silver_clickstream_directory,
        gold_feature_store_directory,
        spark
    )