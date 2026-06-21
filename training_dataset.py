import os
import pandas as pd
from dateutil.relativedelta import relativedelta

print("--- start building training dataset job ---")

feature_pd = pd.read_parquet("./datamart/gold/feature_store")
label_pd = pd.read_parquet("./datamart/gold/label_store")

# fill missing feature values with median
median_fill_cols = [
    "Age",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "inq_per_loan",
]

for col in median_fill_cols:
    print(f"{col} missing before:", feature_pd[col].isna().sum())
    feature_pd[col] = feature_pd[col].fillna(feature_pd[col].median())
    print(f"{col} missing after:", feature_pd[col].isna().sum())

# align label date back to feature date
label_pd["feature_snapshot_date"] = label_pd["snapshot_date"].apply(
    lambda x: x - relativedelta(months=6)
)

label_pd_1 = (
    label_pd
    .groupby(["Customer_ID", "feature_snapshot_date"], as_index=False)
    .agg(label=("label", "max"))
    .rename(columns={"feature_snapshot_date": "snapshot_date"})
)

# merge feature + label
merged_tb = feature_pd.merge(
    label_pd_1,
    on=["Customer_ID", "snapshot_date"],
    how="inner"
)

# remove negative outlier
merged_tb = merged_tb[
    merged_tb["Monthly_Balance"] > -1e10
].copy()

# remove redundant column
merged_tb = merged_tb.drop(columns=["has_credit_limit_change"])

# remove vector columns if your model_train.py does not need them
vec_cols = [
    "Credit_Mix_vec",
    "Payment_Behaviour_vec",
    "Payment_of_Min_Amount_vec",
    "Occupation_vec",
    "age_group_vec"
]

merged_tb = merged_tb.drop(columns=vec_cols)

os.makedirs("datamart/gold/training_dataset", exist_ok=True)

merged_tb.to_parquet(
    "datamart/gold/training_dataset/merged_tb.parquet",
    index=False
)

print("Saved to datamart/gold/training_dataset/merged_tb.parquet")
print("Final rows:", len(merged_tb))
print("--- completed build training dataset job ---")