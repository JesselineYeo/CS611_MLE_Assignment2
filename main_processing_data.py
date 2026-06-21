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

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_tables
import utils.data_processing_silver_table
import utils.data_processing_gold_table


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2025-11-01"

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)

##### REFRESH #####
# create bronze datalake
bronze_lms_directory = "datamart/bronze/lms/"
bronze_attributes_directory = "datamart/bronze/features_attributes/"
bronze_financials_directory = "datamart/bronze/features_financials/"
bronze_clickstream_directory = "datamart/bronze/features_clickstream/"

for directory in [
    bronze_lms_directory,
    bronze_attributes_directory,
    bronze_financials_directory,
    bronze_clickstream_directory,
]:
    if not os.path.exists(directory):
        os.makedirs(directory)

##### REFRESH #####
# run bronze backfill
for date_str in dates_str_lst:
    utils.data_processing_bronze_tables.process_bronze_table(
        date_str,
        bronze_lms_directory,
        bronze_attributes_directory,
        bronze_financials_directory,
        bronze_clickstream_directory,
        spark
    )

##### REFRESH #####
# create silver datalake
silver_loan_daily_directory = "datamart/silver/loan_daily/"
silver_profile_directory = "datamart/silver/profile/"
silver_clickstream_directory = "datamart/silver/features_clickstream/"

for directory in [
    silver_loan_daily_directory,
    silver_profile_directory,
    silver_clickstream_directory,
]:
    if not os.path.exists(directory):
        os.makedirs(directory)

##### REFRESH #####
# run silver backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_silver_table(
        date_str,
        bronze_lms_directory,
        bronze_attributes_directory,
        bronze_financials_directory,
        bronze_clickstream_directory,
        silver_loan_daily_directory,
        silver_profile_directory,
        silver_clickstream_directory,
        spark
    )

##### REFRESH #####
# create gold datalake
# gold_risk_profile_directory = "datamart/gold/risk_profile/"
# gold_risk_behaviour_directory = "datamart/gold/risk_behaviour/"
gold_feature_store_directory = "datamart/gold/feature_store/"
gold_risk_performance_directory = "datamart/gold/label_store/"

for directory in [
    # gold_risk_profile_directory,
    # gold_risk_behaviour_directory,
    gold_feature_store_directory,
    gold_risk_performance_directory,
]:
    if not os.path.exists(directory):
        os.makedirs(directory)

##### REFRESH #####
# run gold backfill
for date_str in dates_str_lst:
    if date_str >= "2025-01-01":
        utils.data_processing_gold_table.process_gold_loan_daily(
            date_str,
            silver_loan_daily_directory,
            gold_risk_performance_directory,
            spark,
            dpd=30,
            mob=6
        )
    else:
        utils.data_processing_gold_table.process_gold_table(
            date_str, 
            silver_loan_daily_directory,
            silver_profile_directory,
            silver_clickstream_directory,
            gold_feature_store_directory,
            gold_risk_performance_directory,
            spark, 
            dpd=30, 
            mob=6
        )

# # risk profile
# profile_files = [
#     gold_risk_profile_directory + os.path.basename(f)
#     for f in glob.glob(os.path.join(gold_risk_profile_directory, "*"))
# ]

# df_profile = spark.read.parquet(*profile_files)
# print("row_count:",df_profile.count())

# df_profile.show()

# # risk behaviour
# behaviour_files = [
#     gold_risk_behaviour_directory + os.path.basename(f)
#     for f in glob.glob(os.path.join(gold_risk_behaviour_directory, "*"))
# ]

# df_behaviour = spark.read.parquet(*behaviour_files)
# print("row_count:",df_behaviour.count())

# df_behaviour.show()

# feature store
feature_files = [
    gold_feature_store_directory + os.path.basename(f)
    for f in glob.glob(os.path.join(gold_feature_store_directory, "*"))
]

df_feature= spark.read.parquet(*feature_files)
print("row_count:",df_feature.count())

df_feature.show()

# label store
label_files = [
    gold_risk_performance_directory + os.path.basename(f)
    for f in glob.glob(os.path.join(gold_risk_performance_directory, "*"))
]

df_label = spark.read.parquet(*label_files)
print("row_count:",df_label.count())

df_label.show()


    