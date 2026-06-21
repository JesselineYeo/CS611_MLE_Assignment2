import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import (
    StringType,
    IntegerType,
    FloatType,
    DoubleType,
    DateType
)
from pyspark.sql.window import Window


# Cleaning
# General functions


def clean_string_spark(column_name):
    return (
        F.when(
            F.lower(F.trim(col(column_name))).isin("_", "!@9#%8", "nan", "null", "none", ""),
            "unknown"
        )
        .otherwise(F.lower(F.trim(col(column_name).cast(StringType()))))
    )


def clean_numeric_string_spark(column_name):
    return (
        F.regexp_replace(
            F.trim(col(column_name).cast(StringType())),
            "_",
            ""
        ).cast(DoubleType())
    )


def replace_missing_values(df):
    missing_values = ["_______", "_", "", "null", "None", "NaN"]

    for c in df.columns:
        df = df.withColumn(
            c,
            F.when(col(c).cast(StringType()).isin(missing_values), None)
             .otherwise(col(c))
        )

    return df

# Attributes

def clean_features_attributes(df):
    df = replace_missing_values(df)

    df = df.withColumn("Customer_ID", clean_string_spark("Customer_ID"))

    df = df.withColumn("Age", clean_numeric_string_spark("Age"))

    df = df.withColumn(
        "Age",
        F.when((col("Age") < 0) | (col("Age") > 60), None)
         .otherwise(col("Age"))
         .cast(IntegerType())
    )

    df = df.withColumn(
        "Occupation",
        F.when(
            col("Occupation").isNull(),
            "unknown"
        ).otherwise(F.lower(F.trim(col("Occupation"))))
    )

    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    df = df.drop("Name", "SSN")

    return df

# Clickstream

def clean_features_clickstream(df):
    df = df.withColumn("Customer_ID", clean_string_spark("Customer_ID"))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    return df


# Financials

def clean_features_financials(df):
   
    df = df.withColumn("Customer_ID", clean_string_spark("Customer_ID"))
    df = df.withColumn("Payment_of_Min_Amount", clean_string_spark("Payment_of_Min_Amount"))

    numeric_cols = [
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Num_of_Loan",
        "Num_of_Delayed_Payment",
        "Changed_Credit_Limit",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Total_EMI_per_month",
        "Amount_invested_monthly",
        "Monthly_Balance"
    ]

    for c in numeric_cols:
        df = df.withColumn(c, clean_numeric_string_spark(c))

    decimal_cols = [
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Total_EMI_per_month",
        "Amount_invested_monthly",
        "Monthly_Balance"
    ]

    for c in decimal_cols:
        df = df.withColumn(c, F.round(col(c), 3))

    df = df.withColumn(
        "Num_Bank_Accounts",
        F.when((col("Num_Bank_Accounts") < 0) | (col("Num_Bank_Accounts") > 15), None)
         .otherwise(col("Num_Bank_Accounts"))
    )

    df = df.withColumn(
        "Num_Credit_Card",
        F.when((col("Num_Credit_Card") < 0) | (col("Num_Credit_Card") > 15), None)
         .otherwise(col("Num_Credit_Card"))
    )

    df = df.withColumn(
        "Interest_Rate",
        F.when((col("Interest_Rate") < 0) | (col("Interest_Rate") > 50), None)
         .otherwise(col("Interest_Rate"))
    )

    df = df.withColumn(
        "Num_of_Loan",
        F.when((col("Num_of_Loan") < 0) | (col("Num_of_Loan") > 15), None)
         .otherwise(col("Num_of_Loan"))
    )

    df = df.withColumn(
        "Num_Credit_Inquiries",
        F.when(col("Num_Credit_Inquiries") > 20, None)
         .otherwise(col("Num_Credit_Inquiries"))
    )

    # Clean Type_of_Loan into array
    df = df.withColumn(
        "Type_of_Loan",
        F.lower(F.trim(col("Type_of_Loan")))
    )

    df = df.withColumn(
        "Type_of_Loan",
        F.regexp_replace(col("Type_of_Loan"), " and ", ", ")
    )

    df = df.withColumn(
        "Type_of_Loan",
        F.split(col("Type_of_Loan"), ",")
    )

    df = df.withColumn(
        "Type_of_Loan",
        F.expr("""
            array_distinct(
                filter(
                    transform(Type_of_Loan, x -> trim(x)),
                    x -> x != '' and x != 'not specified'
                )
            )
        """)
    )

    df = df.withColumn("Credit_Mix", clean_string_spark("Credit_Mix"))
    df = df.withColumn("Payment_Behaviour", clean_string_spark("Payment_Behaviour"))

    # Convert "22 Years and 3 Months" to total months
    df = df.withColumn(
        "Credit_History_Age",
        (
            F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Years", 1).cast(IntegerType()) * 12
            +
            F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Months", 1).cast(IntegerType())
        )
    )

    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    return df

# LMS Loan Daily

def clean_lms_loan_daily(df):
    df = df.withColumn("loan_id", clean_string_spark("loan_id"))
    df = df.withColumn("Customer_ID", clean_string_spark("Customer_ID"))

    # enforce schema / data types
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # sort logic using window
    window_spec = Window.partitionBy("loan_id") \
                        .orderBy("installment_num")

    # expected balance
    df = df.withColumn(
        "expected_balance",
        col("loan_amt") -
        F.sum("paid_amt").over(window_spec)
    )

    # balance validation
    df = df.withColumn(
        "balance_check",
        col("balance") == col("expected_balance")
    )

    print("Balance check:")
    df.groupBy("balance_check").count().show()

    # drop temp columns
    df = df.drop("expected_balance", "balance_check")

    # add mob
    df = df.withColumn(
        "mob",
        col("installment_num").cast(IntegerType())
    )

    # installments missed
    df = df.withColumn(
        "installments_missed",
        F.ceil(col("overdue_amt") / col("due_amt"))
    )

    df = df.fillna(
        {"installments_missed": 0}
    )

    # first missed date
    df = df.withColumn(
        "first_missed_date",
        F.when(
            col("installments_missed") > 0,
            F.add_months(
                col("snapshot_date"),
                -1 * col("installments_missed")
            )
        ).cast(DateType())
    )

    # dpd
    df = df.withColumn(
        "dpd",
        F.when(
            col("overdue_amt") > 0.0,
            F.datediff(
                col("snapshot_date"),
                col("first_missed_date")
            )
        ).otherwise(0).cast(IntegerType())
    )

    return df

###################################################################################################################

##### Processing #####

# def process_silver_customer_profile(
#         snapshot_date_str, 
#         bronze_attributes_directory, 
#         bronze_financials_directory,
#         silver_profile_directory, 
#         spark):

#     # connect to bronze table [attributes]
#     partition_name = "bronze_features_attributes_" + snapshot_date_str.replace("-", "_") + ".csv"
#     filepath = bronze_attributes_directory + partition_name
#     df_attributes = spark.read.csv(filepath, header=True, inferSchema=True)
#     print("attributes loaded from:", filepath, "row count:", df_attributes.count())

#     # apply cleaning [attributes]
#     df_attributes_clean = clean_features_attributes(df_attributes)

#     # connect to bronze table [financials]
#     partition_name = "bronze_features_financials_" + snapshot_date_str.replace("-", "_") + ".csv"
#     filepath = bronze_financials_directory + partition_name
#     df_financials = spark.read.csv(filepath, header=True, inferSchema=True)
#     print("financials loaded from:", filepath, "row count:", df_financials.count())

#     # apply cleaning [financials]
#     df_financials_clean = clean_features_financials(df_financials)

#     # join cleaned tables
#     df_profile = df_attributes_clean.join(
#         df_financials_clean,
#         on=["Customer_ID", "snapshot_date"],
#         how="left"
#     )

#     print("profile row count:", df_profile.count())

#     # save silver table
#     partition_name = "silver_customer_profile_" + snapshot_date_str.replace("-", "_") + ".parquet"
#     filepath = silver_profile_directory + partition_name
#     df_profile.write.mode("overwrite").parquet(filepath)
#     print("saved to:", filepath)

#     return df_profile

def process_silver_loan_daily(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace("-", "_") + ".csv"
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print("loaded from:", filepath, "row count:", df.count())

    # apply cleaning
    df = clean_lms_loan_daily(df)

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace("-", "_") + ".parquet"
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print("saved to:", filepath)

    return df

def process_silver_customer_profile(
        snapshot_date_str, 
        bronze_attributes_directory, 
        bronze_financials_directory,
        silver_profile_directory, 
        spark):

    # connect to bronze table [attributes]
    partition_name = "bronze_features_attributes_" + snapshot_date_str.replace("-", "_") + ".csv"
    filepath = bronze_attributes_directory + partition_name
    df_attributes = spark.read.csv(filepath, header=True, inferSchema=True)
    print("attributes loaded from:", filepath, "row count:", df_attributes.count())

    # apply cleaning [attributes]
    df_attributes_clean = clean_features_attributes(df_attributes)

    # connect to bronze table [financials]
    partition_name = "bronze_features_financials_" + snapshot_date_str.replace("-", "_") + ".csv"
    filepath = bronze_financials_directory + partition_name
    df_financials = spark.read.csv(filepath, header=True, inferSchema=True)
    print("financials loaded from:", filepath, "row count:", df_financials.count())

    # apply cleaning [financials]
    df_financials_clean = clean_features_financials(df_financials)

    # join tables
    df_profile = df_attributes_clean.join(
        df_financials_clean,
        on=["Customer_ID", "snapshot_date"],
        how="left"
        )

    print("profile row count: ", df_profile.count())

    # save silver table - IRL connect to database to write
    partition_name = "silver_customer_profile_" + snapshot_date_str.replace("-", "_") + ".parquet"
    filepath = silver_profile_directory + partition_name
    df_profile.write.mode("overwrite").parquet(filepath)
    print("saved to:", filepath)

    return df_profile

def process_silver_clickstream(snapshot_date_str, bronze_clickstream_directory, silver_clickstream_directory, spark):
    # connect to bronze table 
    partition_name = "bronze_features_clickstream_" + snapshot_date_str.replace("-", "_") + ".csv"
    filepath = bronze_clickstream_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print("loaded from:", filepath, "row count:", df.count())

    # apply cleaning
    df = clean_features_clickstream(df)

    # save silver table - IRL connect to database to write
    partition_name = "silver_features_clickstream_" + snapshot_date_str.replace("-", "_") + ".parquet"
    filepath = silver_clickstream_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print("saved to:", filepath)

    return df

def process_silver_table(
    snapshot_date_str,
    bronze_lms_directory,
    bronze_attributes_directory,
    bronze_financials_directory,
    bronze_clickstream_directory,
    silver_loan_daily_directory,
    silver_profile_directory,
    silver_clickstream_directory,
    spark
):
    process_silver_loan_daily(
        snapshot_date_str,
        bronze_lms_directory,
        silver_loan_daily_directory,
        spark
    )

    process_silver_customer_profile(
        snapshot_date_str,
        bronze_attributes_directory,
        bronze_financials_directory,
        silver_profile_directory,
        spark
    )

    process_silver_clickstream(
        snapshot_date_str,
        bronze_clickstream_directory,
        silver_clickstream_directory,
        spark
    )