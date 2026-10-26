import sys

from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.sql.functions import (
    col,
    lit,
    when,
    datediff,
    current_timestamp,
    coalesce,
    round,
    max as spark_max
)


# ============================================================
# 1. INITIALISE GLUE JOB
# ============================================================

args = getResolvedOptions(
    sys.argv,
    ["JOB_NAME"]
)

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

job = Job(glueContext)
job.init(args["JOB_NAME"], args)

#config

CUSTOMERS_SILVER_PATH = (
    "s3://finsight-bank-data/"
    "silver/customers/"
)

CUSTOMER_SPENDING_GOLD_PATH = (
    "s3://finsight-bank-data/"
    "gold/customer_spending/"
)

GOLD_PATH = (
    "s3://finsight-bank-data/"
    "gold/ml_features/"
)

PIPELINE_VERSION = "1.1"


#read silver customer

print("==========================================")
print("GOLD ML FEATURE PIPELINE")
print("==========================================")

print("Reading Silver customers...")

customers = spark.read.parquet(
    CUSTOMERS_SILVER_PATH
)

customer_count = customers.count()

print(
    f"Silver customers: {customer_count}"
)

#reading gold customer spending

print("Reading Gold customer spending...")

customer_spending = spark.read.parquet(
    CUSTOMER_SPENDING_GOLD_PATH
)

customer_spending_count = (
    customer_spending.count()
)

print(
    "Gold customer spending records: "
    f"{customer_spending_count}"
)

# determine feature as-of date
# We use the latest observed transaction date in the dataset
# rather than current_date().
#
# This makes the ML feature snapshot reproducible.
#
# Example:
# If the latest transaction in the dataset is 2025-08-12,
# all recency calculations are relative to 2025-08-12.
#
# Customers with no transactions are still retained.

print("Determining feature as-of date...")

feature_as_of_date = (
    customer_spending
    .select(
        spark_max("last_transaction_date")
        .alias("feature_as_of_date")
    )
    .collect()[0]["feature_as_of_date"]
)

if feature_as_of_date is None:
    raise ValueError(
        "Could not determine feature_as_of_date "
        "because no transaction dates were found."
    )

print(
    f"Feature as-of date: {feature_as_of_date}"
)

#prepare customer attributes

print("Preparing customer attributes...")

customer_dim = (
    customers
    .select(
        col("customer_id")
        .cast("long")
        .alias("customer_id"),

        *[
            col(c)
            for c in customers.columns
            if c != "customer_id"
        ]
    )
    .dropDuplicates(["customer_id"])
)

#select customer spending features

print("Preparing customer spending features...")

spending_features = (
    customer_spending
    .select(
        col("customer_id")
        .cast("long")
        .alias("customer_id"),

        # Monetary measures
        col("total_spending")
        .cast("decimal(18,2)")
        .alias("total_spending"),

        col("transaction_count")
        .cast("long")
        .alias("transaction_count"),

        col("active_account_count")
        .cast("long")
        .alias("active_account_count"),

        col("average_transaction_amount")
        .cast("decimal(18,2)")
        .alias("average_transaction_amount"),

        col("spending_per_transaction")
        .cast("decimal(18,2)")
        .alias("spending_per_transaction"),

        # Transaction timing
        col("first_transaction_timestamp"),
        col("last_transaction_timestamp"),
        col("first_transaction_date"),
        col("last_transaction_date"),

        # Existing business segmentation
        col("spending_segment")
        .alias("source_spending_segment")
    )
    .dropDuplicates(["customer_id"])
)

#join customer and spending features

print(
    "Joining customer attributes with "
    "spending features..."
)

ml_features = (
    customer_dim.alias("c")
    .join(
        spending_features.alias("s"),
        col("c.customer_id")
        == col("s.customer_id"),
        "left"
    )
    .select(
        col("c.customer_id"),

        # Customer attributes

        *[
            col(f"c.{c}").alias(c)
            for c in customer_dim.columns
            if c != "customer_id"
        ],
        # Spending features
        col("s.total_spending"),
        col("s.transaction_count"),
        col("s.active_account_count"),
        col("s.average_transaction_amount"),
        col("s.spending_per_transaction"),

        # Transaction timing
        col("s.first_transaction_timestamp"),
        col("s.last_transaction_timestamp"),
        col("s.first_transaction_date"),
        col("s.last_transaction_date"),

        # Existing spending segment
        col("s.source_spending_segment")
    )
)


#handling customer without txn (transaction)
print(
    "Handling customers without "
    "transaction activity..."
)

ml_features = (
    ml_features

    # Monetary measures
    .withColumn(
        "total_spending",
        coalesce(
            col("total_spending"),
            lit(0).cast("decimal(18,2)")
        )
    )

    .withColumn(
        "average_transaction_amount",
        coalesce(
            col("average_transaction_amount"),
            lit(0).cast("decimal(18,2)")
        )
    )

    .withColumn(
        "spending_per_transaction",
        coalesce(
            col("spending_per_transaction"),
            lit(0).cast("decimal(18,2)")
        )
    )
#counts

    .withColumn(
        "transaction_count",
        coalesce(
            col("transaction_count"),
            lit(0).cast("long")
        )
    )

    .withColumn(
        "active_account_count",
        coalesce(
            col("active_account_count"),
            lit(0).cast("long")
        )
    )
)


#transaction observation span
#
# The original field was called customer_lifetime_days.
#
#the calculation was:
# last_transaction_date - first_transaction_date
#
#represents the span of observed transaction activity
print(
    "Calculating transaction observation span..."
)

ml_features = (
    ml_features

    .withColumn(
        "transaction_observation_span_days",

        when(
            col("first_transaction_date").isNotNull()
            & col("last_transaction_date").isNotNull(),

            datediff(
                col("last_transaction_date"),
                col("first_transaction_date")
            )
        )

        .otherwise(lit(0))
        .cast("long")
    )
)


#transaction frequency per month


print(
    "Calculating transaction frequency..."
)

ml_features = (
    ml_features

    .withColumn(
        "transaction_frequency_per_month",

        when(
            col("transaction_observation_span_days") > 0,

            (
                col("transaction_count").cast("double")
                /
                (
                    col(
                        "transaction_observation_span_days"
                    ).cast("double")
                    / lit(30.44)
                )
            )
        )

        .otherwise(
            col("transaction_count").cast("double")
        )
    )

    .withColumn(
        "transaction_frequency_per_month",

        round(
            col("transaction_frequency_per_month"),
            6
        )
    )
)

#transaction activity flag

print(
    "Creating transaction activity flag..."
)

ml_features = (
    ml_features

    .withColumn(
        "transaction_activity_flag",

        when(
            col("transaction_count") > 0,
            lit(1)
        )

        .otherwise(lit(0))
    )
)

#high average transaction flag
#     "customer's average transaction amount >= 500"
#
# Therefore the feature name has been changed to reflect
# the actual calculation.
# ============================================================

print(
    "Creating high-average transaction flag..."
)

ml_features = (
    ml_features

    .withColumn(
        "high_average_transaction_flag",

        when(
            col("average_transaction_amount") >= 500,
            lit(1)
        )

        .otherwise(lit(0))
    )
)


#customer activity recency
#
# We use the dataset feature_as_of_date instead of
# current_date().
#
# Customers with no transaction history receive NULL.
#
# NULL is preferable to -1 because -1 is an artificial
# numeric value that could accidentally be interpreted as
# a real measurement by an ML algorithm.


print(
    "Calculating customer activity recency..."
)

ml_features = (
    ml_features

    .withColumn(
        "days_since_last_transaction",

        when(
            col("last_transaction_date").isNotNull(),

            datediff(
                lit(feature_as_of_date),
                col("last_transaction_date")
            )
        )

        .otherwise(lit(None).cast("long"))
    )
)

#customer activity status
print(
    "Creating customer activity status..."
)

ml_features = (
    ml_features

    .withColumn(
        "customer_activity_status",

        when(
            col("transaction_count") == 0,
            lit("NO_ACTIVITY")
        )

        .when(
            col("days_since_last_transaction") <= 30,
            lit("ACTIVE")
        )

        .when(
            col("days_since_last_transaction") <= 90,
            lit("AT_RISK")
        )

        .otherwise(
            lit("INACTIVE")
        )
    )
)

#spending per active account
# Original name:
#     spending_intensity
# Actual formula:
#     total_spending / active_account_count
# We rename it so the feature meaning is immediately clear.


print(
    "Creating spending per active account..."
)

ml_features = (
    ml_features

    .withColumn(
        "spending_per_active_account",

        when(
            col("active_account_count") > 0,

            (
                col("total_spending").cast("double")
                /
                col("active_account_count").cast("double")
            )
        )

        .otherwise(lit(0.0))
    )

    .withColumn(
        "spending_per_active_account",

        round(
            col("spending_per_active_account"),
            6
        )
    )
)

#transaction per active account 

print(
    "Creating transactions per active account..."
)

ml_features = (
    ml_features

    .withColumn(
        "transactions_per_active_account",

        when(
            col("active_account_count") > 0,

            (
                col("transaction_count").cast("double")
                /
                col("active_account_count").cast("double")
            )
        )

        .otherwise(lit(0.0))
    )

    .withColumn(
        "transactions_per_active_account",

        round(
            col("transactions_per_active_account"),
            6
        )
    )
)


#spending segment
# A customer who does not appear in customer_spending because they have no transactions will receive NULL after the LEFT JOIN.
#
#     NO_SPEND = 0
#     LOW      = >0 and <1000
#     MEDIUM   = 1000 to <5000
#     PREMIUM  = >=5000
# This is preferable to silently treating "no activity" as ordinary low spending.
print(
    "Creating final spending segment..."
)

ml_features = (
    ml_features

    .withColumn(
        "spending_segment",

        when(
            col("total_spending") == 0,
            lit("NO_SPEND")
        )

        .when(
            col("total_spending") < 1000,
            lit("LOW")
        )

        .when(
            col("total_spending") < 5000,
            lit("MEDIUM")
        )

        .otherwise(
            lit("PREMIUM")
        )
    )
)

#final feature selection
print(
    "Selecting final ML features..."
)

ml_features = (
    ml_features

    .select(

        # Customer identifier
        col("customer_id"),

        # Customer attributes
        *[
            col(c)
            for c in customer_dim.columns
            if c != "customer_id"
        ],
      
        # Core spending features
        col("total_spending"),
        col("transaction_count"),
        col("active_account_count"),
        col("average_transaction_amount"),
        col("spending_per_transaction"),
      
        # Transaction timing
        col("first_transaction_timestamp"),
        col("last_transaction_timestamp"),
        col("first_transaction_date"),
        col("last_transaction_date"),

        col(
            "transaction_observation_span_days"
        ),
        # Behavioural features
        col("transaction_frequency_per_month"),
        col("transaction_activity_flag"),
        col("high_average_transaction_flag"),
        col("days_since_last_transaction"),
        col("customer_activity_status"),
        col("spending_per_active_account"),
        col("transactions_per_active_account"),

        # Spending segmentation
        col("spending_segment")
    )
)


#meta data bits
ml_features = (
    ml_features

    .withColumn(
        "feature_as_of_date",
        lit(feature_as_of_date)
    )
)

#remove duplicate customers

print(
    "Removing duplicate customers..."
)

ml_features = (
    ml_features
    .dropDuplicates(["customer_id"])
)
# final data quality checks

print(
    "Running final ML feature quality checks..."
)

# handling NULL customer IDs

invalid_customer_ids = (
    ml_features
    .filter(
        col("customer_id").isNull()
    )
    .count()
)

print(
    "Records with NULL customer_id: "
    f"{invalid_customer_ids}"
)

if invalid_customer_ids > 0:

    raise ValueError(
        "ML feature dataset contains "
        "records with NULL customer_id."
    )

# handling duplicate customer id

duplicate_customer_ids = (
    ml_features
    .groupBy("customer_id")
    .count()
    .filter(col("count") > 1)
    .count()
)

print(
    "Duplicate customer IDs: "
    f"{duplicate_customer_ids}"
)

if duplicate_customer_ids > 0:

    raise ValueError(
        "ML feature dataset contains "
        "duplicate customer IDs."
    )

#negative monetary value handling

negative_spending = (
    ml_features
    .filter(
        col("total_spending") < 0
    )
    .count()
)

print(
    "Records with negative total spending: "
    f"{negative_spending}"
)

if negative_spending > 0:

    raise ValueError(
        "ML feature dataset contains "
        "negative total_spending values."
    )
#invalid txn count

negative_transaction_counts = (
    ml_features
    .filter(
        col("transaction_count") < 0
    )
    .count()
)

print(
    "Records with negative transaction count: "
    f"{negative_transaction_counts}"
)

if negative_transaction_counts > 0:

    raise ValueError(
        "ML feature dataset contains "
        "negative transaction_count values."
    )
#adding metadata

ml_features = (
    ml_features

    .withColumn(
        "pipeline_version",
        lit(PIPELINE_VERSION)
    )

    .withColumn(
        "gold_created_timestamp",
        current_timestamp()
    )

    .withColumn(
        "source_system",
        lit(
            "synthetic_banking_transactions"
        )
    )
)
# writing gold features
print(
    "Writing Gold ML features..."
)

(
    ml_features
    .write
    .mode("overwrite")
    .parquet(GOLD_PATH)
)


gold_count = ml_features.count()
print("GOLD ML FEATURE PIPELINE COMPLETE")
print(
    f"Silver customers:       {customer_count}"
)

print(
    f"Gold spending records:  "
    f"{customer_spending_count}"
)

print(
    f"Gold ML feature rows:   {gold_count}"
)

print(
    f"Number of ML features:  "
    f"{len(ml_features.columns)}"
)

print(
    f"Feature as-of date:     "
    f"{feature_as_of_date}"
)

print(
    f"Pipeline version:       "
    f"{PIPELINE_VERSION}"
)

print(
    f"Gold location:          "
    f"{GOLD_PATH}"
)


job.commit()
