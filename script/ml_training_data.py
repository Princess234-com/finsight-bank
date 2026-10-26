import sys

from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.sql.functions import (
    col,
    when,
    max as spark_max,
    count,
    sum as spark_sum,
    lit,
    current_timestamp
)


# ============================================================
# 1. INITIALISE GLUE JOB
# ============================================================
#config
args = getResolvedOptions(
    sys.argv,
    ["JOB_NAME"]
)

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

job = Job(glueContext)
job.init(args["JOB_NAME"], args)

ML_FEATURES_PATH = (
    "s3://finsight-bank-data/"
    "gold/ml_features/"
)

LOANS_PATH = (
    "s3://finsight-bank-data/"
    "silver/loans/"
)

OUTPUT_PATH = (
    "s3://finsight-bank-data/"
    "gold/ml_training_data/"
)

PIPELINE_VERSION = "1.0"

#readign features

print("Reading Gold ML features...")

ml_features = spark.read.parquet(
    ML_FEATURES_PATH
)

print(
    f"ML feature records: "
    f"{ml_features.count()}"
)


print("Reading Silver loans...")

loans = spark.read.parquet(
    LOANS_PATH
)

print(
    f"Loan records: "
    f"{loans.count()}"
)

# 5. CREATE CUSTOMER-LEVEL LOAN TARGET
#
# Target:
#
# 1 = customer has at least one Late loan
# 0 = customer has no Late loans
#
# IMPORTANT:
#
# We DO NOT pass late_loans, active_loans,
# paid_off_loans etc. into the ML features.
#
# Those variables are derived from the target/outcome
# and would cause target leakage.

print("Creating customer-level loan target...")


loan_targets = (
    loans
    .groupBy("customer_id")
    .agg(
        spark_max(
            when(
                col("status") == "Late",
                1
            ).otherwise(0)
        ).alias("loan_late_target"),

        count("*").alias(
            "loan_count_for_validation"
        )
    )
)

#check target distribution

print("Checking target distribution...")

target_distribution = (
    loan_targets
    .groupBy("loan_late_target")
    .count()
    .orderBy("loan_late_target")
)

target_distribution.show()


#join ML features with target
#
# Only customers with loans can have a meaningful
# loan-risk target.
#
# Therefore we use an INNER JOIN.

print(
    "Joining ML features with loan target..."
)


training_data = (
    ml_features
    .join(
        loan_targets,
        ml_features.customer_id
        == loan_targets.customer_id,
        "inner"
    )
    .select(
        ml_features["*"],

        loan_targets["loan_late_target"]
    )
)



# remove metadata as its not necessary

training_data = (
    training_data
    .drop(
        "ingestion_timestamp",
        "source_system",
        "pipeline_version",
        "gold_created_timestamp"
    )
)


# QA checks

print("Running ML training data checks...")


#duplicates checking
duplicate_customers = (
    training_data
    .groupBy("customer_id")
    .count()
    .filter(col("count") > 1)
    .count()
)

print(
    f"Customers with duplicate training rows: "
    f"{duplicate_customers}"
)

# Check null customer IDs


null_customer_ids = (
    training_data
    .filter(col("customer_id").isNull())
    .count()
)

print(
    f"Rows with null customer_id: "
    f"{null_customer_ids}"
)


# Check target values
invalid_targets = (
    training_data
    .filter(
        ~col("loan_late_target").isin(0, 1)
    )
    .count()
)

print(
    f"Rows with invalid target values: "
    f"{invalid_targets}"
)

#creating ML dataset

training_data = (
    training_data

    .withColumn(
        "training_dataset_created_at",
        current_timestamp()
    )

    .withColumn(
        "training_dataset_version",
        lit(PIPELINE_VERSION)
    )
)


#writng to training data

print(
    "Writing Gold ML training data..."
)

(
    training_data
    .write
    .mode("overwrite")
    .parquet(OUTPUT_PATH)
)

final_count = training_data.count()

print("ML TRAINING DATA PIPELINE COMPLETE")
print(
    f"ML feature records:       "
    f"{ml_features.count()}"
)

print(
    f"Loan target customers:     "
    f"{loan_targets.count()}"
)

print(
    f"Final training records:    "
    f"{final_count}"
)

print(
    f"Duplicate customers:       "
    f"{duplicate_customers}"
)

print(
    f"Null customer IDs:         "
    f"{null_customer_ids}"
)

print(
    f"Invalid targets:           "
    f"{invalid_targets}"
)

print(
    f"Output location:           "
    f"{OUTPUT_PATH}"
)


job.commit()
