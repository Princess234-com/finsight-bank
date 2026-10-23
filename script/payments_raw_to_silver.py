import sys

from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.sql.functions import (
    col,
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



RAW_PATH = (
    "s3://finsight-bank-data/"
    "raw/synthetic_data/payments/payments.csv"
)

CUSTOMERS_SILVER_PATH = (
    "s3://finsight-bank-data/silver/customers/"
)

ACCOUNTS_SILVER_PATH = (
    "s3://finsight-bank-data/silver/accounts/"
)

SILVER_PATH = (
    "s3://finsight-bank-data/silver/payments/"
)

QUARANTINE_PATH = (
    "s3://finsight-bank-data/"
    "quarantine/payments/"
)

PIPELINE_VERSION = "1.0"

#reading raw data

print("Reading RAW payments data...")

payments = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(RAW_PATH)
)

print("Reading Silver customers...")

customers = spark.read.parquet(
    CUSTOMERS_SILVER_PATH
)

print("Reading Silver accounts...")

accounts = spark.read.parquet(
    ACCOUNTS_SILVER_PATH
)

raw_count = payments.count()

print(f"RAW payment records: {raw_count}")


#datatye conversion

payments = (
    payments

    .withColumn(
        "payment_id",
        col("payment_id").cast("long")
    )

    .withColumn(
        "customer_id",
        col("customer_id").cast("long")
    )

    .withColumn(
        "account_id",
        col("account_id").cast("long")
    )

    .withColumn(
        "payment_date",
        col("payment_date").cast("date")
    )

    .withColumn(
        "amount",
        col("amount").cast("decimal(14,2)")
    )
)




customer_keys = (
    customers
    .select(
        col("customer_id")
        .alias("ref_customer_id")
    )
    .dropDuplicates()
)


account_keys = (
    accounts
    .select(
        col("account_id")
        .alias("ref_account_id"),

        col("customer_id")
        .alias("account_customer_id")
    )
    .dropDuplicates()
)


#data quality and ref integ

print("Checking customer referential integrity...")


checked = (
    payments

    .join(
        customer_keys,

        payments["customer_id"]
        == customer_keys["ref_customer_id"],

        "left"
    )

    .select(
        payments["*"],

        customer_keys["ref_customer_id"]
        .alias("matched_customer_id")
    )
)



print("Checking account referential integrity...")


checked = (
    checked

    .join(
        account_keys,

        checked["account_id"]
        == account_keys["ref_account_id"],

        "left"
    )

    .select(
        checked["*"],

        account_keys["ref_account_id"]
        .alias("matched_account_id"),

        account_keys["account_customer_id"]
        .alias("account_customer_id")
    )
)


#data quality and ref integ

print("Running data-quality checks...")


invalid_df = checked.filter(

    col("payment_id").isNull()

    | col("customer_id").isNull()

    | col("account_id").isNull()

    | col("matched_customer_id").isNull()


    | col("matched_account_id").isNull()

    | (
        col("account_customer_id")
        != col("customer_id")
    )

    | (col("amount") <= 0)

    | col("payment_date").isNull()

    | (
        col("payment_date")
        > lit("2026-01-01").cast("date")
    )
)


invalid_count = invalid_df.count()

print(
    f"Invalid / orphan payment records: "
    f"{invalid_count}"
)


#quantine records

if invalid_count > 0:

    print("Writing invalid records to quarantine...")

    (
        invalid_df

        .drop(
            "matched_customer_id",
            "matched_account_id",
            "account_customer_id"
        )

        .write

        .mode("append")

        .parquet(QUARANTINE_PATH)
    )

    print("Invalid records quarantined.")

#keep valid rec

valid_df = checked.filter(

    col("payment_id").isNotNull()

    & col("customer_id").isNotNull()

    & col("account_id").isNotNull()

    # Customer exists
    & col("matched_customer_id").isNotNull()

    # Account exists
    & col("matched_account_id").isNotNull()

    # Account belongs to customer
    & (
        col("account_customer_id")
        == col("customer_id")
    )

    # Valid amount
    & (col("amount") > 0)

    # Valid date
    & col("payment_date").isNotNull()

    & (
        col("payment_date")
        <= lit("2026-01-01").cast("date")
    )
)

#remove dub payment

before_dedup = valid_df.count()

valid_df = valid_df.dropDuplicates(
    ["payment_id"]
)

after_dedup = valid_df.count()

duplicates_removed = (
    before_dedup
    - after_dedup
)

print(
    f"Duplicates removed: "
    f"{duplicates_removed}"
)


#removing validation columns

valid_df = valid_df.drop(
    "matched_customer_id",
    "matched_account_id",
    "account_customer_id"
)


#adding metadata

silver_df = (
    valid_df

    .withColumn(
        "ingestion_timestamp",
        current_timestamp()
    )

    .withColumn(
        "source_system",
        lit(
            "synthetic_banking_payments"
        )
    )

    .withColumn(
        "pipeline_version",
        lit(PIPELINE_VERSION)
    )
)


#writing silver folder

print("Writing Silver payments...")


(
    silver_df

    .write

    .mode("overwrite")

    .parquet(SILVER_PATH)
)



silver_count = silver_df.count()


print("PAYMENTS PIPELINE COMPLETE")

print(f"RAW records:         {raw_count}")
print(f"Invalid records:     {invalid_count}")
print(f"Duplicates removed:  {duplicates_removed}")
print(f"SILVER records:      {silver_count}")
print(f"SILVER path:         {SILVER_PATH}")




job.commit()
