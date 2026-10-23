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
# config
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
    "raw/synthetic_data/transactions/transactions.csv"
)

CUSTOMERS_SILVER_PATH = (
    "s3://finsight-bank-data/"
    "silver/customers/"
)

ACCOUNTS_SILVER_PATH = (
    "s3://finsight-bank-data/"
    "silver/accounts/"
)

SILVER_PATH = (
    "s3://finsight-bank-data/"
    "silver/transactions/"
)

QUARANTINE_PATH = (
    "s3://finsight-bank-data/"
    "quarantine/transactions/"
)

PIPELINE_VERSION = "1.0"


# reading raw transactions

print("Reading RAW transactions...")

transactions = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(RAW_PATH)
)

raw_count = transactions.count()

print(f"RAW transaction records: {raw_count}")


#reading silvertables

print("Reading Silver customers...")

customers = spark.read.parquet(
    CUSTOMERS_SILVER_PATH
)

print("Reading Silver accounts...")

accounts = spark.read.parquet(
    ACCOUNTS_SILVER_PATH
)

#datatype conversion

transactions = (
    transactions
    .withColumn(
        "transaction_id",
        col("transaction_id").cast("long")
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
        "transaction_timestamp",
        col("transaction_timestamp").cast("timestamp")
    )
    .withColumn(
        "amount",
        col("amount").cast("decimal(14,2)")
    )
)


#creating ref keys

customer_keys = (
    customers
    .select(
        col("customer_id").alias("customer_key")
    )
    .dropDuplicates()
)




account_keys = (
    accounts
    .select(
        col("account_id").alias("account_key"),
        col("customer_id").alias(
            "account_customer_key"
        )
    )
    .dropDuplicates()
)

# ref integ

print("Checking referential integrity...")


checked = (
    transactions.alias("t")

    .join(
        customer_keys.alias("c"),
        col("t.customer_id")
        == col("c.customer_key"),
        "left"
    )

    .join(
        account_keys.alias("a"),
        col("t.account_id")
        == col("a.account_key"),
        "left"
    )

    .select(
        col("t.transaction_id"),
        col("t.customer_id"),
        col("t.account_id"),
        col("t.transaction_timestamp"),
        col("t.amount"),

  
        col("c.customer_key").alias(
            "matched_customer_id"
        ),

        col("a.account_key").alias(
            "matched_account_id"
        ),

        
        col("a.account_customer_key").alias(
            "account_customer_id"
        )
    )
)




print("Running data quality checks...")

# data validation

invalid_df = checked.filter(

    col("transaction_id").isNull()

    | col("customer_id").isNull()

    | col("account_id").isNull()

    | col("matched_customer_id").isNull()

    | col("matched_account_id").isNull()

    | (
        col("account_customer_id")
        != col("customer_id")
    )

  

    | col("amount").isNull()

    | (col("amount") <= 0)



    | col("transaction_timestamp").isNull()

    | (
        col("transaction_timestamp")
        > lit("2026-01-01")
        .cast("timestamp")
    )
)


invalid_count = invalid_df.count()

print(
    f"Invalid / referentially invalid "
    f"transactions: {invalid_count}"
)


#quantining invalid records

if invalid_count > 0:

    print("Writing invalid transactions to quarantine...")

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

    print("Invalid transactions quarantined.")


#keeping valid transactions

valid_df = checked.filter(


    col("transaction_id").isNotNull()

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


    & col("amount").isNotNull()

    & (col("amount") > 0)


    & col("transaction_timestamp").isNotNull()

    & (
        col("transaction_timestamp")
        <= lit("2026-01-01")
        .cast("timestamp")
    )
)


#removing deplicate

before_dedup = valid_df.count()

valid_df = valid_df.dropDuplicates(
    ["transaction_id"]
)

after_dedup = valid_df.count()

duplicates_removed = (
    before_dedup - after_dedup
)

print(
    f"Duplicates removed: "
    f"{duplicates_removed}"
)


#remove validation columns

valid_df = valid_df.drop(
    "matched_customer_id",
    "matched_account_id",
    "account_customer_id"
)


# metadata

silver_df = (
    valid_df

    .withColumn(
        "ingestion_timestamp",
        current_timestamp()
    )

    .withColumn(
        "source_system",
        lit(
            "synthetic_banking_transactions"
        )
    )

    .withColumn(
        "pipeline_version",
        lit(PIPELINE_VERSION)
    )
)


#writing to silver

print("Writing Silver transactions...")

(
    silver_df
    .write
    .mode("overwrite")
    .parquet(SILVER_PATH)
)



silver_count = silver_df.count()


print("TRANSACTIONS PIPELINE COMPLETE")

print(f"RAW records:         {raw_count}")
print(f"Invalid records:     {invalid_count}")
print(f"Duplicates removed:  {duplicates_removed}")
print(f"SILVER records:      {silver_count}")
print(f"SILVER location:     {SILVER_PATH}")


job.commit()
