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


RAW_PATH = "s3://finsight-bank-data/raw/synthetic_data/accounts/accounts.csv"

CUSTOMERS_SILVER_PATH = \
    "s3://finsight-bank-data/silver/customers/"

SILVER_PATH = \
    "s3://finsight-bank-data/silver/accounts/"

QUARANTINE_PATH = \
    "s3://finsight-bank-data/quarantine/accounts/"

PIPELINE_VERSION = "1.0"


#reading data

accounts = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(RAW_PATH)
)

customers = spark.read.parquet(
    CUSTOMERS_SILVER_PATH
)

raw_count = accounts.count()

print(f"RAW accounts: {raw_count}")


# casting datatype

accounts = (
    accounts
    .withColumn(
        "account_id",
        col("account_id").cast("long")
    )
    .withColumn(
        "customer_id",
        col("customer_id").cast("long")
    )
    .withColumn(
        "opened_date",
        col("opened_date").cast("date")
    )
    .withColumn(
        "balance",
        col("balance").cast("decimal(14,2)")
    )
    .withColumn(
        "credit_limit",
        col("credit_limit").cast("decimal(14,2)")
    )
)


#ref integ

customer_keys = (
    customers
    .select("customer_id")
    .dropDuplicates()
)

accounts_with_customer = accounts.join(
    customer_keys,
    accounts.customer_id == customer_keys.customer_id,
    "left"
).select(
    accounts["*"],
    customer_keys.customer_id.alias(
        "matched_customer_id"
    )
)


#invalid records

invalid_df = accounts_with_customer.filter(
    col("account_id").isNull()
    | col("customer_id").isNull()
    | col("matched_customer_id").isNull()
    | col("opened_date").isNull()
    | (col("balance") < -100000)
    | (col("credit_limit") < 0)
)

invalid_count = invalid_df.count()

print(
    f"Invalid / referentially invalid accounts: "
    f"{invalid_count}"
)


#quantine

if invalid_count > 0:
    invalid_df.drop(
        "matched_customer_id"
    ).write \
        .mode("append") \
        .parquet(QUARANTINE_PATH)


#valid records

valid_df = accounts_with_customer.filter(
    col("account_id").isNotNull()
    & col("customer_id").isNotNull()
    & col("matched_customer_id").isNotNull()
    & col("opened_date").isNotNull()
    & (col("balance") >= -100000)
    & (col("credit_limit") >= 0)
)


#dedup

before_dedup = valid_df.count()

valid_df = valid_df.dropDuplicates(
    ["account_id"]
)

after_dedup = valid_df.count()

duplicates_removed = (
    before_dedup - after_dedup
)


#removing joiner function

valid_df = valid_df.drop(
    "matched_customer_id"
)


#metadata

silver_df = (
    valid_df
    .withColumn(
        "ingestion_timestamp",
        current_timestamp()
    )
    .withColumn(
        "source_system",
        lit("synthetic_banking_accounts")
    )
    .withColumn(
        "pipeline_version",
        lit(PIPELINE_VERSION)
    )
)



silver_df.write \
    .mode("overwrite") \
    .parquet(SILVER_PATH)


silver_count = silver_df.count()




print("ACCOUNTS PIPELINE COMPLETE")
print(f"RAW records:         {raw_count}")
print(f"Invalid records:     {invalid_count}")
print(f"Duplicates removed:  {duplicates_removed}")
print(f"SILVER records:      {silver_count}")



job.commit()
