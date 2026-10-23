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




RAW_PATH = \
    "s3://finsight-bank-data/raw/synthetic_data/campaign_contacts/campaign_contacts.csv"

CUSTOMERS_SILVER_PATH = \
    "s3://finsight-bank-data/silver/customers/"

SILVER_PATH = \
    "s3://finsight-bank-data/silver/campaign_contacts/"

QUARANTINE_PATH = \
    "s3://finsight-bank-data/quarantine/campaign_contacts/"

PIPELINE_VERSION = "1.0"


#reading data
contacts = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(RAW_PATH)
)

customers = spark.read.parquet(
    CUSTOMERS_SILVER_PATH
)

raw_count = contacts.count()


#datatype casting

contacts = (
    contacts
    .withColumn(
        "contact_id",
        col("contact_id").cast("long")
    )
    .withColumn(
        "customer_id",
        col("customer_id").cast("long")
    )
    .withColumn(
        "campaign_id",
        col("campaign_id").cast("int")
    )
    .withColumn(
        "contact_date",
        col("contact_date").cast("date")
    )
    .withColumn(
        "duration_seconds",
        col("duration_seconds").cast("int")
    )
)



customer_keys = (
    customers
    .select("customer_id")
    .dropDuplicates()
)

# ref integ

checked = contacts.join(
    customer_keys,
    contacts.customer_id ==
    customer_keys.customer_id,
    "left"
).select(
    contacts["*"],
    customer_keys.customer_id.alias(
        "matched_customer_id"
    )
)


invalid_df = checked.filter(
    col("contact_id").isNull()
    | col("customer_id").isNull()
    | col("matched_customer_id").isNull()
    | (col("campaign_id") <= 0)
    | col("contact_date").isNull()
    | (col("duration_seconds") < 0)
    | (
        col("contact_date")
        > lit("2026-01-01").cast("date")
    )
)

invalid_count = invalid_df.count()


#quantine

if invalid_count > 0:

    invalid_df.drop(
        "matched_customer_id"
    ).write \
        .mode("append") \
        .parquet(QUARANTINE_PATH)



valid_df = checked.filter(
    col("contact_id").isNotNull()
    & col("customer_id").isNotNull()
    & col("matched_customer_id").isNotNull()
    & (col("campaign_id") > 0)
    & col("contact_date").isNotNull()
    & (col("duration_seconds") >= 0)
    & (
        col("contact_date")
        <= lit("2026-01-01").cast("date")
    )
)


#dedup

before_dedup = valid_df.count()

valid_df = valid_df.dropDuplicates(
    ["contact_id"]
)

after_dedup = valid_df.count()

duplicates_removed = (
    before_dedup - after_dedup
)


valid_df = valid_df.drop(
    "matched_customer_id"
)


# ============================================================
# METADATA
# ============================================================

silver_df = (
    valid_df
    .withColumn(
        "ingestion_timestamp",
        current_timestamp()
    )
    .withColumn(
        "source_system",
        lit("synthetic_banking_campaign_contacts")
    )
    .withColumn(
        "pipeline_version",
        lit(PIPELINE_VERSION)
    )
)


# metadata

silver_df.write \
    .mode("overwrite") \
    .parquet(SILVER_PATH)

silver_count = silver_df.count()


print("==========================================")
print("CAMPAIGN CONTACTS PIPELINE COMPLETE")
print("==========================================")
print(f"RAW records:         {raw_count}")
print(f"Invalid records:     {invalid_count}")
print(f"Duplicates removed:  {duplicates_removed}")
print(f"SILVER records:      {silver_count}")
print("==========================================")


job.commit()
