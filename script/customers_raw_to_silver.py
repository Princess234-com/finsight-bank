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

RAW_PATH = "s3://finsight-bank-data/raw/synthetic_data/customers/customers.csv"

SILVER_PATH = "s3://finsight-bank-data/silver/customers/"

QUARANTINE_PATH = "s3://finsight-bank-data/quarantine/customers/"

PIPELINE_VERSION = "1.0"


# reading raw

df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(RAW_PATH)
)

raw_count = df.count()

print(f"RAW records: {raw_count}")

#type casting

df = (
    df
    .withColumn("customer_id", col("customer_id").cast("long"))
    .withColumn("age", col("age").cast("int"))
    .withColumn("annual_income", col("annual_income").cast("decimal(12,2)"))
    .withColumn("credit_score", col("credit_score").cast("int"))
    .withColumn(
        "customer_since",
        col("customer_since").cast("date")
    )
)


#data quality

invalid_df = df.filter(
    col("customer_id").isNull()
    | (col("age") < 18)
    | (col("age") > 100)
    | (col("annual_income") <= 0)
    | (col("credit_score") < 300)
    | (col("credit_score") > 850)
    | col("customer_since").isNull()
)

invalid_count = invalid_df.count()

print(f"Invalid customer records: {invalid_count}")


#quarantinne

if invalid_count > 0:
    invalid_df.write \
        .mode("append") \
        .parquet(QUARANTINE_PATH)


#valid record

valid_df = df.filter(
    col("customer_id").isNotNull()
    & (col("age") >= 18)
    & (col("age") <= 100)
    & (col("annual_income") > 0)
    & (col("credit_score") >= 300)
    & (col("credit_score") <= 850)
    & col("customer_since").isNotNull()
)


#dedup
before_dedup = valid_df.count()

valid_df = valid_df.dropDuplicates(
    ["customer_id"]
)

after_dedup = valid_df.count()

duplicates_removed = (
    before_dedup - after_dedup
)


# adding metadata

silver_df = (
    valid_df
    .withColumn(
        "ingestion_timestamp",
        current_timestamp()
    )
    .withColumn(
        "source_system",
        lit("synthetic_banking_customers")
    )
    .withColumn(
        "pipeline_version",
        lit(PIPELINE_VERSION)
    )
)


# writing to silver


silver_df.write \
    .mode("overwrite") \
    .parquet(SILVER_PATH)



silver_count = silver_df.count()


print("CUSTOMERS PIPELINE COMPLETE")
print(f"RAW records:         {raw_count}")
print(f"Invalid records:     {invalid_count}")
print(f"Duplicates removed:  {duplicates_removed}")
print(f"SILVER records:      {silver_count}")
print(f"SILVER path:         {SILVER_PATH}")



job.commit()
