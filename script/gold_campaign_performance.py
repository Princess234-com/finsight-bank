import sys

from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.sql.functions import (
    col,
    lit,
    count,
    countDistinct,
    sum as spark_sum,
    avg,
    min as spark_min,
    max as spark_max,
    when,
    lower,
    trim,
    current_timestamp,
    to_date,
    datediff,
    round as spark_round
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

CAMPAIGNS_SILVER_PATH = (
    "s3://finsight-bank-data/"
    "silver/campaign_contacts/"
)

CUSTOMERS_SILVER_PATH = (
    "s3://finsight-bank-data/"
    "silver/customers/"
)

GOLD_PATH = (
    "s3://finsight-bank-data/"
    "gold/campaign_performance/"
)

PIPELINE_VERSION = "1.0"
#read data

print("GOLD CAMPAIGN PERFORMANCE PIPELINE")


print("Reading Silver campaign contacts...")

campaign_contacts = spark.read.parquet(
    CAMPAIGNS_SILVER_PATH
)

print("Reading Silver customers...")

customers = spark.read.parquet(
    CUSTOMERS_SILVER_PATH
)


campaign_contact_count = campaign_contacts.count()
customer_count = customers.count()

print(
    f"Silver campaign contacts: {campaign_contact_count}"
)

print(
    f"Silver customers:          {customer_count}"
)

required_campaign_columns = [
    "contact_id",
    "customer_id",
    "campaign_id",
    "contact_date",
    "channel",
    "duration_seconds",
    "outcome"
]

missing_campaign_columns = [
    c
    for c in required_campaign_columns
    if c not in campaign_contacts.columns
]

if missing_campaign_columns:
    raise ValueError(
        "Missing required Silver campaign columns: "
        + ", ".join(missing_campaign_columns)
    )


if "customer_id" not in customers.columns:
    raise ValueError(
        "Silver customers table is missing customer_id."
    )


customer_dim = (
    customers
    .select(
        col("customer_id")
            .cast("long")
            .alias("customer_id")
    )
    .dropDuplicates(["customer_id"])
)


campaign_contacts = (
    campaign_contacts
    .select(
        col("contact_id")
            .cast("long")
            .alias("contact_id"),

        col("customer_id")
            .cast("long")
            .alias("customer_id"),

        col("campaign_id")
            .cast("int")
            .alias("campaign_id"),

        col("contact_date")
            .cast("date")
            .alias("contact_date"),

        col("channel")
            .cast("string")
            .alias("channel"),

        col("duration_seconds")
            .cast("int")
            .alias("duration_seconds"),

        col("outcome")
            .cast("string")
            .alias("outcome")
    )
)


print("Applying campaign data quality rules...")

campaign_clean = (
    campaign_contacts

    # Required identifiers
    .filter(col("contact_id").isNotNull())
    .filter(col("customer_id").isNotNull())
    .filter(col("campaign_id").isNotNull())

    # Valid campaign ID
    .filter(col("campaign_id") > 0)

    # Required date
    .filter(col("contact_date").isNotNull())

    # Valid duration
    .filter(
        col("duration_seconds").isNotNull()
    )
    .filter(
        col("duration_seconds") >= 0
    )

    # Required outcome
    .filter(col("outcome").isNotNull())
)

campaign_clean = (
    campaign_clean
    .dropDuplicates(["contact_id"])
)


clean_contact_count = campaign_clean.count()

print(
    f"Clean campaign contacts: {clean_contact_count}"
)


print("Checking campaign/customer relationships...")

campaign_with_customer = (
    campaign_clean.alias("c")
    .join(
        customer_dim.alias("d"),
        col("c.customer_id") == col("d.customer_id"),
        "inner"
    )
    .select(
        col("c.contact_id"),
        col("c.customer_id"),
        col("c.campaign_id"),
        col("c.contact_date"),
        col("c.channel"),
        col("c.duration_seconds"),
        col("c.outcome")
    )
)


valid_customer_contact_count = (
    campaign_with_customer.count()
)

print(
    "Campaign contacts with valid customers: "
    f"{valid_customer_contact_count}"
)


campaign_with_customer = (
    campaign_with_customer

    .withColumn(
        "channel_normalized",
        lower(
            trim(
                col("channel")
            )
        )
    )

    .withColumn(
        "outcome_normalized",
        lower(
            trim(
                col("outcome")
            )
        )
    )
)

campaign_with_customer = (
    campaign_with_customer

    .withColumn(
        "is_conversion",
        when(
            col("outcome_normalized").isin(
                "success",
                "successful",
                "converted",
                "conversion",
                "yes",
                "subscribed",
                "accepted"
            ),
            lit(1)
        )
        .otherwise(lit(0))
    )

    .withColumn(
        "is_interested",
        when(
            col("outcome_normalized").isin(
                "interested",
                "interest",
                "success",
                "successful",
                "converted",
                "conversion",
                "yes",
                "subscribed",
                "accepted"
            ),
            lit(1)
        )
        .otherwise(lit(0))
    )

    .withColumn(
        "is_contacted",
        lit(1)
    )
)


print("Aggregating campaign performance...")

campaign_performance = (
    campaign_with_customer
    .groupBy("campaign_id")
    .agg(

        count("*").alias(
            "total_contacts"
        ),

        countDistinct("customer_id").alias(
            "unique_customers_contacted"
        ),

        spark_sum("is_conversion").alias(
            "conversions"
        ),

        spark_sum("is_interested").alias(
            "interested_contacts"
        ),

        avg("duration_seconds").alias(
            "average_contact_duration_seconds"
        ),

        spark_max("duration_seconds").alias(
            "maximum_contact_duration_seconds"
        ),

        spark_min("duration_seconds").alias(
            "minimum_contact_duration_seconds"
        ),

        spark_sum("duration_seconds").alias(
            "total_contact_duration_seconds"
        ),

        spark_min("contact_date").alias(
            "campaign_start_date"
        ),

        spark_max("contact_date").alias(
            "campaign_last_contact_date"
        )
    )
)


# Calculate the number of distinct channels used by each campaign.

channel_counts = (
    campaign_with_customer
    .groupBy("campaign_id")
    .agg(
        countDistinct("channel_normalized").alias(
            "channels_used"
        )
    )
)


campaign_performance = (
    campaign_performance
    .join(
        channel_counts,
        "campaign_id",
        "left"
    )
)

#calcuate campaign duration

campaign_performance = campaign_performance.withColumn(
    "campaign_duration_days",
    datediff(
        col("campaign_last_contact_date"),
        col("campaign_start_date")
    ) + lit(1)
)


#calculate conversion rate

campaign_performance = campaign_performance.withColumn(
    "conversion_rate",
    when(
        col("total_contacts") > 0,
        col("conversions")
        / col("total_contacts")
    )
    .otherwise(lit(0))
)
#calculate interest rate

campaign_performance = campaign_performance.withColumn(
    "interest_rate",
    when(
        col("total_contacts") > 0,
        col("interested_contacts")
        / col("total_contacts")
    )
    .otherwise(lit(0))
)
#calcuate contact frequency

campaign_performance = campaign_performance.withColumn(
    "contacts_per_day",
    when(
        col("campaign_duration_days") > 0,
        col("total_contacts")
        / col("campaign_duration_days")
    )
    .otherwise(
        col("total_contacts")
    )
)


#calculate conversion efficieny

campaign_performance = campaign_performance.withColumn(
    "conversions_per_100_contacts",
    when(
        col("total_contacts") > 0,
        (
            col("conversions")
            / col("total_contacts")
        ) * lit(100)
    )
    .otherwise(lit(0))
)

#calculate operformance segment

campaign_performance = (
    campaign_performance
    .withColumn(
        "performance_segment",
        when(
            col("conversion_rate") >= 0.20,
            lit("HIGH")
        )
        .when(
            col("conversion_rate") >= 0.10,
            lit("MEDIUM")
        )
        .when(
            col("conversion_rate") > 0,
            lit("LOW")
        )
        .otherwise(
            lit("NO_CONVERSIONS")
        )
    )
)

#round metrics

campaign_performance = (
    campaign_performance

    .withColumn(
        "average_contact_duration_seconds",
        spark_round(
            col("average_contact_duration_seconds"),
            2
        )
    )

    .withColumn(
        "conversion_rate",
        spark_round(
            col("conversion_rate"),
            4
        )
    )

    .withColumn(
        "interest_rate",
        spark_round(
            col("interest_rate"),
            4
        )
    )

    .withColumn(
        "contacts_per_day",
        spark_round(
            col("contacts_per_day"),
            2
        )
    )

    .withColumn(
        "conversions_per_100_contacts",
        spark_round(
            col("conversions_per_100_contacts"),
            2
        )
    )
)


gold_campaign_performance = (
    campaign_performance

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
        lit("synthetic_banking_campaign_contacts")
    )
)


gold_campaign_performance = (
    gold_campaign_performance
    .select(

        # Campaign
        "campaign_id",

        # Volume
        "total_contacts",
        "unique_customers_contacted",

        # Outcomes
        "conversions",
        "interested_contacts",

        # Rates
        "conversion_rate",
        "interest_rate",
        "conversions_per_100_contacts",

        # Duration
        "average_contact_duration_seconds",
        "minimum_contact_duration_seconds",
        "maximum_contact_duration_seconds",
        "total_contact_duration_seconds",

        # Channels
        "channels_used",

        # Dates
        "campaign_start_date",
        "campaign_last_contact_date",
        "campaign_duration_days",

        # Frequency
        "contacts_per_day",

        # Segment
        "performance_segment",

        # Metadata
        "pipeline_version",
        "gold_created_timestamp",
        "source_system"
    )
)


print("Running final Gold data quality checks...")

null_campaign_ids = (
    gold_campaign_performance
    .filter(
        col("campaign_id").isNull()
    )
    .count()
)


duplicate_campaign_ids = (
    gold_campaign_performance
    .groupBy("campaign_id")
    .count()
    .filter(
        col("count") > 1
    )
    .count()
)


invalid_conversion_rates = (
    gold_campaign_performance
    .filter(
        (col("conversion_rate") < 0)
        |
        (col("conversion_rate") > 1)
    )
    .count()
)


invalid_interest_rates = (
    gold_campaign_performance
    .filter(
        (col("interest_rate") < 0)
        |
        (col("interest_rate") > 1)
    )
    .count()
)


negative_contacts = (
    gold_campaign_performance
    .filter(
        col("total_contacts") < 0
    )
    .count()
)


negative_durations = (
    gold_campaign_performance
    .filter(
        col("campaign_duration_days") < 0
    )
    .count()
)


print(
    f"Null campaign IDs: {null_campaign_ids}"
)

print(
    f"Duplicate campaign IDs: {duplicate_campaign_ids}"
)

print(
    f"Invalid conversion rates: {invalid_conversion_rates}"
)

print(
    f"Invalid interest rates: {invalid_interest_rates}"
)

print(
    f"Negative contact counts: {negative_contacts}"
)

print(
    f"Negative campaign durations: {negative_durations}"
)


if null_campaign_ids > 0:
    raise ValueError(
        "Gold campaign performance contains NULL campaign_id."
    )

if duplicate_campaign_ids > 0:
    raise ValueError(
        "Gold campaign performance contains duplicate campaign_id."
    )

if invalid_conversion_rates > 0:
    raise ValueError(
        "Conversion rate outside valid 0-1 range."
    )

if invalid_interest_rates > 0:
    raise ValueError(
        "Interest rate outside valid 0-1 range."
    )

if negative_contacts > 0:
    raise ValueError(
        "Negative contact count detected."
    )

if negative_durations > 0:
    raise ValueError(
        "Negative campaign duration detected."
    )


print("GOLD CAMPAIGN PERFORMANCE SCHEMA")

gold_campaign_performance.printSchema()

print("GOLD CAMPAIGN PERFORMANCE SAMPLE")

gold_campaign_performance.show(
    20,
    truncate=False
)

print("CAMPAIGN PERFORMANCE DISTRIBUTION")

(
    gold_campaign_performance
    .groupBy("performance_segment")
    .count()
    .orderBy("performance_segment")
    .show()
)

print("WRITING GOLD CAMPAIGN PERFORMANCE")

(
    gold_campaign_performance
    .write
    .mode("overwrite")
    .format("parquet")
    .save(GOLD_PATH)
)


print("VERIFYING GOLD OUTPUT")


gold_check = (
    spark.read
    .parquet(GOLD_PATH)
)

gold_count = gold_check.count()

gold_column_count = len(
    gold_check.columns
)

print(
    f"Gold campaign records: {gold_count}"
)

print(
    f"Gold column count: {gold_column_count}"
)

gold_check.printSchema()

gold_check.show(
    10,
    truncate=False
)

print("GOLD CAMPAIGN PERFORMANCE COMPLETE")

print(
    f"Silver campaign contacts: "
    f"{campaign_contact_count}"
)

print(
    f"Valid campaign contacts: "
    f"{valid_customer_contact_count}"
)

print(
    f"Gold campaigns: "
    f"{gold_count}"
)

print(
    f"Gold location: "
    f"{GOLD_PATH}"
)

print(
    f"Pipeline version: "
    f"{PIPELINE_VERSION}"
)


job.commit()
