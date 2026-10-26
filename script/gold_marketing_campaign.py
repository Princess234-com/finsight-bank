import sys

from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.context import SparkContext
from pyspark.sql import functions as F


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

BUCKET = "finsight-bank-data"

SILVER_PATH = (
    f"s3://{BUCKET}/silver/campaign_contacts/"
)

GOLD_PATH = (
    f"s3://{BUCKET}/gold/marketing_campaign/"
)

PIPELINE_VERSION = "1.0"
#reading silver 

campaign_contacts = spark.read.parquet(
    SILVER_PATH
)

print(
    f"Campaign contact records: "
    f"{campaign_contacts.count()}"
)
#column validation

required_columns = [
    "contact_id",
    "customer_id",
    "campaign_id",
    "contact_date",
    "channel",
    "duration_seconds",
    "outcome"
]

missing_columns = [
    column_name
    for column_name in required_columns
    if column_name not in campaign_contacts.columns
]

if missing_columns:
    raise Exception(
        "Missing required campaign columns: "
        + ", ".join(missing_columns)
    )


campaign_contacts = (
    campaign_contacts

    .withColumn(
        "contact_id",
        F.col("contact_id").cast("long")
    )

    .withColumn(
        "customer_id",
        F.col("customer_id").cast("long")
    )

    .withColumn(
        "campaign_id",
        F.col("campaign_id").cast("long")
    )

    .withColumn(
        "duration_seconds",
        F.col("duration_seconds").cast("double")
    )

    .withColumn(
        "contact_date",
        F.to_date("contact_date")
    )
)
#remove invalid records

campaign_contacts = (
    campaign_contacts

    .filter(
        F.col("contact_id").isNotNull()
    )

    .filter(
        F.col("customer_id").isNotNull()
    )

    .filter(
        F.col("campaign_id").isNotNull()
    )
)


#removing duplivating contacts

campaign_contacts = (
    campaign_contacts
    .dropDuplicates(["contact_id"])
)

#campaign aggregation

marketing_campaign = (
    campaign_contacts
    .groupBy("campaign_id")
    .agg(

        # Contact volume
        F.count(
            "contact_id"
        ).alias(
            "total_contacts"
        ),

        F.countDistinct(
            "customer_id"
        ).alias(
            "unique_customers"
        ),

        # Duration
        F.avg(
            "duration_seconds"
        ).alias(
            "average_contact_duration_seconds"
        ),

        F.sum(
            "duration_seconds"
        ).alias(
            "total_contact_duration_seconds"
        ),
        # Outcomes

        F.sum(
            F.when(
                F.lower(
                    F.col("outcome")
                ).isin(
                    "converted",
                    "subscribed",
                    "success"
                ),
                1
            ).otherwise(0)
        ).alias(
            "conversions"
        ),

        F.sum(
            F.when(
                F.lower(
                    F.col("outcome")
                ).isin(
                    "interested",
                    "converted",
                    "subscribed",
                    "success"
                ),
                1
            ).otherwise(0)
        ).alias(
            "interested_contacts"
        ),

        # Channels
        F.countDistinct(
            "channel"
        ).alias(
            "channels_used"
        ),
        # Date range
        F.min(
            "contact_date"
        ).alias(
            "campaign_start_date"
        ),

        F.max(
            "contact_date"
        ).alias(
            "campaign_last_contact_date"
        )
    )
)
# campaign conversion rate
marketing_campaign = marketing_campaign.withColumn(
    "conversion_rate",

    F.when(
        F.col("total_contacts") > 0,

        F.col("conversions")
        / F.col("total_contacts")
    )

    .otherwise(0.0)
)

#interest rate

marketing_campaign = marketing_campaign.withColumn(
    "interest_rate",

    F.when(
        F.col("total_contacts") > 0,

        F.col("interested_contacts")
        / F.col("total_contacts")
    )

    .otherwise(0.0)
)

#campaign duration

marketing_campaign = marketing_campaign.withColumn(
    "campaign_duration_days",

    F.when(
        F.col("campaign_start_date").isNotNull()
        & F.col("campaign_last_contact_date").isNotNull(),

        F.datediff(
            F.col("campaign_last_contact_date"),
            F.col("campaign_start_date")
        ) + 1
    )

    .otherwise(0)
)


marketing_campaign = (
    marketing_campaign

    .withColumn(
        "average_contact_duration_seconds",
        F.round(
            "average_contact_duration_seconds",
            2
        )
    )

    .withColumn(
        "total_contact_duration_seconds",
        F.round(
            "total_contact_duration_seconds",
            2
        )
    )

    .withColumn(
        "conversion_rate",
        F.round(
            "conversion_rate",
            4
        )
    )

    .withColumn(
        "interest_rate",
        F.round(
            "interest_rate",
            4
        )
    )
)


marketing_campaign = (
    marketing_campaign

    .withColumn(
        "gold_created_at",
        F.current_timestamp()
    )

    .withColumn(
        "source_layer",
        F.lit("silver")
    )

    .withColumn(
        "gold_table",
        F.lit("marketing_campaign")
    )

    .withColumn(
        "pipeline_version",
        F.lit(PIPELINE_VERSION)
    )
)

# final column
marketing_campaign = marketing_campaign.select(

    "campaign_id",

    # Volume
    "total_contacts",
    "unique_customers",

    # Engagement
    "average_contact_duration_seconds",
    "total_contact_duration_seconds",

    # Outcomes
    "conversions",
    "interested_contacts",
    "conversion_rate",
    "interest_rate",

    # Channels
    "channels_used",

    # Dates
    "campaign_start_date",
    "campaign_last_contact_date",
    "campaign_duration_days",

    # Metadata
    "gold_created_at",
    "source_layer",
    "gold_table",
    "pipeline_version"
)

total_campaigns = (
    marketing_campaign.count()
)

unique_campaigns = (
    marketing_campaign
    .select("campaign_id")
    .distinct()
    .count()
)

null_campaign_ids = (
    marketing_campaign
    .filter(
        F.col("campaign_id").isNull()
    )
    .count()
)

print("MARKETING CAMPAIGN DATA QUALITY")


print(
    f"Campaigns:          {total_campaigns}"
)

print(
    f"Unique campaigns:   {unique_campaigns}"
)

print(
    f"Null campaign IDs:  {null_campaign_ids}"
)

if null_campaign_ids > 0:
    raise Exception(
        "Data quality failure: NULL campaign_id."
    )

if total_campaigns != unique_campaigns:
    raise Exception(
        "Data quality failure: duplicate campaign_id."
    )


print(
    "Writing Marketing Campaign Gold dataset..."
)

(
    marketing_campaign
    .write
    .mode("overwrite")
    .format("parquet")
    .save(GOLD_PATH)
)


gold_check = spark.read.parquet(
    GOLD_PATH
)

print(
    f"Gold campaign records: "
    f"{gold_check.count()}"
)

gold_check.printSchema()

gold_check.show(
    20,
    truncate=False
)

print("MARKETING CAMPAIGN GOLD PIPELINE COMPLETED")

print(f"Output path: {GOLD_PATH}")

print(f"Pipeline version: {PIPELINE_VERSION}")

job.commit()
