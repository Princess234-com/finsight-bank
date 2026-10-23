import sys

from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.sql.functions import (col, lit, when, count, sum as spark_sum, avg, max as spark_max, min as spark_min, round as spark_round, current_timestamp)


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


LOANS_SILVER_PATH = (
    "s3://finsight-bank-data/"
    "silver/loans/"
)

GOLD_PATH = (
    "s3://finsight-bank-data/"
    "gold/loan_risk/"
)

PIPELINE_VERSION = "2.0"

print("GOLD LOAN RISK PIPELINE")

print("Reading Silver loans...")

loans = spark.read.parquet(
    LOANS_SILVER_PATH
)

print("SILVER LOANS SCHEMA")
loans.printSchema()

source_count = loans.count()

print(
    f"Silver loan records: {source_count}"
)


required_columns = [
    "loan_id",
    "customer_id",
    "principal",
    "interest_rate",
    "term_months",
    "status",
    "loan_start_date"
]

missing_columns = [
    c
    for c in required_columns
    if c not in loans.columns
]

if missing_columns:
    raise ValueError(
        "Missing required Silver loan columns: "
        + ", ".join(missing_columns)
    )


print("Standardising loan data types...")

loans = (
    loans
    .select(
        col("loan_id")
            .cast("long")
            .alias("loan_id"),

        col("customer_id")
            .cast("long")
            .alias("customer_id"),

        col("loan_type")
            .cast("string")
            .alias("loan_type"),

        col("principal")
            .cast("double")
            .alias("principal"),

        col("interest_rate")
            .cast("double")
            .alias("interest_rate"),

        col("term_months")
            .cast("int")
            .alias("term_months"),

        col("status")
            .cast("string")
            .alias("status"),

        col("loan_start_date")
            .cast("date")
            .alias("loan_start_date")
    )
)

# DQ
print("Applying loan data quality rules...")

loans_clean = (
    loans

    .filter(
        col("loan_id").isNotNull()
    )

    .filter(
        col("customer_id").isNotNull()
    )

    .filter(
        col("principal").isNotNull()
    )

    .filter(
        col("principal") > 0
    )

    .filter(
        col("interest_rate").isNotNull()
    )

    .filter(
        col("interest_rate") > 0
    )

    .filter(
        col("interest_rate") <= 50
    )

    .filter(
        col("term_months").isNotNull()
    )

    .filter(
        col("term_months") > 0
    )

    .filter(
        col("loan_start_date").isNotNull()
    )

    .filter(
        col("status").isNotNull()
    )
)
#remove duplicates

loans_clean = (
    loans_clean
    .dropDuplicates(
        ["loan_id"]
    )
)


clean_count = loans_clean.count()

print(
    f"Clean loan records: {clean_count}"
)
#status normalization

loans_clean = loans_clean.withColumn(
    "status_normalized",
    lower_status := col("status")
)
# create loan status flags

# Active loans
loans_clean = loans_clean.withColumn(
    "is_active",
    when(
        col("status_normalized").isin(
            "active",
            "open"
        ),
        lit(1)
    )
    .otherwise(lit(0))
)


# Paid-off loans
loans_clean = loans_clean.withColumn(
    "is_paid_off",
    when(
        col("status_normalized").isin(
            "paid off",
            "paid_off",
            "paid"
        ),
        lit(1)
    )
    .otherwise(lit(0))
)


# Late loans
loans_clean = loans_clean.withColumn(
    "is_late",
    when(
        col("status_normalized") == "late",
        lit(1)
    )
    .otherwise(lit(0))
)


# Defaulted loans
loans_clean = loans_clean.withColumn(
    "is_defaulted",
    when(
        col("status_normalized").isin(
            "defaulted",
            "default"
        ),
        lit(1)
    )
    .otherwise(lit(0))
)


# Late OR defaulted
loans_clean = loans_clean.withColumn(
    "is_delinquent",
    when(
        (
            col("is_late") == 1
        )
        |
        (
            col("is_defaulted") == 1
        ),
        lit(1)
    )
    .otherwise(lit(0))
)

# calculating montly interest rate

loans_clean = loans_clean.withColumn(
    "monthly_interest_rate",
    col("interest_rate")
    / lit(12.0)
    / lit(100.0)
)


loans_clean = loans_clean.withColumn(
    "estimated_monthly_payment",
    when(
        col("monthly_interest_rate") == 0,
        col("principal")
        / col("term_months")
    )
    .otherwise(
        (
            col("principal")
            * col("monthly_interest_rate")
            * (
                1
                + col("monthly_interest_rate")
            )
            ** col("term_months")
        )
        /
        (
            (
                1
                + col("monthly_interest_rate")
            )
            ** col("term_months")
            - 1
        )
    )
)

# aggregate loan metrics by customer

print("Creating customer loan risk summary...")

loan_risk = (
    loans_clean
    .groupBy(
        "customer_id"
    )
    .agg(
      
        # Loan counts

        count("*").alias(
            "total_loans"
        ),

        spark_sum("is_active").alias(
            "active_loans"
        ),

        spark_sum("is_paid_off").alias(
            "paid_off_loans"
        ),

        spark_sum("is_late").alias(
            "late_loans"
        ),

        spark_sum("is_defaulted").alias(
            "defaulted_loans"
        ),

        spark_sum("is_delinquent").alias(
            "late_or_defaulted_loans"
        ),
      
        # Principal exposure

        spark_sum("principal").alias(
            "total_loan_principal"
        ),

        spark_sum(
            when(
                col("is_active") == 1,
                col("principal")
            )
            .otherwise(lit(0))
        ).alias(
            "active_debt"
        ),

        spark_max("principal").alias(
            "largest_loan_principal"
        ),

        avg("principal").alias(
            "average_loan_principal"
        ),
        # Interest
        avg("interest_rate").alias(
            "average_interest_rate"
        ),

        spark_max("interest_rate").alias(
            "maximum_interest_rate"
        ),
        # Loan terms

        avg("term_months").alias(
            "average_term_months"
        ),

        spark_max("term_months").alias(
            "maximum_term_months"
        ),


        # Monthly repayment


        spark_sum(
            when(
                col("is_active") == 1,
                col("estimated_monthly_payment")
            )
            .otherwise(lit(0))
        ).alias(
            "monthly_repayment"
        ),
#date

        spark_min("loan_start_date").alias(
            "first_loan_start_date"
        ),

        spark_max("loan_start_date").alias(
            "latest_loan_start_date"
        )
    )
)


#calculate delinquency rate

loan_risk = loan_risk.withColumn(
    "delinquency_rate",
    when(
        col("total_loans") > 0,
        col("late_or_defaulted_loans")
        / col("total_loans")
    )
    .otherwise(lit(0.0))
)

#calculate default rate

loan_risk = loan_risk.withColumn(
    "default_rate",
    when(
        col("total_loans") > 0,
        col("defaulted_loans")
        / col("total_loans")
    )
    .otherwise(lit(0.0))
)

# calculate active loan ratio
loan_risk = loan_risk.withColumn(
    "active_loan_ratio",
    when(
        col("total_loans") > 0,
        col("active_loans")
        / col("total_loans")
    )
    .otherwise(lit(0.0))
)



loan_risk = (
    loan_risk

    .withColumn(
        "has_active_debt",
        when(
            col("active_debt") > 0,
            lit(1)
        )
        .otherwise(lit(0))
    )

    .withColumn(
        "has_delinquent_loan",
        when(
            col("late_or_defaulted_loans") > 0,
            lit(1)
        )
        .otherwise(lit(0))
    )

    .withColumn(
        "has_defaulted_loan",
        when(
            col("defaulted_loans") > 0,
            lit(1)
        )
        .otherwise(lit(0))
    )
)

loan_risk = loan_risk.withColumn(
    "loan_risk_band",
    when(
        col("defaulted_loans") > 0,
        lit("DEFAULT")
    )
    .when(
        col("late_or_defaulted_loans") > 0,
        lit("DELINQUENT")
    )
    .when(
        col("active_debt") > 0,
        lit("ACTIVE")
    )
    .otherwise(
        lit("NO_ACTIVE_RISK")
    )
)

loan_risk = (
    loan_risk

    .withColumn(
        "total_loan_principal",
        spark_round(
            col("total_loan_principal"),
            2
        )
    )

    .withColumn(
        "active_debt",
        spark_round(
            col("active_debt"),
            2
        )
    )

    .withColumn(
        "largest_loan_principal",
        spark_round(
            col("largest_loan_principal"),
            2
        )
    )

    .withColumn(
        "average_loan_principal",
        spark_round(
            col("average_loan_principal"),
            2
        )
    )

    .withColumn(
        "average_interest_rate",
        spark_round(
            col("average_interest_rate"),
            2
        )
    )

    .withColumn(
        "maximum_interest_rate",
        spark_round(
            col("maximum_interest_rate"),
            2
        )
    )

    .withColumn(
        "average_term_months",
        spark_round(
            col("average_term_months"),
            2
        )
    )

    .withColumn(
        "monthly_repayment",
        spark_round(
            col("monthly_repayment"),
            2
        )
    )

    .withColumn(
        "delinquency_rate",
        spark_round(
            col("delinquency_rate"),
            4
        )
    )

    .withColumn(
        "default_rate",
        spark_round(
            col("default_rate"),
            4
        )
    )

    .withColumn(
        "active_loan_ratio",
        spark_round(
            col("active_loan_ratio"),
            4
        )
    )
)

loan_risk = (
    loan_risk

    .withColumn(
        "pipeline_version",
        lit(PIPELINE_VERSION)
    )

    .withColumn(
        "gold_created_timestamp",
        current_timestamp()
    )

    .withColumn(
        "source_layer",
        lit("silver")
    )

    .withColumn(
        "gold_table",
        lit("loan_risk")
    )
)


gold_loan_risk = loan_risk.select(

    # Customer
    "customer_id",

    # Loan counts
    "total_loans",
    "active_loans",
    "paid_off_loans",
    "late_loans",
    "defaulted_loans",
    "late_or_defaulted_loans",

    # Debt
    "total_loan_principal",
    "active_debt",
    "largest_loan_principal",
    "average_loan_principal",

    # Interest
    "average_interest_rate",
    "maximum_interest_rate",

    # Term
    "average_term_months",
    "maximum_term_months",

    # Repayment
    "monthly_repayment",

    # Risk ratios
    "delinquency_rate",
    "default_rate",
    "active_loan_ratio",

    # Dates
    "first_loan_start_date",
    "latest_loan_start_date",

    # Flags
    "has_active_debt",
    "has_delinquent_loan",
    "has_defaulted_loan",

    # Descriptive risk segment
    "loan_risk_band",

    # Metadata
    "pipeline_version",
    "gold_created_timestamp",
    "source_layer",
    "gold_table"
)


print("FINAL DATA QUALITY CHECKS")



null_customer_ids = (
    gold_loan_risk
    .filter(
        col("customer_id").isNull()
    )
    .count()
)


duplicate_customer_ids = (
    gold_loan_risk
    .groupBy("customer_id")
    .count()
    .filter(
        col("count") > 1
    )
    .count()
)


negative_active_debt = (
    gold_loan_risk
    .filter(
        col("active_debt") < 0
    )
    .count()
)


negative_monthly_repayment = (
    gold_loan_risk
    .filter(
        col("monthly_repayment") < 0
    )
    .count()
)


invalid_default_rate = (
    gold_loan_risk
    .filter(
        (col("default_rate") < 0)
        |
        (col("default_rate") > 1)
    )
    .count()
)


invalid_delinquency_rate = (
    gold_loan_risk
    .filter(
        (col("delinquency_rate") < 0)
        |
        (col("delinquency_rate") > 1)
    )
    .count()
)


print(
    f"Null customer IDs: {null_customer_ids}"
)

print(
    f"Duplicate customer IDs: {duplicate_customer_ids}"
)

print(
    f"Negative active debt: {negative_active_debt}"
)

print(
    f"Negative monthly repayment: "
    f"{negative_monthly_repayment}"
)

print(
    f"Invalid default rates: "
    f"{invalid_default_rate}"
)

print(
    f"Invalid delinquency rates: "
    f"{invalid_delinquency_rate}"
)


if null_customer_ids > 0:
    raise ValueError(
        "Data quality failure: NULL customer_id values found."
    )

if duplicate_customer_ids > 0:
    raise ValueError(
        "Data quality failure: duplicate customer_id values found."
    )

if negative_active_debt > 0:
    raise ValueError(
        "Data quality failure: negative active debt found."
    )

if negative_monthly_repayment > 0:
    raise ValueError(
        "Data quality failure: negative monthly repayment found."
    )

if invalid_default_rate > 0:
    raise ValueError(
        "Data quality failure: default rate outside 0-1."
    )

if invalid_delinquency_rate > 0:
    raise ValueError(
        "Data quality failure: delinquency rate outside 0-1."
    )
  
print("GOLD LOAN RISK SCHEMA")


gold_loan_risk.printSchema()

print("GOLD LOAN RISK SAMPLE")

gold_loan_risk.show(
    20,
    truncate=False
)

print("LOAN RISK BAND DISTRIBUTION")

(
    gold_loan_risk
    .groupBy("loan_risk_band")
    .count()
    .orderBy("loan_risk_band")
    .show()
)

print("WRITING GOLD LOAN RISK")


(
    gold_loan_risk
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
    f"Gold customer records: {gold_count}"
)

print(
    f"Gold column count: {gold_column_count}"
)

gold_check.printSchema()

gold_check.show(
    10,
    truncate=False
)

print("GOLD LOAN RISK PIPELINE COMPLETE")

print(f"Silver loan records: {source_count}")

print(f"Clean loan records: {clean_count}")

print(f"Gold customer records: {gold_count}")

print(f"Gold location: {GOLD_PATH}")

print(f"Pipeline version: {PIPELINE_VERSION}")


job.commit()

