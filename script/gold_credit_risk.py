import sys

from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.context import SparkContext
from pyspark.sql import functions as F

args = getResolvedOptions(
    sys.argv,
    ["JOB_NAME"]
)

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

job = Job(glueContext)
job.init(args["JOB_NAME"], args)
BUCKET = "finsight-bank-data"

GOLD_BASE = f"s3://{BUCKET}/gold"

CUSTOMER_360_PATH = (
    f"{GOLD_BASE}/customer_360/"
)

GOLD_PATH = (
    f"{GOLD_BASE}/credit_risk/"
)

PIPELINE_VERSION = "1.0"
print("Reading Gold Customer 360...")

customer_360 = spark.read.parquet(
    CUSTOMER_360_PATH
)

print(
    f"Customer 360 records: "
    f"{customer_360.count()}"
)

required_columns = [
    "customer_id",
    "age",
    "annual_income",
    "credit_score",
    "total_balance",
    "total_credit_limit",
    "total_loans",
    "active_loans",
    "total_debt",
    "active_debt",
    "average_interest_rate",
    "late_or_defaulted_loans",
    "monthly_repayment",
    "monthly_income",
    "debt_to_income_ratio",
    "debt_service_ratio",
    "payment_count",
    "successful_payments",
    "failed_payments",
    "late_payments",
    "successful_payment_rate"
]

missing_columns = [
    column_name
    for column_name in required_columns
    if column_name not in customer_360.columns
]

if missing_columns:
    raise Exception(
        "Missing required Customer 360 columns: "
        + ", ".join(missing_columns)
    )

credit_risk = customer_360.select(
    # Customer
    "customer_id",
    "age",
    "annual_income",
    "credit_score",
    # Account / credit exposure
    "total_balance",
    "total_credit_limit",
    "credit_accounts",
    # Loan exposure
    "total_loans",
    "active_loans",
    "total_debt",
    "active_debt",
    "average_interest_rate",
    "late_or_defaulted_loans",
    "monthly_repayment",
    # Debt ratios
    "monthly_income",
    "debt_to_income_ratio",
    "debt_service_ratio",
    # Payment behaviour
    "payment_count",
    "successful_payments",
    "failed_payments",
    "late_payments",
    "successful_payment_rate"
)

numeric_columns = [
    "total_balance",
    "total_credit_limit",
    "credit_accounts",
    "total_loans",
    "active_loans",
    "total_debt",
    "active_debt",
    "average_interest_rate",
    "late_or_defaulted_loans",
    "monthly_repayment",
    "monthly_income",
    "debt_to_income_ratio",
    "debt_service_ratio",
    "payment_count",
    "successful_payments",
    "failed_payments",
    "late_payments",
    "successful_payment_rate"
]

for column_name in numeric_columns:

    credit_risk = credit_risk.withColumn(
        column_name,
        F.coalesce(
            F.col(column_name).cast("double"),
            F.lit(0.0)
        )
    )
#credit score band

credit_risk = credit_risk.withColumn(
    "credit_score_band",

    F.when(
        F.col("credit_score") < 580,
        "poor"
    )

    .when(
        F.col("credit_score") < 670,
        "fair"
    )

    .when(
        F.col("credit_score") < 740,
        "good"
    )

    .when(
        F.col("credit_score") < 800,
        "very_good"
    )

    .otherwise(
        "excellent"
    )
)
#debt to ratio band

credit_risk = credit_risk.withColumn(
    "debt_to_income_band",

    F.when(
        F.col("debt_to_income_ratio") < 0.20,
        "low"
    )

    .when(
        F.col("debt_to_income_ratio") < 0.40,
        "moderate"
    )

    .when(
        F.col("debt_to_income_ratio") < 0.60,
        "high"
    )

    .otherwise(
        "very_high"
    )
)
#payment failare rate
credit_risk = credit_risk.withColumn(
    "payment_failure_rate",

    F.when(
        F.col("payment_count") > 0,

        F.col("failed_payments")
        / F.col("payment_count")
    )

    .otherwise(0.0)
)
#late payment rate

credit_risk = credit_risk.withColumn(
    "late_payment_rate",

    F.when(
        F.col("payment_count") > 0,

        F.col("late_payments")
        / F.col("payment_count")
    )

    .otherwise(0.0)
)
#debt to credit limit ratio

credit_risk = credit_risk.withColumn(
    "debt_to_credit_limit_ratio",

    F.when(
        F.col("total_credit_limit") > 0,

        F.col("active_debt")
        / F.col("total_credit_limit")
    )

    .otherwise(0.0)
)
credit_risk = (
    credit_risk

    # Low credit score indicator
    .withColumn(
        "low_credit_score_flag",

        F.when(
            F.col("credit_score") < 670,
            1
        ).otherwise(0)
    )

    # High debt indicator
    .withColumn(
        "high_debt_flag",

        F.when(
            F.col("debt_to_income_ratio") >= 0.40,
            1
        ).otherwise(0)
    )

    # Payment problems
    .withColumn(
        "payment_problem_flag",

        F.when(
            (F.col("failed_payments") > 0)
            | (F.col("late_payments") > 0),
            1
        ).otherwise(0)
    )

    # Loan problems
    .withColumn(
        "loan_problem_flag",

        F.when(
            F.col("late_or_defaulted_loans") > 0,
            1
        ).otherwise(0)
    )

    # High debt-service burden
    .withColumn(
        "high_debt_service_flag",

        F.when(
            F.col("debt_service_ratio") >= 0.40,
            1
        ).otherwise(0)
    )
)

credit_risk = (
    credit_risk

    .withColumn(
        "total_balance",
        F.round("total_balance", 2)
    )

    .withColumn(
        "total_credit_limit",
        F.round("total_credit_limit", 2)
    )

    .withColumn(
        "total_debt",
        F.round("total_debt", 2)
    )

    .withColumn(
        "active_debt",
        F.round("active_debt", 2)
    )

    .withColumn(
        "monthly_repayment",
        F.round("monthly_repayment", 2)
    )

    .withColumn(
        "debt_to_income_ratio",
        F.round("debt_to_income_ratio", 4)
    )

    .withColumn(
        "debt_service_ratio",
        F.round("debt_service_ratio", 4)
    )

    .withColumn(
        "payment_failure_rate",
        F.round("payment_failure_rate", 4)
    )

    .withColumn(
        "late_payment_rate",
        F.round("late_payment_rate", 4)
    )

    .withColumn(
        "debt_to_credit_limit_ratio",
        F.round("debt_to_credit_limit_ratio", 4)
    )
)
credit_risk = (
    credit_risk

    .withColumn(
        "gold_created_at",
        F.current_timestamp()
    )

    .withColumn(
        "source_layer",
        F.lit("gold")
    )

    .withColumn(
        "gold_table",
        F.lit("credit_risk")
    )

    .withColumn(
        "pipeline_version",
        F.lit(PIPELINE_VERSION)
    )
)
credit_risk = credit_risk.select(

    "customer_id",

    # Customer
    "age",
    "annual_income",
    "credit_score",
    "credit_score_band",

    # Credit exposure
    "total_balance",
    "total_credit_limit",
    "credit_accounts",
    "debt_to_credit_limit_ratio",

    # Loans
    "total_loans",
    "active_loans",
    "total_debt",
    "active_debt",
    "average_interest_rate",
    "late_or_defaulted_loans",
    "monthly_repayment",

    # Income / debt
    "monthly_income",
    "debt_to_income_ratio",
    "debt_to_income_band",
    "debt_service_ratio",

    # Payments
    "payment_count",
    "successful_payments",
    "failed_payments",
    "late_payments",
    "successful_payment_rate",
    "payment_failure_rate",
    "late_payment_rate",

    # Risk indicators
    "low_credit_score_flag",
    "high_debt_flag",
    "payment_problem_flag",
    "loan_problem_flag",
    "high_debt_service_flag",

    # Metadata
    "gold_created_at",
    "source_layer",
    "gold_table",
    "pipeline_version"
)

total_records = credit_risk.count()

unique_customers = (
    credit_risk
    .select("customer_id")
    .distinct()
    .count()
)

null_customer_ids = (
    credit_risk
    .filter(F.col("customer_id").isNull())
    .count()
)

print("CREDIT RISK DATA QUALITY")

print(f"Total records:       {total_records}")
print(f"Unique customers:    {unique_customers}")
print(f"Null customer IDs:   {null_customer_ids}")

if null_customer_ids > 0:
    raise Exception(
        "Data quality failure: NULL customer_id values."
    )

if total_records != unique_customers:
    raise Exception(
        "Data quality failure: duplicate customer_id values."
    )

print("Writing Credit Risk Gold dataset...")

(
    credit_risk
    .write
    .mode("overwrite")
    .format("parquet")
    .save(GOLD_PATH)
)


gold_check = spark.read.parquet(
    GOLD_PATH
)

print(
    f"Gold credit-risk records: "
    f"{gold_check.count()}"
)

gold_check.printSchema()

gold_check.show(
    20,
    truncate=False
)

print("CREDIT RISK GOLD PIPELINE COMPLETED")

print(
    f"Output path: {GOLD_PATH}"
)

print(
    f"Pipeline version: {PIPELINE_VERSION}"
)

job.commit()
