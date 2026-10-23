import sys

from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.sql import functions as F


# 1. GLUE JOB SETUP

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

BUCKET = "finsight-bank-data"

SILVER_BASE = f"s3://{BUCKET}/silver"

GOLD_PATH = (
    f"s3://{BUCKET}/gold/customer_360/"
)

PIPELINE_VERSION = "3.0"

#reading silver tables

print("Reading Silver customers...")

customers = spark.read.parquet(
    f"{SILVER_BASE}/customers/"
)

print("Reading Silver accounts...")

accounts = spark.read.parquet(
    f"{SILVER_BASE}/accounts/"
)

print("Reading Silver transactions...")

transactions = spark.read.parquet(
    f"{SILVER_BASE}/transactions/"
)

print("Reading Silver loans...")

loans = spark.read.parquet(
    f"{SILVER_BASE}/loans/"
)

print("Reading Silver payments...")

payments = spark.read.parquet(
    f"{SILVER_BASE}/payments/"
)

print("Reading Silver campaign contacts...")

campaign_contacts = spark.read.parquet(
    f"{SILVER_BASE}/campaign_contacts/"
)


#customer

customer_base = (
    customers
    .select(
        "customer_id",
        "age",
        "gender",
        "education",
        "marital_status",
        "occupation",
        "annual_income",
        "credit_score",
        "customer_since"
    )
    .dropDuplicates(
        ["customer_id"]
    )
)


#account aggregation

print("Aggregating account information...")

account_agg = (
    accounts
    .groupBy("customer_id")
    .agg(

        # Account count
        F.countDistinct(
            "account_id"
        ).alias("total_accounts"),

        # Active / open accounts
        F.sum(
            F.when(
                F.lower(
                    F.col("status")
                ).isin(
                    "active",
                    "open"
                ),
                1
            ).otherwise(0)
        ).alias("active_accounts"),

        # Total balances
        F.sum(
            F.col("balance").cast("double")
        ).alias("total_balance"),

        # Average balance
        F.avg(
            F.col("balance").cast("double")
        ).alias("average_balance"),

        # Largest account balance
        F.max(
            F.col("balance").cast("double")
        ).alias("largest_balance"),

        # Credit limit
        F.sum(
            F.col("credit_limit").cast("double")
        ).alias("total_credit_limit"),

        F.max(
            F.col("credit_limit").cast("double")
        ).alias("maximum_credit_limit"),

        # Credit-card accounts
        F.sum(
            F.when(
                F.lower(
                    F.col("account_type")
                ).isin(
                    "credit_card",
                    "credit card",
                    "credit"
                ),
                1
            ).otherwise(0)
        ).alias("credit_accounts")
    )
)



print("Aggregating transaction information...")


transactions_enriched = (
    transactions
    .withColumn(
        "transaction_timestamp",
        F.col(
            "transaction_timestamp"
        ).cast("timestamp")
    )
    .withColumn(
        "transaction_month",
        F.date_format(
            F.col("transaction_timestamp"),
            "yyyy-MM"
        )
    )
)


transaction_agg = (
    transactions_enriched
    .groupBy("customer_id")
    .agg(

        # Number of transactions
        F.count(
            "transaction_id"
        ).alias("transaction_count"),

        # Number of active transaction months
        F.countDistinct(
            "transaction_month"
        ).alias(
            "active_transaction_months"
        ),

        # Total transaction value
        F.sum(
            F.col("amount").cast("double")
        ).alias(
            "total_transaction_amount"
        ),

        # Average transaction
        F.avg(
            F.col("amount").cast("double")
        ).alias(
            "average_transaction_amount"
        ),

        # Largest transaction
        F.max(
            F.col("amount").cast("double")
        ).alias(
            "largest_transaction_amount"
        ),

        # Smallest transaction
        F.min(
            F.col("amount").cast("double")
        ).alias(
            "smallest_transaction_amount"
        )
    )
)


print("Calculating loan exposure...")


loans_enriched = (
    loans

    .withColumn(
        "principal_double",
        F.col("principal").cast("double")
    )

    .withColumn(
        "interest_rate_double",
        F.col("interest_rate").cast("double")
    )

    .withColumn(
        "term_months_double",
        F.col("term_months").cast("double")
    )

    .withColumn(
        "monthly_interest_rate",
        F.col("interest_rate").cast("double")
        / 12.0
        / 100.0
    )
)


loans_enriched = (
    loans_enriched

    .withColumn(
        "estimated_monthly_payment",

        F.when(
            F.col("term_months_double") <= 0,
            F.lit(0.0)
        )

        # Zero interest
        .when(
            F.col("monthly_interest_rate") == 0,
            F.col("principal_double")
            /
            F.col("term_months_double")
        )

        # Standard amortisation
        .otherwise(
            (
                F.col("principal_double")
                *
                F.col("monthly_interest_rate")
                *
                F.pow(
                    1 +
                    F.col("monthly_interest_rate"),
                    F.col("term_months_double")
                )
            )
            /
            (
                F.pow(
                    1 +
                    F.col("monthly_interest_rate"),
                    F.col("term_months_double")
                )
                - 1
            )
        )
    )
)


loan_agg = (
    loans_enriched
    .groupBy("customer_id")
    .agg(

        # Total loans
        F.countDistinct(
            "loan_id"
        ).alias("total_loans"),

        # Active loans
        F.sum(
            F.when(
                F.lower(
                    F.col("status")
                ).isin(
                    "active",
                    "open"
                ),
                1
            ).otherwise(0)
        ).alias("active_loans"),

        # Historical / total principal
        F.sum(
            F.col("principal_double")
        ).alias("total_debt"),

        # Current active debt
        F.sum(
            F.when(
                F.lower(
                    F.col("status")
                ).isin(
                    "active",
                    "open"
                ),
                F.col("principal_double")
            ).otherwise(0.0)
        ).alias("active_debt"),

        # Average interest rate
        F.avg(
            F.col("interest_rate_double")
        ).alias(
            "average_interest_rate"
        ),

        # Late/defaulted loans
        F.sum(
            F.when(
                F.lower(
                    F.col("status")
                ).isin(
                    "late",
                    "default",
                    "defaulted",
                    "delinquent"
                ),
                1
            ).otherwise(0)
        ).alias(
            "late_or_defaulted_loans"
        ),

        # Estimated repayment for active loans only
        F.sum(
            F.when(
                F.lower(
                    F.col("status")
                ).isin(
                    "active",
                    "open"
                ),
                F.col(
                    "estimated_monthly_payment"
                )
            ).otherwise(0.0)
        ).alias(
            "monthly_repayment"
        )
    )
)


# payment aggregation

print("Aggregating payment behaviour...")


payment_agg = (
    payments
    .groupBy("customer_id")
    .agg(

        # Payment count
        F.count(
            "payment_id"
        ).alias("payment_count"),

        # Total payments
        F.sum(
            F.col("amount").cast("double")
        ).alias("total_payments"),

        # Successful payments
        F.sum(
            F.when(
                F.lower(
                    F.col("payment_status")
                ).isin(
                    "successful",
                    "success",
                    "completed",
                    "paid"
                ),
                1
            ).otherwise(0)
        ).alias(
            "successful_payments"
        ),

        # Failed payments
        F.sum(
            F.when(
                F.lower(
                    F.col("payment_status")
                ).isin(
                    "failed",
                    "failure"
                ),
                1
            ).otherwise(0)
        ).alias(
            "failed_payments"
        ),

        # Late payments
        F.sum(
            F.when(
                F.lower(
                    F.col("payment_status")
                ).isin(
                    "late",
                    "overdue"
                ),
                1
            ).otherwise(0)
        ).alias(
            "late_payments"
        )
    )
)


# campaign aggregation

print("Aggregating campaign engagement...")


campaign_agg = (
    campaign_contacts
    .groupBy("customer_id")
    .agg(

        # Contacts
        F.count(
            "contact_id"
        ).alias(
            "campaign_contacts"
        ),

        # Conversions
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
            "campaign_conversions"
        ),

        # Interested
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
            "campaign_interested"
        )
    )
)


# joining to customer 360 table

print("Building Customer 360...")


customer_360 = (
    customer_base

    .join(
        account_agg,
        on="customer_id",
        how="left"
    )

    .join(
        transaction_agg,
        on="customer_id",
        how="left"
    )

    .join(
        loan_agg,
        on="customer_id",
        how="left"
    )

    .join(
        payment_agg,
        on="customer_id",
        how="left"
    )

    .join(
        campaign_agg,
        on="customer_id",
        how="left"
    )
)


#handling null values

zero_columns = [

    # Accounts
    "total_accounts",
    "active_accounts",
    "total_balance",
    "average_balance",
    "largest_balance",
    "total_credit_limit",
    "maximum_credit_limit",
    "credit_accounts",

    # Transactions
    "transaction_count",
    "active_transaction_months",
    "total_transaction_amount",
    "average_transaction_amount",
    "largest_transaction_amount",
    "smallest_transaction_amount",

    # Loans
    "total_loans",
    "active_loans",
    "total_debt",
    "active_debt",
    "average_interest_rate",
    "late_or_defaulted_loans",
    "monthly_repayment",

    # Payments
    "payment_count",
    "total_payments",
    "successful_payments",
    "failed_payments",
    "late_payments",

    # Campaigns
    "campaign_contacts",
    "campaign_conversions",
    "campaign_interested"
]


for column_name in zero_columns:

    customer_360 = (
        customer_360
        .withColumn(
            column_name,
            F.coalesce(
                F.col(column_name),
                F.lit(0)
            )
        )
    )


#deriving customer metrics

customer_360 = (

    customer_360

    # Monthly income


    .withColumn(
        "monthly_income",
        F.when(
            F.col("annual_income") > 0,
            F.col(
                "annual_income"
            ).cast("double") / 12.0
        ).otherwise(0.0)
    )


    # Debt-to-income ratio


    .withColumn(
        "debt_to_income_ratio",
        F.when(
            F.col("annual_income") > 0,

            F.col("active_debt")
            /
            F.col(
                "annual_income"
            ).cast("double")

        ).otherwise(0.0)
    )

 
    # Debt service ratio


    .withColumn(
        "debt_service_ratio",
        F.when(
            F.col("monthly_income") > 0,

            F.col("monthly_repayment")
            /
            F.col("monthly_income")

        ).otherwise(0.0)
    )


    # Payment success rate


    .withColumn(
        "successful_payment_rate",
        F.when(
            F.col("payment_count") > 0,

            F.col("successful_payments")
            /
            F.col("payment_count")

        ).otherwise(0.0)
    )

    # campaign conversion rate
   

    .withColumn(
        "campaign_conversion_rate",
        F.when(
            F.col("campaign_contacts") > 0,

            F.col("campaign_conversions")
            /
            F.col("campaign_contacts")

        ).otherwise(0.0)
    )

    # customer flags

    .withColumn(
        "has_active_loan",
        F.when(
            F.col("active_loans") > 0,
            True
        ).otherwise(False)
    )

    .withColumn(
        "has_credit_card",
        F.when(
            F.col("credit_accounts") > 0,
            True
        ).otherwise(False)
    )

    .withColumn(
        "has_transaction_activity",
        F.when(
            F.col("transaction_count") > 0,
            True
        ).otherwise(False)
    )

    .withColumn(
        "has_campaign_activity",
        F.when(
            F.col("campaign_contacts") > 0,
            True
        ).otherwise(False)
    )
)


# rounding metric and final data quality checks

round_2_columns = [
    "annual_income",
    "monthly_income",
    "total_balance",
    "average_balance",
    "largest_balance",
    "total_credit_limit",
    "maximum_credit_limit",
    "total_transaction_amount",
    "average_transaction_amount",
    "largest_transaction_amount",
    "smallest_transaction_amount",
    "total_debt",
    "active_debt",
    "average_interest_rate",
    "monthly_repayment",
    "total_payments"
]


for column_name in round_2_columns:

    customer_360 = customer_360.withColumn(
        column_name,
        F.round(
            F.col(column_name).cast("double"),
            2
        )
    )


customer_360 = (
    customer_360

    .withColumn(
        "debt_to_income_ratio",
        F.round(
            "debt_to_income_ratio",
            4
        )
    )

    .withColumn(
        "debt_service_ratio",
        F.round(
            "debt_service_ratio",
            4
        )
    )

    .withColumn(
        "successful_payment_rate",
        F.round(
            "successful_payment_rate",
            4
        )
    )

    .withColumn(
        "campaign_conversion_rate",
        F.round(
            "campaign_conversion_rate",
            4
        )
    )
)




customer_360 = (
    customer_360
    .filter(
        F.col("customer_id").isNotNull()
    )
)


#gold metadata

customer_360 = (

    customer_360

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
        F.lit("customer_360")
    )

    .withColumn(
        "pipeline_version",
        F.lit(PIPELINE_VERSION)
    )
)


  #column order

customer_360 = customer_360.select(

    #customer

    "customer_id",
    "age",
    "gender",
    "education",
    "marital_status",
    "occupation",
    "annual_income",
    "monthly_income",
    "credit_score",
    "customer_since",

    #accounts

    "total_accounts",
    "active_accounts",
    "total_balance",
    "average_balance",
    "largest_balance",
    "total_credit_limit",
    "maximum_credit_limit",
    "credit_accounts",

    #transaction

    "transaction_count",
    "active_transaction_months",
    "total_transaction_amount",
    "average_transaction_amount",
    "largest_transaction_amount",
    "smallest_transaction_amount",
    #loan
    "total_loans",
    "active_loans",
    "total_debt",
    "active_debt",
    "average_interest_rate",
    "late_or_defaulted_loans",
    "monthly_repayment",
    # debt

    "debt_to_income_ratio",
    "debt_service_ratio",

    #payment

    "payment_count",
    "total_payments",
    "successful_payments",
    "failed_payments",
    "late_payments",
    "successful_payment_rate",

    #campaigns

    "campaign_contacts",
    "campaign_conversions",
    "campaign_interested",
    "campaign_conversion_rate",

    #flags

    "has_active_loan",
    "has_credit_card",
    "has_transaction_activity",
    "has_campaign_activity",

    #metadata

    "gold_created_at",
    "source_layer",
    "gold_table",
    "pipeline_version"
)


#writing gold
print("Writing Customer 360 Gold dataset...")


(
    customer_360
    .write
    .mode("overwrite")
    .format("parquet")
    .save(GOLD_PATH)
)


#validation

customer_count = (
    customer_360
    .select("customer_id")
    .distinct()
    .count()
)

column_count = len(
    customer_360.columns
)

print("CUSTOMER 360 GOLD PIPELINE COMPLETE")

print(f"Gold output path: {GOLD_PATH}")
print(f"Customer count:   {customer_count}")
print(f"Column count:     {column_count}")



print("")
print("Gold schema:")
customer_360.printSchema()


print("")
print("Sample records:")
customer_360.show(
    20,
    truncate=False
)

job.commit()
