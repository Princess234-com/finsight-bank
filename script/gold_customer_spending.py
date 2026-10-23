import sys

from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.sql.functions import (
    col,
    lit,
    sum as spark_sum,
    count,
    avg,
    min as spark_min,
    max as spark_max,
    countDistinct,
    when,
    current_timestamp,
    datediff,
    to_date,
    row_number
)

from pyspark.sql.window import Window


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

CUSTOMERS_SILVER_PATH = (
    "s3://finsight-bank-data/"
    "silver/customers/"
)

ACCOUNTS_SILVER_PATH = (
    "s3://finsight-bank-data/"
    "silver/accounts/"
)

TRANSACTIONS_SILVER_PATH = (
    "s3://finsight-bank-data/"
    "silver/transactions/"
)

GOLD_PATH = (
    "s3://finsight-bank-data/"
    "gold/customer_spending/"
)

PIPELINE_VERSION = "1.0"

#reading table

print("GOLD CUSTOMER SPENDING PIPELINE")

print("Reading Silver customers...")

customers = spark.read.parquet(
    CUSTOMERS_SILVER_PATH
)

print("Reading Silver accounts...")

accounts = spark.read.parquet(
    ACCOUNTS_SILVER_PATH
)

print("Reading Silver transactions...")

transactions = spark.read.parquet(
    TRANSACTIONS_SILVER_PATH
)

customer_count = customers.count()
account_count = accounts.count()
transaction_count = transactions.count()

print(f"Silver customers:    {customer_count}")
print(f"Silver accounts:     {account_count}")
print(f"Silver transactions: {transaction_count}")

print("Preparing customer dimension...")

customer_dim = (
    customers
    .select(
        col("customer_id").cast("long").alias("customer_id"),
        *[
            col(c)
            for c in customers.columns
            if c != "customer_id"
        ]
    )
    .dropDuplicates(["customer_id"])
)

# prepare account and transaction data

print("Preparing account dimension...")

account_dim = (
    accounts
    .select(
        col("account_id").cast("long").alias("account_id"),
        col("customer_id").cast("long").alias("account_customer_id")
    )
    .dropDuplicates(["account_id"])
)

print("Preparing transactions...")

transaction_fact = (
    transactions
    .select(
        col("transaction_id").cast("long").alias("transaction_id"),
        col("customer_id").cast("long").alias("transaction_customer_id"),
        col("account_id").cast("long").alias("transaction_account_id"),
        col("transaction_timestamp").cast("timestamp").alias(
            "transaction_timestamp"
        ),
        col("amount").cast("decimal(14,2)").alias("amount")
    )
)


#joing table to account table

print("Joining transactions to accounts...")

transactions_with_accounts = (
    transaction_fact.alias("t")
    .join(
        account_dim.alias("a"),
        col("t.transaction_account_id") == col("a.account_id"),
        "left"
    )
    .select(
        col("t.transaction_id"),
        col("t.transaction_customer_id"),
        col("t.transaction_account_id"),
        col("t.transaction_timestamp"),
        col("t.amount"),
        col("a.account_customer_id")
    )
)

#constistency check

print("Checking account/customer relationships...")

transactions_valid = (
    transactions_with_accounts
    .filter(
        col("transaction_customer_id").isNotNull()
    )
    .filter(
        col("transaction_account_id").isNotNull()
    )
    .filter(
        col("amount").isNotNull()
    )
    .filter(
        col("amount") > 0
    )
    .filter(
        col("transaction_timestamp").isNotNull()
    )
    .filter(
        col("account_customer_id").isNotNull()
    )
    .filter(
        col("transaction_customer_id")
        == col("account_customer_id")
    )
)


# adding transaction date

transactions_valid = (
    transactions_valid
    .withColumn(
        "transaction_date",
        to_date(col("transaction_timestamp"))
    )
)


#customer spending metrics

print("Calculating customer spending metrics...")

customer_spending = (
    transactions_valid
    .groupBy(
        col("transaction_customer_id").alias("customer_id")
    )
    .agg(
        spark_sum("amount").alias("total_spending"),

        count("transaction_id").alias(
            "transaction_count"
        ),

        countDistinct("transaction_account_id").alias(
            "active_account_count"
        ),

        avg("amount").alias(
            "average_transaction_amount"
        ),

        spark_min("transaction_timestamp").alias(
            "first_transaction_timestamp"
        ),

        spark_max("transaction_timestamp").alias(
            "last_transaction_timestamp"
        ),

        spark_min("transaction_date").alias(
            "first_transaction_date"
        ),

        spark_max("transaction_date").alias(
            "last_transaction_date"
        )
    )
)


#calculating other metrics

print("Calculating additional customer metrics...")

customer_spending = (
    customer_spending

    .withColumn(
        "average_transaction_amount",
        col("average_transaction_amount").cast("decimal(14,2)")
    )

    .withColumn(
        "total_spending",
        col("total_spending").cast("decimal(18,2)")
    )

    .withColumn(
        "customer_lifetime_days",
        datediff(
            col("last_transaction_date"),
            col("first_transaction_date")
        )
    )

    .withColumn(
        "spending_per_transaction",
        (
            col("total_spending")
            / col("transaction_count")
        ).cast("decimal(18,2)")
    )
)


#spending segment

print("Creating spending segments...")

customer_spending = (
    customer_spending
    .withColumn(
        "spending_segment",
        when(
            col("total_spending") < 1000,
            lit("LOW")
        )
        .when(
            col("total_spending") < 5000,
            lit("MEDIUM")
        )
        .when(
            col("total_spending") < 10000,
            lit("HIGH")
        )
        .otherwise(
            lit("PREMIUM")
        )
    )
)

#join to customer table

print("Joining customer attributes...")

customer_columns = customer_dim.columns

customer_attribute_columns = [
    c
    for c in customer_columns
    if c != "customer_id"
]

customer_spending = (
    customer_spending.alias("s")
    .join(
        customer_dim.alias("c"),
        col("s.customer_id") == col("c.customer_id"),
        "left"
    )
    .select(
        col("s.customer_id"),
        *[
            col(f"c.{c}").alias(c)
            for c in customer_attribute_columns
        ],
        col("s.total_spending"),
        col("s.transaction_count"),
        col("s.active_account_count"),
        col("s.average_transaction_amount"),
        col("s.spending_per_transaction"),
        col("s.first_transaction_timestamp"),
        col("s.last_transaction_timestamp"),
        col("s.first_transaction_date"),
        col("s.last_transaction_date"),
        col("s.customer_lifetime_days"),
        col("s.spending_segment")
    )
)

gold_customer_spending = (
    customer_spending

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
        lit("synthetic_banking_transactions")
    )
)

print("Running final Gold data quality checks...")

invalid_gold = (
    gold_customer_spending
    .filter(
        col("customer_id").isNull()
    )
    .count()
)

print(
    f"Invalid Gold customer records: {invalid_gold}"
)


if invalid_gold > 0:
    raise ValueError(
        "Gold customer spending contains "
        "records with NULL customer_id."
    )
# removing dedup

gold_customer_spending = (
    gold_customer_spending
    .dropDuplicates(["customer_id"])
)

print("Writing Gold customer spending...")

(
    gold_customer_spending
    .write
    .mode("overwrite")
    .parquet(GOLD_PATH)
)

gold_count = gold_customer_spending.count()

total_spending = (
    gold_customer_spending
    .agg(
        spark_sum("total_spending").alias(
            "total_spending"
        )
    )
    .collect()[0]["total_spending"]
)

total_transactions = (
    gold_customer_spending
    .agg(
        spark_sum("transaction_count").alias(
            "transaction_count"
        )
    )
    .collect()[0]["transaction_count"]
)

print("GOLD CUSTOMER SPENDING COMPLETE")
print(f"Silver customers:       {customer_count}")
print(f"Silver accounts:        {account_count}")
print(f"Silver transactions:    {transaction_count}")
print(f"Gold customers:         {gold_count}")
print(f"Gold transactions:      {total_transactions}")
print(f"Total customer spending:{total_spending}")
print(f"Gold location:          {GOLD_PATH}")

job.commit()
