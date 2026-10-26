# FinSight — Banking Intelligence Platform

An AWS-based data engineering and machine learning portfolio project using **synthetic banking data**. FinSight transforms raw customer, account, transaction, loan, payment, and marketing records into queryable analytics datasets, then explores customer behaviour through machine learning.

> **Project status:** Bronze, Silver, Gold, Athena analytics, ML training-data preparation, and exploratory K-Means customer segmentation have been implemented. Supervised model training/evaluation, prediction APIs, dashboards, and production orchestration are future work unless subsequently documented.

## Business questions

- How do customers differ in transaction activity and transaction value?
- Which customers currently have loans marked late?
- What customer-level signals might be useful for modelling loan delinquency?
- How do account activity and marketing outcomes vary across customers?

## Architecture

```text
Synthetic banking CSVs + independent reference datasets
                       |
                       v
              Amazon S3 — Bronze/raw
                       |
                       v
               AWS Glue / PySpark
        (validation, deduplication, typing,
         referential-integrity checks)
                       |
                       v
             Amazon S3 — Silver/Parquet
                       |
                       v
               AWS Glue / PySpark
          (aggregations, joins, features)
                       |
                       v
              Amazon S3 — Gold/Parquet
                       |
                Glue Data Catalog
                       |
                 Amazon Athena
                       |
               ML-ready dataset
                       |
            SageMaker / Python analysis
                       |
             K-Means segmentation

Planned: supervised classification, model evaluation,
         prediction API, dashboards, orchestration/monitoring
```

**Warehouse approach:** Gold is a curated analytical layer within an S3 data lake, queried with Athena and the Glue Data Catalog. A separate Amazon Redshift warehouse is not required for the current architecture.

## Data sources

### Core synthetic banking datasets

| Dataset | Key columns | Purpose |
| --- | --- | --- |
| `customers` | `customer_id`, `age`, `gender`, `education`, `marital_status`, `occupation`, `annual_income`, `credit_score`, `customer_since` | Customer profiles |
| `accounts` | `account_id`, `customer_id`, `account_type`, `opened_date`, `balance`, `credit_limit`, `status` | Customer accounts |
| `transactions` | `transaction_id`, `customer_id`, `account_id`, `transaction_timestamp`, `amount` | Transaction activity |
| `loans` | `loan_id`, `customer_id`, `loan_type`, `principal`, `interest_rate`, `term_months`, `status`, `loan_start_date` | Loan records |
| `payments` | `payment_id`, `customer_id`, `account_id`, `payment_date`, `amount`, `payment_status` | Payment outcomes |
| `campaign_contacts` | `contact_id`, `customer_id`, `campaign_id`, `contact_date`, `channel`, `duration_seconds`, `outcome` | Marketing interactions |

### Independent reference datasets

- `credit_default`: credit limits, repayment statuses, bills, payments, and a `default_next_month` label.
- `bank_marketing`: demographics, contact history, campaign characteristics, and a `subscribed` label.

These two reference datasets are **not linked to the same customers** as the synthetic banking datasets and should not be joined on customer identifiers without a valid mapping.

## Medallion data architecture

### Bronze — Raw ingestion

Source CSV files are retained in Amazon S3 for traceability and reproducibility.

### Silver — Cleaning and validation

AWS Glue PySpark jobs convert raw records to typed Parquet datasets. Implemented transaction-processing logic includes:

- Type conversion and required-field validation.
- Referential-integrity checks against customers and accounts.
- Verification that an account belongs to the referenced customer.
- Duplicate transaction removal.
- Quarantine of invalid transaction records.
- Ingestion timestamps, source-system labels, and pipeline versions.

### Gold — Analytical datasets

| Dataset | Analytical purpose |
| --- | --- |
| `customer_360` | Unified customer-level account, transaction, loan, payment, and campaign metrics |
| `customer_spending` | Transaction frequency, totals, averages, activity dates, and flags |
| `loan_risk` | Loan risk analysis |
| `credit_risk` | Credit risk analysis |
| `campaign_performance` | Marketing performance measures |
| `marketing_campaign` | Campaign analysis |
| `ml_features` | Customer-level behavioural features |
| `ml_training_data` | Features plus a loan-late classification label |

Gold datasets are stored in Parquet and queried with Athena through the Glue Data Catalog. The Gold layer is **analytics-ready storage**, not a separately provisioned Redshift warehouse.

## Analytics with Amazon Athena

Example customer-level delinquency distribution query:

```sql
SELECT
    loan_late_target,
    COUNT(*) AS customer_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS percentage
FROM banking_gold.ml_training_data
GROUP BY loan_late_target
ORDER BY loan_late_target;
```

Observed training dataset:

| Loan-late label | Customers | Share |
| --- | ---: | ---: |
| `0` | 34,312 | 87.41% |
| `1` | 4,944 | 12.59% |
| **Total** | **39,256** | **100%** |

The label represents **currently observed loan status**: `1` indicates a customer with at least one loan marked `Late`; it is not a validated prediction of a future default event. Training a prospective risk model will require an explicit as-of date, a subsequent outcome window, and leakage checks.

## Machine learning

### Loan delinquency classification — data preparation

A customer-level dataset with **39,256 rows and 31 source columns** was produced and loaded into a SageMaker notebook from S3 using `awswrangler`.

Preparation completed or investigated:

- Checked the label distribution and class imbalance.
- Identified **8,887** missing `days_since_last_transaction` values.
- Confirmed these correspond to customers with no transaction history.
- Added a `has_transaction_history` indicator and filled recency with zero, retaining the indicator to distinguish no history from recent activity.
- Identified numeric columns loaded as `object` types requiring conversion.
- Corrected the proposed feature list to exclude `total_loans`, which is not present in `ml_training_data`.

**Next supervised ML steps:** validate types, stratified train/validation/test split, fit preprocessing on training data only, train and evaluate a baseline and XGBoost model, and report precision, recall, F1, ROC-AUC/PR-AUC, and confusion matrices. Avoid interpreting accuracy alone because the positive class is approximately 12.59%.

### K-Means customer segmentation — exploratory analysis

K-Means clustering has been explored for customer segmentation, visualised using **transaction count** and **total transaction amount**.

![K-Means customer segmentation scatter plot](images/customer_segments.png)

The displayed cluster labels are:

| Segment | Plot colour | Observed pattern | Potential use |
| --- | --- | --- | --- |
| Inactive / Low Activity | Red | Near-zero transaction count and amount | Investigate inactivity and re-engagement |
| Moderate Activity / Spending | Blue | Generally lower counts and transaction totals | Understand engagement opportunities |
| High Activity / High Spending | Green | Broadly higher transaction counts and totals | Explore retention and customer-value strategies |

**Interpretation:** transaction count and total transaction amount are positively associated. The green group spans a wide range of transaction sizes and frequencies, so additional features such as average transaction amount may reveal finer subsegments. The blue and green groups overlap in this two-dimensional projection.

**Limitations:** The plot alone does not establish cluster sizes, optimal `k`, silhouette score, model inputs, or stability. These should be documented from the notebook. The dataset has no transaction direction/type, so the metric called *spending* in the chart is more precisely **total recorded transaction amount**, not confirmed outgoing expenditure. Segments are exploratory and should not be treated as validated financial-risk classifications.

## Technologies

- **Cloud:** AWS S3, AWS Glue, AWS Glue Data Catalog, Amazon Athena, Amazon SageMaker
- **Processing:** Python, PySpark, pandas, NumPy, awswrangler
- **Machine learning:** scikit-learn / K-Means; supervised XGBoost workflow planned
- **Storage and analytics:** CSV, Apache Parquet, SQL

## Suggested repository structure

```text
finsight-banking-intelligence/
├── README.md
├── images/
│   └── customer_segments.png
├── glue_jobs/
│   ├── silver/
│   └── gold/
├── notebooks/
│   ├── customer_segmentation.ipynb
│   └── loan_delinquency_preparation.ipynb
├── sql/
│   └── athena_analysis.sql
└── docs/
    └── data_dictionary.md
```

Add only scripts and notebooks that have been reviewed and verified. Some historical job versions use different output prefixes (`analytics/` versus `gold/`), so paths should be standardised before publishing runnable instructions.

## Known limitations and next steps

1. Validate and document K-Means preprocessing, selected features, `k` selection, cluster counts, and silhouette score.
2. Finish supervised loan-delinquency preprocessing and train/evaluate models.
3. Define a future outcome window to support genuinely predictive loan-risk modelling.
4. Add reproducible execution instructions, parameterised S3 paths, and automated tests.
5. Implement cost-conscious model inference, dashboards, pipeline orchestration, and monitoring.

## Responsible use

All core banking records are synthetic. Results are educational and are **not** suitable for real lending, credit eligibility, or automated customer decisions without additional data governance, fairness assessment, temporal validation, and regulatory review.
