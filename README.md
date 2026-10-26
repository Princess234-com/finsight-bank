# FinSight — Banking Intelligence Platform

An AWS-based data engineering and machine learning portfolio project using **synthetic banking data**. FinSight transforms raw customer, account, transaction, loan, payment, and marketing records into queryable analytics datasets, then explores customer behaviour through machine learning.

## Business questions

- How do customers differ in transaction activity and transaction value?
- Which customers currently have loans marked late?

### Core synthetic banking datasets

| Dataset | Key columns | Purpose |
| --- | --- | --- |
| customers | customer_id, age, gender, education, marital_status, occupation, annual_income, credit_score, customer_since | Customer profiles |
| accounts | account_id, customer_id, account_type, opened_date, balance, credit_limit, status | Customer accounts |
| transactions | transaction_id, customer_id, account_id, transaction_timestamp, amount | Transaction activity |
| loans | loan_id, customer_id, loan_type, principal, interest_rate, term_months, status, loan_start_date | Loan records |
| payments | payment_id, customer_id, account_id, payment_date, amount, payment_status | Payment outcomes |
| campaign_contacts | contact_id, customer_id, campaign_id, contact_date, channel, duration_seconds, outcome | Marketing interactions |




## Medallion data architecture
<img width="762" height="390" alt="image" src="https://github.com/user-attachments/assets/8b9dc566-07c1-4f25-beaf-3da2137e1803" />


### Bronze — Raw ingestion

Source CSV files are uploaded to an S3 bucket.

### Silver — Cleaning and validation

AWS Glue PySpark jobs convert raw records to Parquet datasets. Implemented transformation logic includes:

- Type conversion and required-field validation.
- Referential-integrity checks against customers and accounts.
- Verification that an account belongs to the referenced customer.
- Duplicate transaction removal.
- Quarantine of invalid transaction records.
- Ingestion timestamps, source-system labels, and pipeline versions.

### Gold — Analytical datasets

| Dataset | Use |
| --- | --- |
| customer_360 | Unified customer-level account, transaction, loan, payment, and campaign metrics |
| customer_spending | Transaction frequency, totals, averages, activity dates, and flags |
| loan_risk | Loan risk analysis |
| credit_risk | Credit risk analysis |
| campaign_performance | Marketing performance measures |
| marketing_campaign | Campaign analysis |
| ml_features | Customer-level behavioural features |
| ml_training_data | Features plus a loan-late classification label |



Gold datasets are stored in Parquet and queried with Athena through the Glue Data Catalog. The Gold layer is **analytics-ready storage**, not a separately provisioned Redshift warehouse.

## Q1
- How do customers differ in transaction activity and transaction value?
<img width="1222" height="747" alt="image" src="https://github.com/user-attachments/assets/77b29c7e-4b21-4fe6-be22-8efa0489b350" />

The Insight: Customers are segmented into three distinct behavioral tiers based on transaction count and total spending:
- Inactive / Low Activity (Red): Near-zero transaction counts and minimal spending values.
- Moderate Activity / Spending (Blue): Lower-to-mid transaction frequencies and spending totals.
- High Activity / High Spending (Green): High transaction counts and broad spending volumes (ranging up to $60,000+), though transaction sizes vary widely within this group.

## Q2
- Which customers currently have loans marked late?
  <img width="1602" height="615" alt="image" src="https://github.com/user-attachments/assets/9781e9cd-f9c2-4dd6-b72e-93d18a5c2022" />
  
Out of the analyzed customer base, 12.59% (4,944 customers) have at least one loan currently marked as late (loan_late_target = 1), while 87.41% are don't.

**I attached  pdf files of my Power BI dashboard**

## Technologies

- **Cloud:** AWS S3, AWS Glue, AWS Glue Data Catalog, Amazon Athena, Amazon SageMaker
- **Processing:** Python, PySpark, pandas, NumPy, awswrangler
- **Machine learning:** scikit-learn / K-Means; supervised XGBoost workflow planned
- **Storage and analytics:** CSV, Apache Parquet, SQL
- **PowerBI**


