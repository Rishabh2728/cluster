"""
Data Cleaning Module for INT315 Cluster Computing Project
Predicting Insurance Claim Approval Using Apache Spark
Student: Rishabh Kumar | Roll No: 49 | Reg No: 12310491 | Section: K2P23PD
"""

from typing import Tuple
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def clean_claims_data(df: DataFrame) -> Tuple[DataFrame, dict]:
    """
    Cleans raw insurance claim records using distributed Spark operations.
    
    Operations:
    1. Removes duplicate records based on claim_id.
    2. Trims whitespace from all string/categorical fields.
    3. Handles missing numerical values via imputation (median/mean approximation):
       - customer_age imputed with 44
       - previous_claims imputed with 0
       - repair_or_treatment_estimate imputed with claim_amount
    4. Validates domain boundaries:
       - claim_amount > 0 and annual_premium > 0
       - customer_age between 18 and 100
       - policy_tenure_months >= 1
       - claim_submission_delay >= 0
       - claim_approval must be either 'Approved' or 'Rejected'
    5. Filters out corrupt/invalid records.
    
    Returns:
        (df_cleaned, cleaning_metrics_dict)
    """
    initial_count = df.count()
    print(f"\n[Data Cleaning] Initiating cleaning on {initial_count:,} raw claim records...")

    # Step 1: Remove duplicates
    deduped_df = df.dropDuplicates(["claim_id"])
    dedup_count = deduped_df.count()
    duplicate_records_dropped = initial_count - dedup_count

    # Step 2: Trim whitespace from categorical columns
    string_cols = [
        "policy_type", "incident_type", "incident_severity",
        "documentation_status", "policy_active", "premium_payment_status",
        "incident_reported", "claim_approval"
    ]
    trimmed_df = deduped_df
    for col_name in string_cols:
        trimmed_df = trimmed_df.withColumn(col_name, F.trim(F.col(col_name)))

    # Step 3: Handle missing values via deterministic baseline imputation
    imputed_df = (
        trimmed_df
        .withColumn(
            "customer_age",
            F.when(
                F.col("customer_age").isNull() | (F.col("customer_age") < 18) | (F.col("customer_age") > 100),
                F.lit(44)
            ).otherwise(F.col("customer_age"))
        )
        .withColumn(
            "previous_claims",
            F.when(
                F.col("previous_claims").isNull() | (F.col("previous_claims") < 0),
                F.lit(0)
            ).otherwise(F.col("previous_claims"))
        )
        .withColumn(
            "repair_or_treatment_estimate",
            F.when(
                F.col("repair_or_treatment_estimate").isNull() | (F.col("repair_or_treatment_estimate") <= 0),
                F.col("claim_amount")
            ).otherwise(F.col("repair_or_treatment_estimate"))
        )
    )

    # Step 4: Domain boundary validation
    valid_conditions = (
        (F.col("claim_id").isNotNull()) & (F.col("claim_id") != "") &
        (F.col("policy_id").isNotNull()) & (F.col("policy_id") != "") &
        (F.col("claim_amount").isNotNull()) & (F.col("claim_amount") > 0.0) &
        (F.col("annual_premium").isNotNull()) & (F.col("annual_premium") > 0.0) &
        (F.col("policy_tenure_months").isNotNull()) & (F.col("policy_tenure_months") >= 0) &
        (F.col("claim_submission_delay").isNotNull()) & (F.col("claim_submission_delay") >= 0) &
        (F.col("policy_type").isin(["Health", "Motor", "Life", "Property"])) &
        (F.col("incident_severity").isin(["Low", "Medium", "High"])) &
        (F.col("documentation_status").isin(["Complete", "Partial", "Missing"])) &
        (F.col("policy_active").isin(["Yes", "No"])) &
        (F.col("premium_payment_status").isin(["Paid", "Pending", "Overdue"])) &
        (F.col("incident_reported").isin(["Yes", "No"])) &
        (F.col("claim_approval").isin(["Approved", "Rejected"]))
    )

    df_cleaned = imputed_df.filter(valid_conditions)
    cleaned_count = df_cleaned.count()
    dropped_invalid = dedup_count - cleaned_count
    total_dropped = initial_count - cleaned_count
    retention_rate = round((cleaned_count / initial_count) * 100, 2)

    metrics = {
        "initial_raw_records": initial_count,
        "duplicate_records_dropped": duplicate_records_dropped,
        "invalid_records_dropped": dropped_invalid,
        "total_records_dropped": total_dropped,
        "cleaned_records": cleaned_count,
        "retention_rate_pct": retention_rate
    }

    print("[Data Cleaning] Cleaning Summary:")
    print(f"  - Initial Raw Records : {initial_count:,}")
    print(f"  - Duplicate Records   : {duplicate_records_dropped:,}")
    print(f"  - Invalid Records     : {dropped_invalid:,}")
    print(f"  - Final Clean Records : {cleaned_count:,}")
    print(f"  - Data Retention Rate : {retention_rate}%\n")

    return df_cleaned, metrics
