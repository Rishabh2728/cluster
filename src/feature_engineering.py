"""
Feature Engineering Module for INT315 Cluster Computing Project
Predicting Insurance Claim Approval Using Apache Spark
Student: Rishabh Kumar | Roll No: 49 | Reg No: 12310491 | Section: K2P23PD
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def apply_feature_engineering(df: DataFrame) -> DataFrame:
    """
    Constructs domain-specific derived features for insurance claim adjudication
    using distributed PySpark DataFrame transformations.
    
    Derived Features:
    1. claim_to_premium_ratio = claim_amount / annual_premium
    2. claim_to_estimate_ratio = claim_amount / repair_or_treatment_estimate
    3. policy_tenure_category:
       - 'New (< 12m)'
       - 'Established (12-48m)'
       - 'Mature (> 48m)'
    4. submission_delay_category:
       - 'Immediate (<= 3d)'
       - 'Normal (4-14d)'
       - 'Delayed (> 14d)'
    5. previous_claim_category:
       - 'None (0)'
       - 'Few (1-2)'
       - 'Frequent (3+)'
    6. label: Binary numerical target (1.0 for Approved, 0.0 for Rejected)
    """
    print("[Feature Engineering] Generating derived ratios, categories, and target indices...")

    engineered_df = (
        df
        # 1. Financial Risk Ratios
        .withColumn(
            "claim_to_premium_ratio",
            F.round(F.col("claim_amount") / F.col("annual_premium"), 4)
        )
        .withColumn(
            "claim_to_estimate_ratio",
            F.round(F.col("claim_amount") / F.when(F.col("repair_or_treatment_estimate") > 0, F.col("repair_or_treatment_estimate")).otherwise(F.lit(1.0)), 4)
        )
        # 2. Policy Tenure Category
        .withColumn(
            "policy_tenure_category",
            F.when(F.col("policy_tenure_months") < 12, "New (< 12m)")
            .when(F.col("policy_tenure_months") <= 48, "Established (12-48m)")
            .otherwise("Mature (> 48m)")
        )
        # 3. Submission Delay Category
        .withColumn(
            "submission_delay_category",
            F.when(F.col("claim_submission_delay") <= 3, "Immediate (<= 3d)")
            .when(F.col("claim_submission_delay") <= 14, "Normal (4-14d)")
            .otherwise("Delayed (> 14d)")
        )
        # 4. Previous Claims Category
        .withColumn(
            "previous_claim_category",
            F.when(F.col("previous_claims") == 0, "None (0)")
            .when(F.col("previous_claims") <= 2, "Few (1-2)")
            .otherwise("Frequent (3+)")
        )
        # 5. Numerical target label for Spark ML (1.0 = Approved, 0.0 = Rejected)
        .withColumn(
            "label",
            F.when(F.col("claim_approval") == "Approved", 1.0).otherwise(0.0)
        )
    )

    return engineered_df
