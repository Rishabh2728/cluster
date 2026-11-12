"""
Exploratory Data Analysis Module for INT315 Cluster Computing Project
Predicting Insurance Claim Approval Using Apache Spark
Student: Rishabh Kumar | Roll No: 49 | Reg No: 12310491 | Section: K2P23PD
"""

from typing import Dict, Any
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def register_claims_view(spark: SparkSession, df: DataFrame) -> None:
    """Registers temporary view for Spark SQL execution."""
    df.createOrReplaceTempView("claims_view")
    print("[EDA] Spark SQL temporary view 'claims_view' registered successfully.")


def compute_claim_kpis(df: DataFrame) -> dict:
    """
    Computes overarching insurance portfolio KPIs:
    1. Total claims
    2. Approved claims
    3. Rejected claims
    4. Approval percentage
    5. Average claim amount
    """
    row = df.agg(
        F.count("claim_id").alias("total_claims"),
        F.sum(F.when(F.col("claim_approval") == "Approved", 1).otherwise(0)).alias("approved_claims"),
        F.sum(F.when(F.col("claim_approval") == "Rejected", 1).otherwise(0)).alias("rejected_claims"),
        F.round(F.avg("claim_amount"), 2).alias("avg_claim_amount"),
        F.round(F.avg("annual_premium"), 2).alias("avg_annual_premium"),
        F.round(F.sum("claim_amount"), 2).alias("total_claimed_amount")
    ).collect()[0]

    total = int(row["total_claims"] or 0)
    approved = int(row["approved_claims"] or 0)
    rejected = int(row["rejected_claims"] or 0)
    approval_pct = round((approved / total) * 100, 2) if total > 0 else 0.0

    kpis = {
        "total_claims": total,
        "approved_claims": approved,
        "rejected_claims": rejected,
        "approval_rate_pct": approval_pct,
        "rejection_rate_pct": round(100.0 - approval_pct, 2),
        "avg_claim_amount": float(row["avg_claim_amount"] or 0.0),
        "avg_annual_premium": float(row["avg_annual_premium"] or 0.0),
        "total_claimed_amount": float(row["total_claimed_amount"] or 0.0)
    }

    print("\n================== INSURANCE CLAIM SUMMARY KPIS ==================")
    print(f"Total Claims Analyzed  : {kpis['total_claims']:,}")
    print(f"Approved Claims        : {kpis['approved_claims']:,} ({kpis['approval_rate_pct']}%)")
    print(f"Rejected Claims        : {kpis['rejected_claims']:,} ({kpis['rejection_rate_pct']}%)")
    print(f"Average Claim Amount   : ${kpis['avg_claim_amount']:,.2f}")
    print(f"Total Claim Volume     : ${kpis['total_claimed_amount']:,.2f}")
    print("==================================================================\n")

    return kpis


def analyze_approval_by_policy_type(spark: SparkSession) -> DataFrame:
    """Analysis 6: Approval rate by insurance policy type."""
    query = """
        SELECT 
            policy_type,
            COUNT(claim_id) AS total_claims,
            SUM(CASE WHEN claim_approval = 'Approved' THEN 1 ELSE 0 END) AS approved_count,
            SUM(CASE WHEN claim_approval = 'Rejected' THEN 1 ELSE 0 END) AS rejected_count,
            ROUND(AVG(CASE WHEN claim_approval = 'Approved' THEN 100.0 ELSE 0.0 END), 2) AS approval_rate_pct,
            ROUND(AVG(claim_amount), 2) AS avg_claim_amount
        FROM claims_view
        GROUP BY policy_type
        ORDER BY approval_rate_pct DESC
    """
    return spark.sql(query)


def analyze_approval_by_documentation(spark: SparkSession) -> DataFrame:
    """Analysis 7: Approval rate by documentation verification status."""
    query = """
        SELECT 
            documentation_status,
            COUNT(claim_id) AS total_claims,
            SUM(CASE WHEN claim_approval = 'Approved' THEN 1 ELSE 0 END) AS approved_count,
            SUM(CASE WHEN claim_approval = 'Rejected' THEN 1 ELSE 0 END) AS rejected_count,
            ROUND(AVG(CASE WHEN claim_approval = 'Approved' THEN 100.0 ELSE 0.0 END), 2) AS approval_rate_pct
        FROM claims_view
        GROUP BY documentation_status
        ORDER BY approval_rate_pct DESC
    """
    return spark.sql(query)


def analyze_approval_by_severity(spark: SparkSession) -> DataFrame:
    """Analysis 8: Approval rate by incident severity tier."""
    query = """
        SELECT 
            incident_severity,
            COUNT(claim_id) AS total_claims,
            ROUND(AVG(CASE WHEN claim_approval = 'Approved' THEN 100.0 ELSE 0.0 END), 2) AS approval_rate_pct,
            ROUND(AVG(claim_amount), 2) AS avg_claim_amount,
            ROUND(AVG(repair_or_treatment_estimate), 2) AS avg_estimate
        FROM claims_view
        GROUP BY incident_severity
        ORDER BY avg_claim_amount DESC
    """
    return spark.sql(query)


def analyze_approval_by_payment_status(spark: SparkSession) -> DataFrame:
    """Analysis 9: Approval rate by policyholder premium payment standing."""
    query = """
        SELECT 
            premium_payment_status,
            COUNT(claim_id) AS total_claims,
            SUM(CASE WHEN claim_approval = 'Approved' THEN 1 ELSE 0 END) AS approved_count,
            SUM(CASE WHEN claim_approval = 'Rejected' THEN 1 ELSE 0 END) AS rejected_count,
            ROUND(AVG(CASE WHEN claim_approval = 'Approved' THEN 100.0 ELSE 0.0 END), 2) AS approval_rate_pct
        FROM claims_view
        GROUP BY premium_payment_status
        ORDER BY approval_rate_pct DESC
    """
    return spark.sql(query)


def analyze_approval_by_tenure_category(spark: SparkSession) -> DataFrame:
    """Analysis 10: Approval rate by policy tenure cohort."""
    query = """
        SELECT 
            policy_tenure_category,
            COUNT(claim_id) AS total_claims,
            ROUND(AVG(CASE WHEN claim_approval = 'Approved' THEN 100.0 ELSE 0.0 END), 2) AS approval_rate_pct,
            ROUND(AVG(annual_premium), 2) AS avg_premium
        FROM claims_view
        GROUP BY policy_tenure_category
        ORDER BY approval_rate_pct DESC
    """
    return spark.sql(query)


def analyze_claim_amount_distribution(spark: SparkSession) -> DataFrame:
    """Analysis 11: Statistical distribution of claim amounts by approval outcome."""
    query = """
        SELECT 
            claim_approval,
            COUNT(claim_id) AS claim_count,
            ROUND(MIN(claim_amount), 2) AS min_amount,
            ROUND(AVG(claim_amount), 2) AS mean_amount,
            ROUND(MAX(claim_amount), 2) AS max_amount,
            ROUND(STDDEV(claim_amount), 2) AS stddev_amount
        FROM claims_view
        GROUP BY claim_approval
    """
    return spark.sql(query)


def analyze_previous_claims_vs_approval(spark: SparkSession) -> DataFrame:
    """Analysis 12: Claim history volume versus approval rate."""
    query = """
        SELECT 
            previous_claims,
            COUNT(claim_id) AS total_claims,
            SUM(CASE WHEN claim_approval = 'Approved' THEN 1 ELSE 0 END) AS approved_count,
            ROUND(AVG(CASE WHEN claim_approval = 'Approved' THEN 100.0 ELSE 0.0 END), 2) AS approval_rate_pct
        FROM claims_view
        GROUP BY previous_claims
        ORDER BY previous_claims ASC
    """
    return spark.sql(query)


def run_exploratory_analysis(spark: SparkSession, df: DataFrame) -> Dict[str, Any]:
    """
    Coordinates execution of all Spark-based exploratory data analyses.
    """
    register_claims_view(spark, df)

    kpis = compute_claim_kpis(df)
    policy_type_df = analyze_approval_by_policy_type(spark)
    doc_df = analyze_approval_by_documentation(spark)
    severity_df = analyze_approval_by_severity(spark)
    payment_df = analyze_approval_by_payment_status(spark)
    tenure_df = analyze_approval_by_tenure_category(spark)
    amount_dist_df = analyze_claim_amount_distribution(spark)
    prev_claims_df = analyze_previous_claims_vs_approval(spark)

    return {
        "kpis": kpis,
        "approval_by_policy_type": policy_type_df,
        "approval_by_documentation": doc_df,
        "approval_by_severity": severity_df,
        "approval_by_payment_status": payment_df,
        "approval_by_tenure": tenure_df,
        "claim_amount_distribution": amount_dist_df,
        "previous_claims_vs_approval": prev_claims_df
    }
