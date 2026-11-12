"""
Unit Tests for Data Cleaning and Feature Engineering
INT315 - Cluster Computing Project
Student: Rishabh Kumar | Roll No: 49 | Reg No: 12310491
"""

import tempfile
from pathlib import Path
import pytest
from src.data_loader import get_insurance_claims_schema
from src.data_cleaning import clean_claims_data
from src.feature_engineering import apply_feature_engineering
from src.model_training import build_feature_pipeline_stages


def get_mock_claims_df(spark):
    """Loads a small deterministic CSV fixture for fast, robust unit testing."""
    schema = get_insurance_claims_schema()
    csv_content = (
        "claim_id,policy_id,policy_type,claim_amount,annual_premium,policy_tenure_months,"
        "customer_age,incident_type,incident_severity,previous_claims,documentation_status,"
        "policy_active,premium_payment_status,claim_submission_delay,incident_reported,"
        "repair_or_treatment_estimate,claim_approval\n"
        "CLM-001,POL-101,Health,5000.0,1000.0,24,40,Medical Emergency,Medium,0,Complete,Yes,Paid,2,Yes,4800.0,Approved\n"
        "CLM-002,POL-102,Motor,12000.0,1500.0,6,25,Vehicle Collision,High,3,Missing,No,Overdue,45,No,8000.0,Rejected\n"
        "CLM-001,POL-101,Health,5000.0,1000.0,24,40,Medical Emergency,Medium,0,Complete,Yes,Paid,2,Yes,4800.0,Approved\n"
        "CLM-003,POL-103,Property,8000.0,1200.0,10,,Water Damage,Low,,Partial,Yes,Pending,10,Yes,,Approved\n"
        "CLM-004,POL-104,Life,-500.0,1000.0,12,50,Critical Illness,High,0,Complete,Yes,Paid,5,Yes,1000.0,Approved\n"
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(csv_content)
        temp_path = f.name

    return spark.read.format("csv").option("header", "true").schema(schema).load(temp_path)


def test_data_cleaning_and_imputation(spark):
    """Verify deduplication, validation, and imputation."""
    raw_df = get_mock_claims_df(spark)
    df_clean, metrics = clean_claims_data(raw_df)

    # 5 input rows: 1 duplicate, 1 negative claim dropped -> 3 clean rows
    assert df_clean.count() == 3
    assert metrics["duplicate_records_dropped"] == 1
    assert metrics["invalid_records_dropped"] == 1

    # Check that imputed row has valid values
    imputed_row = df_clean.filter(df_clean.claim_id == "CLM-003").collect()[0]
    assert imputed_row.customer_age == 44  # Imputed default
    assert imputed_row.previous_claims == 0  # Imputed default
    assert imputed_row.repair_or_treatment_estimate == 8000.0  # Imputed with claim_amount


def test_feature_engineering_calculations(spark):
    """Verify calculation of derived financial ratios and categorical tiers."""
    raw_df = get_mock_claims_df(spark)
    df_clean, _ = clean_claims_data(raw_df)
    df_feat = apply_feature_engineering(df_clean)

    rows = df_feat.collect()
    row1 = next(r for r in rows if r["claim_id"] == "CLM-001")

    # claim_to_premium = 5000 / 1000 = 5.0
    assert row1["claim_to_premium_ratio"] == 5.0
    # claim_to_estimate = 5000 / 4800 = 1.0417
    assert abs(row1["claim_to_estimate_ratio"] - 1.0417) < 0.001
    assert row1["policy_tenure_category"] == "Established (12-48m)"
    assert row1["submission_delay_category"] == "Immediate (<= 3d)"
    assert row1["previous_claim_category"] == "None (0)"
    assert row1["label"] == 1.0  # Approved -> 1.0


def test_data_leakage_prevention():
    """Verify that identifiers and raw target labels are excluded from ML pipeline."""
    _, assembler_inputs = build_feature_pipeline_stages()

    assert "claim_id" not in assembler_inputs
    assert "policy_id" not in assembler_inputs
    assert "claim_approval" not in assembler_inputs
    assert "label" not in assembler_inputs
