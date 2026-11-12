"""
Unit Tests for Synthetic Insurance Claims Dataset Generator
INT315 - Cluster Computing Project
Student: Rishabh Kumar | Roll No: 49 | Reg No: 12310491
"""

import tempfile
from pathlib import Path
import pandas as pd
import pytest

from src.dataset_generator import generate_synthetic_claims


def test_claims_generation_and_schema():
    """Verify that dataset generator creates valid CSV with all expected columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_csv = Path(tmpdir) / "test_claims.csv"
        target_count = 1000

        stats = generate_synthetic_claims(
            target_records=target_count,
            seed=42,
            output_path=test_csv
        )

        assert test_csv.exists(), "Claims CSV file must exist on disk."
        assert stats["total_records"] == target_count

        df = pd.read_csv(test_csv)
        assert len(df) == target_count

        expected_columns = [
            "claim_id",
            "policy_id",
            "policy_type",
            "claim_amount",
            "annual_premium",
            "policy_tenure_months",
            "customer_age",
            "incident_type",
            "incident_severity",
            "previous_claims",
            "documentation_status",
            "policy_active",
            "premium_payment_status",
            "claim_submission_delay",
            "incident_reported",
            "repair_or_treatment_estimate",
            "claim_approval"
        ]
        assert list(df.columns) == expected_columns


def test_claims_reproducibility():
    """Verify deterministic dataset generation with identical random seed 42."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv1 = Path(tmpdir) / "run1.csv"
        csv2 = Path(tmpdir) / "run2.csv"

        generate_synthetic_claims(target_records=500, seed=42, output_path=csv1)
        generate_synthetic_claims(target_records=500, seed=42, output_path=csv2)

        df1 = pd.read_csv(csv1)
        df2 = pd.read_csv(csv2)

        pd.testing.assert_frame_equal(df1, df2)


def test_target_class_balance_and_anomalies():
    """Verify presence of both target classes and injected missing values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_csv = Path(tmpdir) / "test_balance.csv"
        stats = generate_synthetic_claims(target_records=2000, seed=42, output_path=test_csv)

        assert stats["approved_records"] > 0
        assert stats["rejected_records"] > 0
        # Approval rate should fall within realistic range (60-80%)
        assert 60.0 <= stats["approved_pct"] <= 80.0
        assert stats["injected_missing_records"] > 0
