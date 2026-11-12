"""
Unit Tests for Machine Learning Pipeline, Evaluation, and Partitioning
INT315 - Cluster Computing Project
Student: Rishabh Kumar | Roll No: 49 | Reg No: 12310491
"""

from pathlib import Path
import pytest
from src.config import DATA_PROCESSED_DIR
from src.data_loader import load_claims_data
from src.data_cleaning import clean_claims_data
from src.feature_engineering import apply_feature_engineering
from src.model_training import train_models, split_data
from src.model_evaluation import evaluate_binary_classifier
from src.prediction import predict_single_claim
from src.performance import demonstrate_partitioning_mechanics


@pytest.fixture(scope="module")
def sample_claims_df(spark):
    """Loads a 200-row sample fixture from cleaned claims sample for fast model testing."""
    sample_csv = DATA_PROCESSED_DIR / "cleaned_claims_sample.csv"
    if not sample_csv.exists():
        pytest.skip("Run pipeline first to generate sample CSV")
    
    # Read sample CSV into Spark DataFrame
    df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(str(sample_csv))
    return df.limit(200)


def test_model_training_and_evaluation(spark, sample_claims_df):
    """Verify that both MLlib pipelines train and evaluate without errors on sample data."""
    train_df, test_df = split_data(sample_claims_df, train_ratio=0.75, seed=42)

    models_dict = train_models(train_df, test_df)

    assert "logistic_regression" in models_dict
    assert "random_forest" in models_dict

    lr_preds = models_dict["logistic_regression"]["predictions"]
    rf_preds = models_dict["random_forest"]["predictions"]

    assert "prediction" in lr_preds.columns
    assert "probability" in lr_preds.columns
    assert "prediction" in rf_preds.columns

    lr_eval = evaluate_binary_classifier(lr_preds, "Logistic Regression")
    assert 0.0 <= lr_eval["accuracy"] <= 1.0
    assert 0.0 <= lr_eval["roc_auc"] <= 1.0


def test_single_prediction_inference(spark, sample_claims_df):
    """Verify inference engine on new sample claim record."""
    train_df, test_df = split_data(sample_claims_df, train_ratio=0.8, seed=42)
    models_dict = train_models(train_df, test_df)
    lr_model = models_dict["logistic_regression"]["model"]

    sample_claim = {
        "policy_type": "Health",
        "claim_amount": 4000.0,
        "annual_premium": 900.0,
        "policy_tenure_months": 24,
        "customer_age": 35,
        "incident_type": "Medical Emergency",
        "incident_severity": "Low",
        "previous_claims": 0,
        "documentation_status": "Complete",
        "policy_active": "Yes",
        "premium_payment_status": "Paid",
        "claim_submission_delay": 2,
        "incident_reported": "Yes",
        "repair_or_treatment_estimate": 4000.0
    }

    result = predict_single_claim(spark, lr_model, sample_claim)

    assert result["predicted_outcome"] in ["Approved", "Rejected"]
    assert 0.0 <= result["approval_probability"] <= 100.0
    assert 0.0 <= result["rejection_probability"] <= 100.0
    assert "disclaimer" in result


def test_partitioning_mechanics(spark, sample_claims_df):
    """Verify partition inspection, repartition, and coalesce."""
    demo = demonstrate_partitioning_mechanics(sample_claims_df)
    assert demo["repartitioned_count"] == 4
    assert demo["coalesced_count"] == 2
    assert demo["aggregation_time_seconds"] > 0
