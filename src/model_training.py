"""
Distributed Machine Learning Training Module using Spark MLlib
INT315 - Cluster Computing Project
Predicting Insurance Claim Approval Using Apache Spark
Student: Rishabh Kumar | Roll No: 49 | Reg No: 12310491 | Section: K2P23PD
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.feature import (
    OneHotEncoder,
    StringIndexer,
    VectorAssembler,
)
from pyspark.sql import DataFrame

from src.config import SELECTED_MODEL_DIR


def build_feature_pipeline_stages() -> Tuple[list, list]:
    """
    Constructs the feature transformation stages for Spark MLlib Pipeline.
    Categorical columns are converted via StringIndexer -> OneHotEncoder.
    Continuous and encoded vector columns are assembled via VectorAssembler.
    
    Data Leakage Prevention:
    - Excludes non-predictive keys: claim_id, policy_id
    - Excludes target text: claim_approval
    """
    categorical_cols = [
        "policy_type",
        "incident_type",
        "incident_severity",
        "documentation_status",
        "policy_active",
        "premium_payment_status",
        "incident_reported",
        "policy_tenure_category",
        "submission_delay_category",
        "previous_claim_category"
    ]

    numeric_cols = [
        "claim_amount",
        "annual_premium",
        "policy_tenure_months",
        "customer_age",
        "previous_claims",
        "claim_submission_delay",
        "repair_or_treatment_estimate",
        "claim_to_premium_ratio",
        "claim_to_estimate_ratio"
    ]

    stages = []

    # 1. StringIndexer for each categorical feature
    indexed_cols = [f"{col}_idx" for col in categorical_cols]
    string_indexer = StringIndexer(
        inputCols=categorical_cols,
        outputCols=indexed_cols,
        handleInvalid="keep"
    )
    stages.append(string_indexer)

    # 2. OneHotEncoder for indexed categorical features
    ohe_cols = [f"{col}_vec" for col in categorical_cols]
    one_hot_encoder = OneHotEncoder(
        inputCols=indexed_cols,
        outputCols=ohe_cols,
        dropLast=True
    )
    stages.append(one_hot_encoder)

    # 3. VectorAssembler combining numeric features and encoded categorical vectors
    assembler_inputs = numeric_cols + ohe_cols
    assembler = VectorAssembler(
        inputCols=assembler_inputs,
        outputCol="features",
        handleInvalid="skip"
    )
    stages.append(assembler)

    return stages, assembler_inputs


def split_data(
    df: DataFrame,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[DataFrame, DataFrame]:
    """
    Splits DataFrame into Training (80%) and Test (20%) sets.
    """
    print(f"[ML Pipeline] Splitting data into {int(train_ratio*100)}:{int((1-train_ratio)*100)} Train/Test split (seed={seed})...")
    train_df, test_df = df.randomSplit([train_ratio, 1.0 - train_ratio], seed=seed)
    
    train_df.cache()
    test_df.cache()
    
    train_count = train_df.count()
    test_count = test_df.count()
    print(f"  - Training Set Count : {train_count:,} records")
    print(f"  - Test Set Count     : {test_count:,} records")
    
    return train_df, test_df


def train_models(
    train_df: DataFrame,
    test_df: DataFrame
) -> Dict[str, Any]:
    """
    Builds and trains Logistic Regression and Random Forest models using Spark MLlib.
    
    Returns:
        dict: Fitted pipeline models and their test predictions.
    """
    base_stages, _ = build_feature_pipeline_stages()

    # Model 1: Logistic Regression Classifier
    print("\n[ML Training] Training Model 1: Logistic Regression (Baseline)...")
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=25,
        regParam=0.01,
        elasticNetParam=0.0
    )
    lr_pipeline = Pipeline(stages=base_stages + [lr])
    lr_model = lr_pipeline.fit(train_df)
    lr_predictions = lr_model.transform(test_df)
    lr_predictions.cache()
    print("  - Logistic Regression training complete.")

    # Model 2: Random Forest Classifier
    print("\n[ML Training] Training Model 2: Random Forest Classifier (Ensemble)...")
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=40,
        maxDepth=8,
        seed=42
    )
    rf_pipeline = Pipeline(stages=base_stages + [rf])
    rf_model = rf_pipeline.fit(train_df)
    rf_predictions = rf_model.transform(test_df)
    rf_predictions.cache()
    print("  - Random Forest Classifier training complete.")

    return {
        "logistic_regression": {
            "model": lr_model,
            "predictions": lr_predictions,
            "classifier": lr
        },
        "random_forest": {
            "model": rf_model,
            "predictions": rf_predictions,
            "classifier": rf
        }
    }


def save_selected_model(model: PipelineModel, output_path: Path = SELECTED_MODEL_DIR) -> str:
    """
    Persists the chosen Spark MLlib PipelineModel and its full parameter specification.
    Includes fallback for Windows native IO limitations to prevent pipeline aborts.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    native_save_path = output_path / "spark_mllib_pipeline"
    try:
        model.write().overwrite().save(str(native_save_path))
        print(f"[ML Persistence] Native Spark MLlib model successfully saved to: {native_save_path}")
    except Exception as e:
        print(f"[ML Persistence] Windows native file committer note ({type(e).__name__}): saving portable model parameters.")

    # Save comprehensive portable model specification and weights
    try:
        classifier_stage = model.stages[-1]
        model_spec = {
            "model_type": type(classifier_stage).__name__,
            "stages": [type(s).__name__ for s in model.stages],
            "num_stages": len(model.stages)
        }

        if hasattr(classifier_stage, "featureImportances"):
            model_spec["feature_importances"] = classifier_stage.featureImportances.toArray().tolist()
        if hasattr(classifier_stage, "coefficients"):
            model_spec["coefficients"] = classifier_stage.coefficients.toArray().tolist()
            model_spec["intercept"] = float(classifier_stage.intercept)

        spec_file = output_path / "model_spec.json"
        with open(spec_file, "w", encoding="utf-8") as f:
            json.dump(model_spec, f, indent=4)
        print(f"[ML Persistence] Model specification and parameters preserved: {spec_file}")
    except Exception as ex:
        print(f"[ML Persistence] Parameter extraction notice: {ex}")

    return str(output_path)
