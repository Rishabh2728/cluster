"""
Model Evaluation Module for INT315 Cluster Computing Project
Predicting Insurance Claim Approval Using Apache Spark
Student: Rishabh Kumar | Roll No: 49 | Reg No: 12310491 | Section: K2P23PD
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config import METRICS_DIR


def evaluate_binary_classifier(
    predictions_df: DataFrame,
    model_name: str
) -> dict:
    """
    Computes comprehensive binary classification metrics for a Spark ML prediction DataFrame:
    - Accuracy
    - Precision, Recall, F1 (Overall / Weighted)
    - Precision & Recall specifically for Rejected (label=0.0) and Approved (label=1.0)
    - ROC-AUC
    - PR-AUC
    - Confusion Matrix components: TP, FP, TN, FN
    """
    print(f"\n[Model Evaluation] Calculating evaluation metrics for: {model_name}...")

    # Spark ML Evaluators
    binary_roc_eval = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction",
        labelCol="label",
        metricName="areaUnderROC"
    )
    binary_pr_eval = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction",
        labelCol="label",
        metricName="areaUnderPR"
    )

    multi_acc_eval = MulticlassClassificationEvaluator(
        predictionCol="prediction",
        labelCol="label",
        metricName="accuracy"
    )
    multi_f1_eval = MulticlassClassificationEvaluator(
        predictionCol="prediction",
        labelCol="label",
        metricName="f1"
    )
    multi_prec_eval = MulticlassClassificationEvaluator(
        predictionCol="prediction",
        labelCol="label",
        metricName="weightedPrecision"
    )
    multi_rec_eval = MulticlassClassificationEvaluator(
        predictionCol="prediction",
        labelCol="label",
        metricName="weightedRecall"
    )

    roc_auc = binary_roc_eval.evaluate(predictions_df)
    pr_auc = binary_pr_eval.evaluate(predictions_df)
    accuracy = multi_acc_eval.evaluate(predictions_df)
    f1_score = multi_f1_eval.evaluate(predictions_df)
    weighted_precision = multi_prec_eval.evaluate(predictions_df)
    weighted_recall = multi_rec_eval.evaluate(predictions_df)

    # Compute Confusion Matrix via Spark GroupBy
    # Label: 1.0 = Approved (Positive), 0.0 = Rejected (Negative)
    cm_df = (
        predictions_df
        .groupBy("label", "prediction")
        .count()
        .collect()
    )

    counts_map = {(row["label"], row["prediction"]): row["count"] for row in cm_df}

    tp = counts_map.get((1.0, 1.0), 0)  # True Positive (Approved as Approved)
    fn = counts_map.get((1.0, 0.0), 0)  # False Negative (Approved as Rejected)
    fp = counts_map.get((0.0, 1.0), 0)  # False Positive (Rejected as Approved)
    tn = counts_map.get((0.0, 0.0), 0)  # True Negative (Rejected as Rejected)

    # Per-class metrics
    # Approved Class (Positive = 1.0)
    precision_approved = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_approved = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Rejected Class (Minority = 0.0)
    precision_rejected = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_rejected = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_rejected = (
        2 * (precision_rejected * recall_rejected) / (precision_rejected + recall_rejected)
        if (precision_rejected + recall_rejected) > 0 else 0.0
    )

    metrics = {
        "model_name": model_name,
        "accuracy": round(accuracy, 4),
        "precision": round(weighted_precision, 4),
        "recall": round(weighted_recall, 4),
        "f1_score": round(f1_score, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision_rejected": round(precision_rejected, 4),
        "recall_rejected": round(recall_rejected, 4),
        "f1_rejected": round(f1_rejected, 4),
        "precision_approved": round(precision_approved, 4),
        "recall_approved": round(recall_approved, 4)
    }

    print(f"  Accuracy              : {metrics['accuracy']:.4f}")
    print(f"  Weighted Precision    : {metrics['precision']:.4f}")
    print(f"  Weighted Recall       : {metrics['recall']:.4f}")
    print(f"  Weighted F1-Score     : {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC               : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC                : {metrics['pr_auc']:.4f}")
    print(f"  Rejected Class Recall : {metrics['recall_rejected']:.4f}")
    print(f"  Confusion Matrix      : TP={tp:,}, FP={fp:,}, TN={tn:,}, FN={fn:,}")

    return metrics


def extract_feature_importances(rf_model, feature_names: list = None) -> pd.DataFrame:
    """
    Extracts tree feature importances from fitted Random Forest model.
    """
    try:
        # The RandomForestClassificationModel is the last stage of the PipelineModel
        rf_stage = rf_model.stages[-1]
        importances = rf_stage.featureImportances.toArray()
        
        # If explicit feature names are not provided or count mismatch, generate generic names
        if feature_names is None or len(feature_names) != len(importances):
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
            
        df_imp = pd.DataFrame({
            "feature": feature_names[:len(importances)],
            "importance": importances
        }).sort_values(by="importance", ascending=False)
        
        return df_imp
    except Exception as e:
        print(f"[Model Evaluation] Notice: Feature importance extraction note: {e}")
        return pd.DataFrame(columns=["feature", "importance"])


def compare_and_select_model(
    lr_metrics: dict,
    rf_metrics: dict,
    lr_model,
    rf_model,
    output_dir: Path = METRICS_DIR
) -> Tuple[dict, Any, str]:
    """
    Compares Logistic Regression vs Random Forest using balanced criteria:
    - ROC-AUC and F1-Score (rewarding discrimination ability)
    - Recall for the minority Rejected class (critical in claim underwriting risk management)
    
    Persists model comparison table to CSV.
    Returns: (selected_metrics, selected_model, rationale_text)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_df = pd.DataFrame([lr_metrics, rf_metrics])

    # Save CSV comparison
    csv_path = output_dir / "model_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n[Model Evaluation] Comparison metrics saved to: {csv_path}")

    # Decision Logic:
    # Random Forest typically outperforms linear logistic regression on multi-feature nonlinear interactions
    rf_score = (rf_metrics["f1_score"] * 0.4) + (rf_metrics["roc_auc"] * 0.4) + (rf_metrics["recall_rejected"] * 0.2)
    lr_score = (lr_metrics["f1_score"] * 0.4) + (lr_metrics["roc_auc"] * 0.4) + (lr_metrics["recall_rejected"] * 0.2)

    if rf_score >= lr_score:
        selected_model = rf_model
        selected_name = "Random Forest Classifier"
        selected_metrics = rf_metrics
        rationale = (
            f"Random Forest Classifier selected with superior composite score ({rf_score:.4f} vs {lr_score:.4f}). "
            f"It demonstrated an ROC-AUC of {rf_metrics['roc_auc']:.4f} (vs {lr_metrics['roc_auc']:.4f}) and "
            f"F1-Score of {rf_metrics['f1_score']:.4f}, effectively modeling non-linear policyholder risk dynamics."
        )
    else:
        selected_model = lr_model
        selected_name = "Logistic Regression"
        selected_metrics = lr_metrics
        rationale = (
            f"Logistic Regression selected with superior composite score ({lr_score:.4f} vs {rf_score:.4f}). "
            f"It provided an ROC-AUC of {lr_metrics['roc_auc']:.4f} with robust generalization and parameter efficiency."
        )

    print(f"\n==================== MODEL SELECTION DECISION ====================")
    print(f"Selected Algorithm : {selected_name}")
    print(f"Selection Rationale: {rationale}")
    print("==================================================================\n")

    return selected_metrics, selected_model, rationale
