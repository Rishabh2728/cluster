"""
Main Execution Script for INT315 Cluster Computing Project
Predicting Insurance Claim Approval Using Apache Spark
Student: Rishabh Kumar | Roll No: 49 | Reg No: 12310491 | Section: K2P23PD

Master pipeline orchestrating:
1. Synthetic Insurance Claims Dataset Generation
2. SparkSession Initialization in local[*] Multi-Core Mode
3. Distributed Data Loading and Schema Inspection
4. Distributed Cleaning and Anomaly Handling
5. Feature Engineering (Domain Ratios, Categorical Bins)
6. Distributed Exploratory Data Analysis (EDA via Spark SQL)
7. Train/Test Partitioning (80:20 Split)
8. Spark MLlib Classification Training (Logistic Regression & Random Forest)
9. Rigorous Model Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix)
10. Model Selection, Comparison, and Disk Persistence
11. Spark Partitioning Demonstration & Performance Benchmark
12. Visualization Generation (11 Publication-Quality Charts)
13. Prediction Engine Verification
14. Project Summary Compilation
15. Graceful Spark Session Shutdown
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CHARTS_DIR,
    DATA_PROCESSED_DIR,
    DATASET_CONFIG,
    LOGS_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    SELECTED_MODEL_DIR,
    STUDENT_INFO,
    ensure_directories,
)
from src.dataset_generator import generate_synthetic_claims
from src.spark_session import (
    create_spark_session,
    get_spark_environment_info,
    stop_spark_session,
)
from src.data_loader import inspect_claims_data, load_claims_data
from src.data_cleaning import clean_claims_data
from src.feature_engineering import apply_feature_engineering
from src.exploratory_analysis import run_exploratory_analysis
from src.model_training import split_data, train_models, save_selected_model
from src.model_evaluation import (
    evaluate_binary_classifier,
    compare_and_select_model,
    extract_feature_importances,
)
from src.prediction import predict_single_claim
from src.performance import (
    demonstrate_partitioning_mechanics,
    run_partition_performance_benchmark,
)
from src.visualization import generate_all_insurance_visualizations


def setup_pipeline_logger():
    """Initializes unified file and console logging."""
    ensure_directories()
    log_file = LOGS_DIR / "pipeline_execution.log"

    logger = logging.getLogger("INT315_Claims_Pipeline")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger, log_file


def save_spark_df_to_csv(df, output_path: Path):
    """Exports Spark DataFrame to CSV using pandas."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = df.toPandas()
    pdf.to_csv(output_path, index=False, encoding="utf-8")
    return pdf


def main():
    total_start_time = time.perf_counter()
    logger, log_file = setup_pipeline_logger()

    logger.info("==========================================================================")
    logger.info("  INT315 - CLUSTER COMPUTING PROJECT PIPELINE EXECUTION")
    logger.info("  PREDICTING INSURANCE CLAIM APPROVAL USING APACHE SPARK")
    logger.info(f"  Student: {STUDENT_INFO['name']} | Roll: {STUDENT_INFO['roll_no']} | Reg: {STUDENT_INFO['reg_no']}")
    logger.info(f"  Section: {STUDENT_INFO['section']} | Course: {STUDENT_INFO['course_code']} ({STUDENT_INFO['course_name']})")
    logger.info("==========================================================================")

    spark = None
    try:
        # Step 1: Synthetic Dataset Verification / Generation
        raw_csv = DATASET_CONFIG["raw_csv_path"]
        if not raw_csv.exists() or os.path.getsize(raw_csv) == 0:
            logger.info("[Step 1/14] Generating synthetic insurance claims dataset...")
            gen_stats = generate_synthetic_claims(
                target_records=DATASET_CONFIG["target_records"],
                seed=DATASET_CONFIG["random_seed"],
                output_path=raw_csv
            )
            logger.info(f"  Generated {gen_stats['total_records']:,} rows ({gen_stats['file_size_mb']} MB).")
        else:
            file_mb = os.path.getsize(raw_csv) / (1024 * 1024)
            logger.info(f"[Step 1/14] Using existing synthetic dataset ({file_mb:.2f} MB): {raw_csv}")

        # Step 2: SparkSession Initialization
        logger.info("[Step 2/14] Initializing Apache Spark Session in local[*] mode...")
        spark = create_spark_session(app_name="INT315_Insurance_Claim_Approval_Prediction")
        spark_env = get_spark_environment_info(spark)
        logger.info(f"  Spark Version       : {spark_env['spark_version']}")
        logger.info(f"  Execution Master    : {spark_env['master']}")
        logger.info(f"  Default Parallelism : {spark_env['default_parallelism']}")
        logger.info(f"  Python Version      : {spark_env['python_version']}")

        # Step 3: Dataset Ingestion & Schema Inspection
        logger.info("[Step 3/14] Ingesting CSV records into distributed Spark DataFrame...")
        df_raw = load_claims_data(spark, str(raw_csv))
        inspect_info = inspect_claims_data(df_raw, num_samples=5)
        logger.info(f"  Raw Records Ingested : {inspect_info['raw_record_count']:,}")
        logger.info(f"  Initial Partitions   : {inspect_info['initial_partitions']}")

        # Step 4: Distributed Data Cleaning
        logger.info("[Step 4/14] Executing distributed cleaning & domain validation...")
        df_clean, cleaning_metrics = clean_claims_data(df_raw)
        logger.info(f"  Clean Claim Records  : {cleaning_metrics['cleaned_records']:,}")
        logger.info(f"  Duplicates Dropped   : {cleaning_metrics['duplicate_records_dropped']:,}")
        logger.info(f"  Invalid Rows Dropped : {cleaning_metrics['invalid_records_dropped']:,}")
        logger.info(f"  Data Retention Rate  : {cleaning_metrics['retention_rate_pct']}%")

        # Step 5: Feature Engineering
        logger.info("[Step 5/14] Applying feature engineering & risk ratios...")
        df_featured = apply_feature_engineering(df_clean)
        df_featured.cache()
        logger.info("  Engineered features cached across partitions.")

        # Step 6: Exploratory Data Analysis (EDA)
        logger.info("[Step 6/14] Executing Spark SQL exploratory analyses...")
        eda_results = run_exploratory_analysis(spark, df_featured)
        kpis = eda_results["kpis"]

        # Export EDA CSVs
        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        pandas_eda = {"kpis": kpis}
        for key, s_df in eda_results.items():
            if key == "kpis":
                continue
            csv_path = DATA_PROCESSED_DIR / f"{key}.csv"
            p_df = save_spark_df_to_csv(s_df, csv_path)
            pandas_eda[key] = p_df
            logger.info(f"  Exported EDA Table: {csv_path.name} ({len(p_df)} rows)")

        # Save clean sample for inspection
        clean_sample_csv = DATA_PROCESSED_DIR / "cleaned_claims_sample.csv"
        df_featured.limit(500).toPandas().to_csv(clean_sample_csv, index=False)
        logger.info(f"  Exported Clean Sample: {clean_sample_csv.name}")

        # Step 7: Train/Test Split (80:20)
        logger.info("[Step 7/14] Splitting data into reproducible 80:20 Train/Test partitions...")
        train_df, test_df = split_data(df_featured, train_ratio=0.8, seed=42)

        # Step 8: MLlib Model Training (Logistic Regression & Random Forest)
        logger.info("[Step 8/14] Training classification pipelines via Spark MLlib...")
        models_dict = train_models(train_df, test_df)

        # Step 9: Model Evaluation
        logger.info("[Step 9/14] Evaluating models on holdout test partition...")
        lr_metrics = evaluate_binary_classifier(
            models_dict["logistic_regression"]["predictions"],
            model_name="Logistic Regression"
        )
        rf_metrics = evaluate_binary_classifier(
            models_dict["random_forest"]["predictions"],
            model_name="Random Forest Classifier"
        )

        # Extract Random Forest feature importances
        feature_names = [
            "claim_amount", "annual_premium", "policy_tenure_months", "customer_age",
            "previous_claims", "claim_submission_delay", "repair_or_treatment_estimate",
            "claim_to_premium_ratio", "claim_to_estimate_ratio",
            "policy_type", "incident_type", "incident_severity",
            "documentation_status", "policy_active", "premium_payment_status",
            "incident_reported", "policy_tenure_cat", "submission_delay_cat", "previous_claim_cat"
        ]
        df_imp = extract_feature_importances(models_dict["random_forest"]["model"], feature_names)

        # Step 10: Model Comparison & Selection
        logger.info("[Step 10/14] Comparing models and persisting best pipeline...")
        selected_metrics, selected_model, rationale = compare_and_select_model(
            lr_metrics,
            rf_metrics,
            models_dict["logistic_regression"]["model"],
            models_dict["random_forest"]["model"],
            output_dir=METRICS_DIR
        )
        saved_model_path = save_selected_model(selected_model, SELECTED_MODEL_DIR)
        logger.info(f"  Selected Model Path : {saved_model_path}")

        # Step 11: Partitioning Mechanics Demonstration
        logger.info("[Step 11/14] Executing Spark partitioning mechanics demonstration...")
        part_demo = demonstrate_partitioning_mechanics(df_featured)

        # Step 12: Performance Benchmarking Across Partitions
        logger.info("[Step 12/14] Running performance benchmark across 2, 4, and 8 partitions...")
        benchmark_df = run_partition_performance_benchmark(df_featured, partition_configs=[2, 4, 8], iterations=3)

        # Step 13: High-Resolution Chart Generation
        logger.info("[Step 13/14] Generating publication-grade charts from Spark results...")
        generated_charts = generate_all_insurance_visualizations(
            eda_dfs=pandas_eda,
            lr_metrics=lr_metrics,
            rf_metrics=rf_metrics,
            feature_importances=df_imp,
            benchmark_df=benchmark_df,
            output_dir=CHARTS_DIR
        )

        # Step 14: Sample Prediction & Inference Verification
        logger.info("[Step 14/14] Executing inference engine test on sample incoming claim...")
        sample_claim = {
            "policy_type": "Motor",
            "claim_amount": 7500.0,
            "annual_premium": 1100.0,
            "policy_tenure_months": 36,
            "customer_age": 38,
            "incident_type": "Vehicle Collision",
            "incident_severity": "Medium",
            "previous_claims": 1,
            "documentation_status": "Complete",
            "policy_active": "Yes",
            "premium_payment_status": "Paid",
            "claim_submission_delay": 4,
            "incident_reported": "Yes",
            "repair_or_treatment_estimate": 7200.0
        }
        pred_res = predict_single_claim(spark, selected_model, sample_claim)
        logger.info(f"  Sample Claim Outcome : {pred_res['predicted_outcome']}")
        logger.info(f"  Approval Probability : {pred_res['approval_probability']}%")
        logger.info(f"  Rejection Probability: {pred_res['rejection_probability']}%")

        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        pred_json_path = PREDICTIONS_DIR / "sample_prediction.json"
        with open(pred_json_path, "w", encoding="utf-8") as f:
            json.dump(pred_res, f, indent=4)

        # Consolidated Project Summary JSON
        total_time = time.perf_counter() - total_start_time
        summary_payload = {
            "project_title": STUDENT_INFO["project_title"],
            "student_info": STUDENT_INFO,
            "execution_timestamp": datetime.now().isoformat(),
            "spark_version": spark_env["spark_version"],
            "master": spark_env["master"],
            "raw_records": inspect_info["raw_record_count"],
            "clean_records": cleaning_metrics["cleaned_records"],
            "cleaning_metrics": cleaning_metrics,
            "kpis": kpis,
            "logistic_regression_metrics": lr_metrics,
            "random_forest_metrics": rf_metrics,
            "selected_model": {
                "name": selected_metrics["model_name"],
                "metrics": selected_metrics,
                "rationale": rationale,
                "model_path": str(saved_model_path)
            },
            "partition_demo": part_demo,
            "benchmark_results": benchmark_df.to_dict(orient="records"),
            "charts_generated": list(generated_charts.keys()),
            "sample_prediction": pred_res,
            "total_execution_seconds": round(total_time, 2)
        }

        summary_file = DATA_PROCESSED_DIR / "project_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=4)
        logger.info(f"  Summary JSON saved to : {summary_file}")

        logger.info("==========================================================================")
        logger.info(f"  PIPELINE EXECUTION COMPLETED IN {total_time:.2f} SECONDS")
        logger.info(f"  Log File Available At: {log_file}")
        logger.info("==========================================================================")

    except Exception as e:
        logger.exception(f"FATAL ERROR in pipeline execution: {e}")
        raise
    finally:
        if spark is not None:
            logger.info("Shutting down Apache Spark Session gracefully...")
            stop_spark_session(spark)


if __name__ == "__main__":
    main()
