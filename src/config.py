"""
Configuration Module for INT315 Cluster Computing Project
Predicting Insurance Claim Approval Using Apache Spark
Student: Rishabh Kumar | Roll No: 49 | Reg No: 12310491 | Section: K2P23PD
"""

import os
from pathlib import Path

# Base Project Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
LOGS_DIR = OUTPUTS_DIR / "logs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

MODELS_DIR = PROJECT_ROOT / "models"
SELECTED_MODEL_DIR = MODELS_DIR / "selected_model"

SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
SCREENSHOTS_EXEC_DIR = SCREENSHOTS_DIR / "execution"
SCREENSHOTS_RESULTS_DIR = SCREENSHOTS_DIR / "results"

REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"

# Student & Academic Metadata
STUDENT_INFO = {
    "name": "Rishabh Kumar",
    "roll_no": "49",
    "reg_no": "12310491",
    "section": "K2P23PD",
    "course_code": "INT315",
    "course_name": "Cluster Computing",
    "project_framework": "Apache Spark & Spark MLlib",
    "programming_language": "Python (PySpark)",
    "project_title": "Predicting Insurance Claim Approval Using Apache Spark",
    "project_subtitle": "A Distributed Machine Learning Approach for Insurance Claim Classification Using PySpark",
    "faculty_name": "Priyanka",
    "faculty_uid": "33488",
    "school_name": "School of Computer Science and Engineering",
    "university_location": "Lovely Professional University, Punjab"
}

# Synthetic Dataset Configuration
DATASET_CONFIG = {
    "target_records": 120000,
    "random_seed": 42,
    "raw_csv_path": DATA_RAW_DIR / "insurance_claims.csv",
    "readme_path": DATA_DIR / "DATASET_README.md",
    "missing_rate": 0.02,           # ~2% missing in optional fields
    "approval_target_ratio": 0.70,  # ~70% approved, 30% rejected
    "test_split_ratio": 0.20        # 80:20 train:test split
}

# Spark Configuration
SPARK_CONFIG = {
    "app_name": "INT315_Insurance_Claim_Approval_Prediction",
    "master": "local[*]",
    "driver_memory": "4g",
    "executor_memory": "4g",
    "sql_shuffle_partitions": "8",
    "default_parallelism": "4",
    # Modern Java options for compatibility with Java 17/21/23
    "java_options": (
        "--add-opens=java.base/java.lang=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
        "--add-opens=java.base/java.io=ALL-UNNAMED "
        "--add-opens=java.base/java.net=ALL-UNNAMED "
        "--add-opens=java.base/java.nio=ALL-UNNAMED "
        "--add-opens=java.base/java.util=ALL-UNNAMED "
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
        "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED "
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED"
    )
}

def ensure_directories():
    """Ensure all required directories exist across the project."""
    dirs = [
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        CHARTS_DIR,
        LOGS_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        MODELS_DIR,
        SELECTED_MODEL_DIR,
        SCREENSHOTS_EXEC_DIR,
        SCREENSHOTS_RESULTS_DIR,
        REPORTS_DIR,
        DOCS_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
