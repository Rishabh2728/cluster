"""
Spark Session Management for INT315 Cluster Computing Project
Predicting Insurance Claim Approval Using Apache Spark
Student: Rishabh Kumar | Roll No: 49 | Reg No: 12310491
"""

import os
import sys
from pathlib import Path
from pyspark.sql import SparkSession
from src.config import SPARK_CONFIG, STUDENT_INFO

# Ensure PySpark uses the exact Python interpreter and Hadoop home
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
if Path("c:/hadoop").exists():
    os.environ["HADOOP_HOME"] = "c:\\hadoop"


def create_spark_session(app_name: str = None, master: str = None) -> SparkSession:
    """
    Initializes and returns a configured Apache Spark Session.
    Configured for local multi-core distributed processing (local[*]).
    Includes modern JVM compatibility options for Java 17/21/23.
    """
    selected_app_name = app_name or SPARK_CONFIG["app_name"]
    selected_master = master or SPARK_CONFIG["master"]

    java_opts = SPARK_CONFIG["java_options"]

    builder = (
        SparkSession.builder
        .appName(selected_app_name)
        .master(selected_master)
        .config("spark.driver.memory", SPARK_CONFIG["driver_memory"])
        .config("spark.executor.memory", SPARK_CONFIG["executor_memory"])
        .config("spark.sql.shuffle.partitions", SPARK_CONFIG["sql_shuffle_partitions"])
        .config("spark.default.parallelism", SPARK_CONFIG["default_parallelism"])
        .config("spark.driver.extraJavaOptions", java_opts)
        .config("spark.executor.extraJavaOptions", java_opts)
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    return spark


def get_spark_environment_info(spark: SparkSession) -> dict:
    """Inspect and return metadata regarding the active Spark execution runtime."""
    sc = spark.sparkContext
    return {
        "spark_version": spark.version,
        "master": sc.master,
        "app_name": sc.appName,
        "default_parallelism": sc.defaultParallelism,
        "default_min_partitions": sc.defaultMinPartitions,
        "python_version": sys.version.split()[0],
        "student_name": STUDENT_INFO["name"],
        "registration_number": STUDENT_INFO["reg_no"]
    }


def stop_spark_session(spark: SparkSession) -> None:
    """Gracefully terminates the active SparkSession."""
    if spark is not None:
        try:
            spark.stop()
            print("[Spark Session] SparkSession successfully stopped.")
        except Exception as e:
            print(f"[Spark Session] Warning during shutdown: {e}")
