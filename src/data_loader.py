"""
Data Loading Module for INT315 Cluster Computing Project
Predicting Insurance Claim Approval Using Apache Spark
Student: Rishabh Kumar | Roll No: 49 | Reg No: 12310491 | Section: K2P23PD
"""

from pathlib import Path
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from src.config import DATASET_CONFIG


def get_insurance_claims_schema() -> StructType:
    """
    Defines the explicit schema for insurance claim records.
    Explicit schema prevents costly schema-inference passes across large datasets.
    """
    return StructType([
        StructField("claim_id", StringType(), nullable=True),
        StructField("policy_id", StringType(), nullable=True),
        StructField("policy_type", StringType(), nullable=True),
        StructField("claim_amount", DoubleType(), nullable=True),
        StructField("annual_premium", DoubleType(), nullable=True),
        StructField("policy_tenure_months", IntegerType(), nullable=True),
        StructField("customer_age", IntegerType(), nullable=True),
        StructField("incident_type", StringType(), nullable=True),
        StructField("incident_severity", StringType(), nullable=True),
        StructField("previous_claims", IntegerType(), nullable=True),
        StructField("documentation_status", StringType(), nullable=True),
        StructField("policy_active", StringType(), nullable=True),
        StructField("premium_payment_status", StringType(), nullable=True),
        StructField("claim_submission_delay", IntegerType(), nullable=True),
        StructField("incident_reported", StringType(), nullable=True),
        StructField("repair_or_treatment_estimate", DoubleType(), nullable=True),
        StructField("claim_approval", StringType(), nullable=True),
    ])


def load_claims_data(spark: SparkSession, csv_path: str = None) -> DataFrame:
    """
    Loads raw CSV insurance claim records into a distributed PySpark DataFrame.
    """
    target_path = str(csv_path or DATASET_CONFIG["raw_csv_path"])
    schema = get_insurance_claims_schema()

    print(f"[Data Loader] Loading raw claims data from: {target_path}")

    df = (
        spark.read
        .format("csv")
        .option("header", "true")
        .option("mode", "DROPMALFORMED")
        .schema(schema)
        .load(target_path)
    )

    return df


def inspect_claims_data(df: DataFrame, num_samples: int = 5) -> dict:
    """
    Inspects schema, sample rows, partition count, total count,
    and quantifies missing values per column.
    """
    record_count = df.count()
    num_partitions = df.rdd.getNumPartitions()

    print("\n==================== DATA INGESTION & SCHEMA INSPECTION ====================")
    print(f"Total Raw Claim Records Loaded : {record_count:,}")
    print(f"Initial Spark Partitions       : {num_partitions}")
    print("Schema Definition:")
    df.printSchema()

    print(f"\nDisplaying Sample {num_samples} Claim Records:")
    df.show(num_samples, truncate=False)

    # Inspect missing values across all columns
    print("\nMissing Value Quantification:")
    missing_exprs = []
    for c, dtype in df.dtypes:
        if dtype in ["string", "string"]:
            cond = F.col(c).isNull() | (F.col(c) == "") | (F.col(c) == "null")
        else:
            cond = F.col(c).isNull() | F.isnan(F.col(c))
        missing_exprs.append(F.count(F.when(cond, F.lit(1))).alias(c))
    missing_counts_row = df.select(missing_exprs).collect()[0].asDict()

    has_missing = False
    for col_name, missing_cnt in missing_counts_row.items():
        if missing_cnt > 0:
            pct = (missing_cnt / record_count) * 100
            print(f"  - {col_name:28s}: {missing_cnt:,} ({pct:.2f}%)")
            has_missing = True
    if not has_missing:
        print("  - No missing values detected across columns.")

    print("============================================================================\n")

    return {
        "raw_record_count": record_count,
        "initial_partitions": num_partitions,
        "columns": df.columns,
        "missing_counts": missing_counts_row
    }
