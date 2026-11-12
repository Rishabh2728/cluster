import os
import sys
from pathlib import Path
import pytest

# Add project root to sys.path for test discovery
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure PySpark uses current python
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
if Path("c:/hadoop").exists():
    os.environ["HADOOP_HOME"] = "c:\\hadoop"

from src.config import SPARK_CONFIG
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """Single shared session-scoped SparkSession fixture for all unit tests."""
    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("INT315_UnitTests")
        .config("spark.driver.memory", "1g")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.driver.extraJavaOptions", SPARK_CONFIG["java_options"])
        .config("spark.executor.extraJavaOptions", SPARK_CONFIG["java_options"])
        .config("spark.sql.ansi.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
