"""
Spark Session Manager for Flight Delay Prediction Big Data Project.

Handles:
- PySpark environment configuration (Windows executable path, driver configs)
- SparkSession creation with local parallelism
- Spark DataFrame loading and Spark SQL queries
- Graceful status reporting
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from config import DATASET_CSV

# Ensure Python executable path is configured for PySpark workers on Windows
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

_spark_instance = None

def get_spark_session(app_name: str = "FlightDelayBigDataApp"):
    """
    Returns an active PySpark SparkSession, creating one if not already initialized.
    """
    global _spark_instance
    if _spark_instance is not None:
        try:
            # Check if active
            _spark_instance.sparkContext._jsc.sc().isStopped()
            return _spark_instance
        except Exception:
            _spark_instance = None

    try:
        from pyspark.sql import SparkSession

        # Configure Spark Session for local big data processing
        spark = SparkSession.builder \
            .appName(app_name) \
            .master("local[*]") \
            .config("spark.driver.memory", "2g") \
            .config("spark.sql.shuffle.partitions", "4") \
            .config("spark.ui.enabled", "false") \
            .getOrCreate()

        spark.sparkContext.setLogLevel("ERROR")
        _spark_instance = spark
        return _spark_instance
    except Exception as e:
        print(f"[Warning] Could not initialize PySpark session: {e}")
        return None

def check_spark_status() -> Tuple[bool, str]:
    """
    Verifies if PySpark and Java runtime are functioning properly.
    """
    spark = get_spark_session()
    if spark:
        try:
            ver = spark.version
            return True, f"Apache Spark v{ver} (Local Cluster Active)"
        except Exception as e:
            return False, f"Spark session error: {e}"
    return False, "PySpark session not available (Fallback analytical engine active)"

_flights_table_registered = False

def load_flight_data_spark():
    """
    Loads the flight delay CSV into an Apache Spark DataFrame.
    """
    global _flights_table_registered
    spark = get_spark_session()
    if spark is None:
        return None
    try:
        spark_df = spark.read.csv(
            str(DATASET_CSV),
            header=True,
            inferSchema=True
        )
        spark_df.createOrReplaceTempView("flights")
        _flights_table_registered = True
        return spark_df
    except Exception as e:
        print(f"Error loading CSV in Spark: {e}")
        return None

def run_spark_sql_query(query: str) -> Optional[pd.DataFrame]:
    """
    Executes a Spark SQL query and returns the aggregated result as a Pandas DataFrame.
    """
    global _flights_table_registered
    spark = get_spark_session()
    if spark:
        try:
            if not _flights_table_registered:
                load_flight_data_spark()
            result_df = spark.sql(query).toPandas()
            return result_df
        except Exception as e:
            print(f"Spark SQL execution error: {e}")
    # Fallback to in-memory SQL execution via sqlite3
    try:
        import sqlite3
        pdf = pd.read_csv(DATASET_CSV)
        conn = sqlite3.connect(":memory:")
        pdf.to_sql("flights", conn, index=False)
        res = pd.read_sql_query(query, conn)
        conn.close()
        return res
    except Exception as ex:
        print(f"Fallback SQL error: {ex}")
        return None
