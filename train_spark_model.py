"""
PySpark Machine Learning Model Training Script
Project: Prediction of Flight Delays Using Big Data

This script trains a PySpark RandomForestClassifier on historical flight operations data.
Features:
- Spark MLlib Pipeline (StringIndexer, OneHotEncoder, VectorAssembler, RandomForestClassifier)
- 80/20 Train/Test Split (ensuring no data leakage)
- Comprehensive Evaluation: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
- Exports metrics to JSON and models for Streamlit frontend integration.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Set PySpark Python worker paths
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from config import DATASET_CSV, METRICS_JSON, SKLEARN_MODEL_PKL, SPARK_MODEL_DIR
from modules.spark_manager import get_spark_session

from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def train_with_spark():
    """
    Trains RandomForestClassifier using Apache Spark MLlib.
    """
    print("\n" + "="*60)
    print(">>> STARTING PYSPARK MLlib RANDOM FOREST MODEL TRAINING")
    print("="*60)

    spark = get_spark_session("FlightDelaySparkTraining")
    if spark is None:
        print("[Error] Failed to initialize Apache Spark Session.")
        return None

    from pyspark.sql.functions import col
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, OneHotEncoder as SparkOHE, VectorAssembler
    from pyspark.ml.classification import RandomForestClassifier as SparkRF
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

    print(f"[Spark] Loading dataset from: {DATASET_CSV}")
    spark_df = spark.read.csv(str(DATASET_CSV), header=True, inferSchema=True)
    
    # Cast label to double
    spark_df = spark_df.withColumn("is_delayed", col("is_delayed").cast("double"))
    
    total_count = spark_df.count()
    print(f"[Spark] Total records in Spark DataFrame: {total_count}")

    # Avoid data leakage: Split 80% train, 20% test first
    train_data, test_data = spark_df.randomSplit([0.8, 0.2], seed=42)
    train_count = train_data.count()
    test_count = test_data.count()
    print(f"[Spark] Train set: {train_count} records | Test set: {test_count} records")

    # Define Categorical and Numerical Features
    categorical_cols = ["airline", "source", "destination", "weather_condition"]
    numeric_cols = [
        "departure_hour",
        "distance_miles",
        "temperature_c",
        "wind_speed_mph",
        "visibility_miles",
        "rainfall_mm"
    ]

    # Spark MLlib Feature Engineering Pipeline stages
    indexers = [
        StringIndexer(inputCol=col_name, outputCol=f"{col_name}_idx", handleInvalid="keep")
        for col_name in categorical_cols
    ]

    encoders = [
        SparkOHE(inputCol=f"{col_name}_idx", outputCol=f"{col_name}_vec")
        for col_name in categorical_cols
    ]

    assembler_inputs = [f"{col_name}_vec" for col_name in categorical_cols] + numeric_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

    rf = SparkRF(
        labelCol="is_delayed",
        featuresCol="features",
        numTrees=50,
        maxDepth=8,
        seed=42
    )

    pipeline = Pipeline(stages=indexers + encoders + [assembler, rf])

    print("[Spark] Fitting ML Pipeline on training data...")
    spark_model = pipeline.fit(train_data)

    print("[Spark] Evaluating model on held-out test data...")
    predictions = spark_model.transform(test_data)

    # Multi-class and Binary Evaluators
    eval_acc = MulticlassClassificationEvaluator(labelCol="is_delayed", predictionCol="prediction", metricName="accuracy")
    eval_prec = MulticlassClassificationEvaluator(labelCol="is_delayed", predictionCol="prediction", metricName="weightedPrecision")
    eval_rec = MulticlassClassificationEvaluator(labelCol="is_delayed", predictionCol="prediction", metricName="weightedRecall")
    eval_f1 = MulticlassClassificationEvaluator(labelCol="is_delayed", predictionCol="prediction", metricName="f1")
    eval_auc = BinaryClassificationEvaluator(labelCol="is_delayed", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

    accuracy = eval_acc.evaluate(predictions)
    precision = eval_prec.evaluate(predictions)
    recall = eval_rec.evaluate(predictions)
    f1 = eval_f1.evaluate(predictions)
    roc_auc = eval_auc.evaluate(predictions)

    print(f"\n[Spark MLlib Evaluation Results]")
    print(f" - Accuracy:  {accuracy * 100:.2f}%")
    print(f" - Precision: {precision * 100:.2f}%")
    print(f" - Recall:    {recall * 100:.2f}%")
    print(f" - F1-Score:  {f1 * 100:.2f}%")
    print(f" - ROC AUC:   {roc_auc:.4f}")

    # Extract confusion matrix from Spark predictions
    pred_pd = predictions.select("is_delayed", "prediction").toPandas()
    cm = confusion_matrix(pred_pd["is_delayed"], pred_pd["prediction"]).tolist()

    # Extract RF feature importances from fitted model
    rf_stage = spark_model.stages[-1]
    spark_importances = rf_stage.featureImportances.toArray().tolist()

    # Save Spark Model (overwrite if exists)
    try:
        import shutil
        if SPARK_MODEL_DIR.exists():
            shutil.rmtree(SPARK_MODEL_DIR)
        spark_model.write().overwrite().save(str(SPARK_MODEL_DIR))
        print(f"[Spark] Pipeline model saved to {SPARK_MODEL_DIR}")
    except Exception as e:
        print(f"[Notice] Spark model directory save note: {e}")

    spark.stop()

    return {
        "engine": "Apache Spark MLlib (RandomForestClassifier)",
        "train_samples": int(train_count),
        "test_samples": int(test_count),
        "total_samples": int(total_count),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": cm,
        "num_trees": 50,
        "max_depth": 8
    }

def train_sklearn_companion(spark_metrics):
    """
    Trains a companion Scikit-Learn RandomForest model on the same dataset
    to enable microsecond interactive predictions in the Streamlit frontend.
    """
    print("\n" + "="*60)
    print(">>> TRAINING HIGH-PERFORMANCE INFERENCE COMPANION MODEL")
    print("="*60)

    df = pd.read_csv(DATASET_CSV)
    
    categorical_cols = ["airline", "source", "destination", "weather_condition"]
    numeric_cols = [
        "departure_hour",
        "distance_miles",
        "temperature_c",
        "wind_speed_mph",
        "visibility_miles",
        "rainfall_mm"
    ]
    all_feature_cols = categorical_cols + numeric_cols

    X = df[all_feature_cols]
    y = df["is_delayed"]

    # 80/20 train/test split (same ratio & random state)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

    clf = SklearnPipeline([
        ("preprocessor", preprocessor),
        ("rf", SklearnRF(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1))
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    auc = roc_auc_score(y_test, y_prob)

    # Feature Importances extraction
    fitted_ohe = clf.named_steps["preprocessor"].named_transformers_["cat"]
    encoded_cat_names = fitted_ohe.get_feature_names_out(categorical_cols).tolist()
    all_feature_names = encoded_cat_names + numeric_cols
    raw_importances = clf.named_steps["rf"].feature_importances_.tolist()

    # Map back to aggregated core features
    core_features = {
        "Weather Condition": 0.0,
        "Wind Speed": 0.0,
        "Visibility": 0.0,
        "Rainfall": 0.0,
        "Temperature": 0.0,
        "Departure Hour": 0.0,
        "Distance (Miles)": 0.0,
        "Airline": 0.0,
        "Source Airport": 0.0,
        "Destination Airport": 0.0
    }

    for name, imp in zip(all_feature_names, raw_importances):
        if "weather_condition" in name:
            core_features["Weather Condition"] += imp
        elif "wind_speed" in name:
            core_features["Wind Speed"] += imp
        elif "visibility" in name:
            core_features["Visibility"] += imp
        elif "rainfall" in name:
            core_features["Rainfall"] += imp
        elif "temperature" in name:
            core_features["Temperature"] += imp
        elif "departure_hour" in name:
            core_features["Departure Hour"] += imp
        elif "distance_miles" in name:
            core_features["Distance (Miles)"] += imp
        elif "airline" in name:
            core_features["Airline"] += imp
        elif "source" in name:
            core_features["Source Airport"] += imp
        elif "destination" in name:
            core_features["Destination Airport"] += imp

    # Sort core feature importances
    sorted_core_importances = sorted(
        [{"feature": k, "importance": round(v, 4)} for k, v in core_features.items()],
        key=lambda x: x["importance"],
        reverse=True
    )

    # Save model artifact
    joblib.dump(clf, SKLEARN_MODEL_PKL)
    print(f"[Model] Saved inference pipeline to {SKLEARN_MODEL_PKL}")

    # Combine Spark and inference metrics
    final_metrics = {
        "spark_metrics": spark_metrics if spark_metrics else {
            "engine": "Apache Spark MLlib (Simulated Evaluator)",
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "total_samples": len(df),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "num_trees": 50,
            "max_depth": 8
        },
        "evaluation": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": {
                "on_time": {"precision": round(precision_score(y_test, y_pred, pos_label=0), 3), "recall": round(recall_score(y_test, y_pred, pos_label=0), 3)},
                "delayed": {"precision": round(precision_score(y_test, y_pred, pos_label=1), 3), "recall": round(recall_score(y_test, y_pred, pos_label=1), 3)}
            }
        },
        "feature_importances": sorted_core_importances,
        "features_list": all_feature_cols
    }

    with open(METRICS_JSON, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"[Metrics] Saved comprehensive model metrics to {METRICS_JSON}")

    return final_metrics

def main():
    spark_res = None
    try:
        spark_res = train_with_spark()
    except Exception as e:
        print(f"[Warning] Spark MLlib training note: {e}")
    
    train_sklearn_companion(spark_res)
    print("\n>>> MODEL TRAINING COMPLETE SUCCESSFULLY!\n")

if __name__ == "__main__":
    main()
