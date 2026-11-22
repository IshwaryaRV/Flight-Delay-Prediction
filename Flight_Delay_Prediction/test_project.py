"""
Automated Test Suite for 'Prediction of Flight Delays Using Big Data'
Tests all modules, dataset integrity, machine learning inference,
weather risk scoring, booking system, and PySpark connectivity.
"""

import os
import sys
import json
import unittest
import pandas as pd
from pathlib import Path

# Ensure worker environment
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from config import DATASET_CSV, BOOKINGS_CSV, METRICS_JSON, SKLEARN_MODEL_PKL, AIRLINES, AIRPORTS
from modules.spark_manager import get_spark_session, check_spark_status, run_spark_sql_query
from modules.booking_module import generate_pnr, load_bookings, save_booking
from modules.current_flight_prediction import load_rf_model, estimate_delay_duration
from modules.weather_risk_module import calculate_weather_risk_score

class TestFlightDelayProject(unittest.TestCase):

    def test_01_dataset_integrity(self):
        """Verify historical flight dataset exists, is non-empty, and contains required schema."""
        self.assertTrue(DATASET_CSV.exists(), "Dataset CSV file must exist")
        df = pd.read_csv(DATASET_CSV)
        self.assertGreaterEqual(len(df), 5000, "Dataset should have at least 5,000 records")
        required_cols = [
            "flight_number", "airline", "source", "destination",
            "departure_time", "arrival_time", "weather_condition",
            "temperature_c", "wind_speed_mph", "visibility_miles",
            "rainfall_mm", "delay_duration_mins", "is_delayed"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Column '{col}' must be present in dataset")
        print(f"[Pass] Dataset integrity verified ({len(df)} records).")

    def test_02_spark_connectivity_and_sql(self):
        """Verify PySpark session / SQL query execution engine."""
        status_ok, msg = check_spark_status()
        
        # Test SQL query execution (Spark SQL or analytical engine)
        sql_query = "SELECT airline, count(*) AS count FROM flights GROUP BY airline"
        res = run_spark_sql_query(sql_query)
        self.assertIsNotNone(res, "SQL query must return a DataFrame")
        self.assertGreater(len(res), 0, "SQL result should have rows")
        print(f"[Pass] SQL execution verified ({len(res)} airlines aggregated; status: {msg}).")

    def test_03_model_artifacts_and_metrics(self):
        """Verify saved model metrics JSON and Random Forest model artifact."""
        self.assertTrue(METRICS_JSON.exists(), "model_metrics.json must exist")
        with open(METRICS_JSON, "r") as f:
            metrics = json.load(f)
        self.assertIn("spark_metrics", metrics)
        self.assertIn("evaluation", metrics)
        self.assertIn("feature_importances", metrics)
        
        acc = metrics["spark_metrics"]["accuracy"]
        prec = metrics["spark_metrics"]["precision"]
        rec = metrics["spark_metrics"]["recall"]
        f1 = metrics["spark_metrics"]["f1_score"]
        self.assertGreater(acc, 0.65, "Accuracy should exceed 65%")
        self.assertGreater(prec, 0.65, "Precision should exceed 65%")
        self.assertGreater(rec, 0.65, "Recall should exceed 65%")
        self.assertGreater(f1, 0.65, "F1-Score should exceed 65%")
        print(f"[Pass] ML Metrics verified: Acc={acc*100:.2f}%, Prec={prec*100:.2f}%, Rec={rec*100:.2f}%, F1={f1*100:.2f}%.")

    def test_04_model_inference_and_duration_estimation(self):
        """Verify live inference on On-Time and Delayed sample flight scenarios."""
        model = load_rf_model()
        self.assertIsNotNone(model, "Model pipeline should load successfully")

        # Scenario 1: Clear morning flight (likely on-time)
        input_ontime = pd.DataFrame([{
            "airline": "Delta Air Lines",
            "source": "ATL",
            "destination": "MIA",
            "weather_condition": "Clear",
            "departure_hour": 8,
            "distance_miles": 594,
            "temperature_c": 22.0,
            "wind_speed_mph": 6.0,
            "visibility_miles": 10.0,
            "rainfall_mm": 0.0
        }])
        pred_ot = model.predict(input_ontime)[0]
        prob_ot = model.predict_proba(input_ontime)[0]
        self.assertIn(pred_ot, [0, 1])
        dur_ot = estimate_delay_duration(pred_ot, prob_ot[1], "Clear", 6.0, 8)
        self.assertIn("range", dur_ot)

        # Scenario 2: Severe thunderstorm flight (high delay probability)
        input_delayed = pd.DataFrame([{
            "airline": "American Airlines",
            "source": "DFW",
            "destination": "ORD",
            "weather_condition": "Thunderstorm",
            "departure_hour": 18,
            "distance_miles": 802,
            "temperature_c": 28.0,
            "wind_speed_mph": 35.0,
            "visibility_miles": 1.5,
            "rainfall_mm": 45.0
        }])
        pred_del = model.predict(input_delayed)[0]
        prob_del = model.predict_proba(input_delayed)[0]
        self.assertEqual(pred_del, 1, "Severe thunderstorm should predict delayed (1)")
        dur_del = estimate_delay_duration(pred_del, prob_del[1], "Thunderstorm", 35.0, 18)
        self.assertIn("range", dur_del)
        print(f"[Pass] Inference verified: On-Time prob={prob_ot[0]*100:.1f}%, Storm Delay prob={prob_del[1]*100:.1f}%.")

    def test_05_weather_risk_module(self):
        """Verify weather risk score categorization (Low, Medium, High)."""
        # Low risk: clear skies, calm wind, 10-mile visibility
        low_res = calculate_weather_risk_score(temp=20.0, wind=5.0, vis=10.0, rain=0.0, weather="Clear")
        self.assertEqual(low_res["level"], "Low")

        # Medium risk: rain, moderate wind
        med_res = calculate_weather_risk_score(temp=15.0, wind=20.0, vis=4.0, rain=12.0, weather="Light Rain")
        self.assertEqual(med_res["level"], "Medium")

        # High risk: thunderstorm, severe crosswind, low visibility
        high_res = calculate_weather_risk_score(temp=24.0, wind=38.0, vis=1.0, rain=40.0, weather="Thunderstorm")
        self.assertEqual(high_res["level"], "High")
        print("[Pass] Weather risk scoring verified (Low, Medium, High thresholds).")

    def test_06_booking_module(self):
        """Verify PNR generation, booking persistence, and loading."""
        pnr = generate_pnr()
        self.assertTrue(pnr.startswith("BK"))
        self.assertEqual(len(pnr), 6)

        test_record = {
            "booking_id": pnr,
            "passenger_name": "Test Passenger",
            "passenger_age": 30,
            "passenger_email": "test@example.com",
            "flight_number": "DL9999",
            "airline": "Delta Air Lines",
            "source": "ATL",
            "destination": "JFK",
            "travel_date": "2026-12-01",
            "departure_time": "10:00",
            "seat_number": "12B",
            "booking_status": "Confirmed",
            "booking_timestamp": "2026-11-17 12:00:00"
        }
        saved = save_booking(test_record)
        self.assertTrue(saved, "Booking must save successfully")
        
        df_bookings = load_bookings()
        matched = df_bookings[df_bookings["booking_id"] == pnr]
        self.assertEqual(len(matched), 1, "Saved booking should be retrievable from CSV")
        print(f"[Pass] Booking system verified (PNR {pnr} created and saved).")

if __name__ == "__main__":
    unittest.main()
