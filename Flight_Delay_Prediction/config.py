"""
Configuration and Reference Data for Flight Delay Prediction Big Data Project.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODULES_DIR = BASE_DIR / "modules"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# File Paths
DATASET_CSV = DATA_DIR / "flight_delay_dataset.csv"
BOOKINGS_CSV = DATA_DIR / "booking_records.csv"
METRICS_JSON = MODELS_DIR / "model_metrics.json"
SKLEARN_MODEL_PKL = MODELS_DIR / "trained_rf_classifier.pkl"
SPARK_MODEL_DIR = MODELS_DIR / "spark_rf_model"

# Airlines Reference
AIRLINES = {
    "Delta Air Lines": {"code": "DL", "hub": "ATL"},
    "American Airlines": {"code": "AA", "hub": "DFW"},
    "United Airlines": {"code": "UA", "hub": "ORD"},
    "Southwest Airlines": {"code": "WN", "hub": "DAL"},
    "JetBlue Airways": {"code": "B6", "hub": "JFK"},
    "Alaska Airlines": {"code": "AS", "hub": "SEA"},
    "SkyWest Airlines": {"code": "OO", "hub": "SLC"},
    "Spirit Airlines": {"code": "NK", "hub": "MCO"}
}

# Airports Reference
AIRPORTS = {
    "ATL": {"name": "Hartsfield-Jackson Atlanta Int'l", "city": "Atlanta, GA"},
    "DFW": {"name": "Dallas/Fort Worth Int'l", "city": "Dallas, TX"},
    "ORD": {"name": "O'Hare Int'l Airport", "city": "Chicago, IL"},
    "DEN": {"name": "Denver Int'l Airport", "city": "Denver, CO"},
    "LAX": {"name": "Los Angeles Int'l Airport", "city": "Los Angeles, CA"},
    "JFK": {"name": "John F. Kennedy Int'l", "city": "New York, NY"},
    "SFO": {"name": "San Francisco Int'l", "city": "San Francisco, CA"},
    "SEA": {"name": "Seattle-Tacoma Int'l", "city": "Seattle, WA"},
    "MIA": {"name": "Miami Int'l Airport", "city": "Miami, FL"},
    "BOS": {"name": "Boston Logan Int'l", "city": "Boston, MA"}
}

# Weather Conditions
WEATHER_CONDITIONS = ["Clear", "Partly Cloudy", "Fog", "Light Rain", "Heavy Rain", "Thunderstorm", "Snow"]

# Delay Risk Levels
RISK_LEVELS = {
    "Low": {"color": "#28a745", "desc": "Minimal weather hazard. Expected on-time departure."},
    "Medium": {"color": "#fd7e14", "desc": "Moderate weather disturbance. Moderate probability of minor delay (15-45 mins)."},
    "High": {"color": "#dc3545", "desc": "Severe weather condition. High likelihood of major delay (>45 mins) or hold."}
}
