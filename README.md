# Prediction of Flight Delays Using Big Data

A complete Big Data analytics and machine learning application developed for commercial airline operations. The system processes extensive historical flight archives using **Apache Spark (PySpark)** and trains an ensemble **Random Forest Classifier** to predict flight delays, estimate delay durations, and evaluate meteorological risk factors with a **Streamlit** dashboard frontend.

---

## 📌 Project Overview

Flight delays cause billions of dollars in losses annually and disrupt travel schedules globally. This project leverages Big Data technologies to:
1. Aggregate and analyze thousands of historical commercial flight operations.
2. Build an end-to-end distributed Machine Learning pipeline with **PySpark MLlib**.
3. Identify primary root causes of flight delays, including adverse weather, air traffic control congestion, and carrier turnaround delays.
4. Provide a passenger booking portal and real-time flight delay & duration prediction interface.
5. Deliver an interactive dashboard for operational monitoring.

---

## 🛠️ Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Primary development language |
| **Big Data Engine** | Apache Spark / PySpark | Distributed DataFrame processing, Spark SQL queries |
| **Machine Learning** | PySpark MLlib & Scikit-Learn | RandomForestClassifier, StringIndexer, OneHotEncoder, VectorAssembler |
| **Web Frontend** | Streamlit | Clean, professional airline portal and dashboard |
| **Data Manipulation** | Pandas, NumPy | High-performance in-memory tabular manipulation |
| **Visualization** | Plotly & Matplotlib | Interactive KPI gauges, trend lines, heatmaps, and bar charts |
| **Storage** | CSV Datasets | Big Data historical flights archive (`data/flight_delay_dataset.csv`) |

---

## 🚀 Key Modules

### 1. Flight Booking Module
- Collects passenger details (Full Name, Age, Email) and itinerary information (Flight Number, Origin, Destination, Travel Date, Scheduled Time).
- Generates an automated 6-character alphanumeric booking reference (PNR).
- Renders an electronic boarding pass receipt card with seat assignment and flight routing.
- Automatically saves booking records to `data/booking_records.csv`.
- Includes a 1-click shortcut to check delay prediction for the booked flight.

### 2. Previous Flight Analysis
- Displays historical flight details with dynamic multi-criteria filtering (Airline, Origin, Destination, Delay Status).
- Compares scheduled vs. actual arrival times and delay magnitudes.
- Visualizes delay patterns:
  - Airline On-Time Reliability rankings.
  - 24-hour departure congestion patterns (identifying peak evening cascading delays).
  - Top 10 high-traffic flight corridors and route delay frequencies.
- Interactive **Apache Spark SQL Query Engine** console for custom distributed SQL queries.

### 3. Current Flight Delay Prediction
- Accepts real-time or simulated flight inputs (Airline, Origin, Destination, Hour, Distance, Temperature, Wind Speed, Visibility, Rainfall, Weather Condition).
- Includes one-click demonstration scenarios (Clear Morning Flight, Evening Thunderstorm, Winter Snow).
- Predicts binary delay status (**ON TIME** or **DELAYED**) using the trained Random Forest model.
- Calculates anticipated delay duration (e.g., `0 - 10 mins` for on-time or `45 - 85 mins` for delayed flights).
- Features an interactive **Delay Probability Gauge** and operational risk assessment breakdown.

### 4. Weather Risk Prediction
- Assesses ambient meteorological stressors: temperature, wind speed, visibility, rainfall, and condition.
- Computes a composite **Meteorological Hazard Index (0 to 100)**.
- Classifies flight weather risk into **Low**, **Medium**, or **High** risk levels with specific advisory notes.
- Feature importance ranking based on the trained Random Forest model, showing that Visibility, Weather Phenomena, and Wind Speed are top contributors.
- Multi-Airport Weather Status table showing simulated telemetry across 8 major commercial hubs (ATL, ORD, DFW, DEN, JFK, LAX, SFO, MIA).

### 5. Machine Learning & Model Evaluation
- Distributed machine learning pipeline using **PySpark MLlib** (`pyspark.ml.classification.RandomForestClassifier`).
- **Strict Data Leakage Prevention:** 80% train / 20% test partition executed prior to fitting feature indexers or encoders; no post-flight target variables included in feature matrix.
- Comprehensive performance evaluation on held-out test data:
  - **Accuracy:** ~76.0%
  - **Precision (Weighted):** ~76.1%
  - **Recall (Weighted):** ~76.0%
  - **F1-Score:** ~74.4%
  - **ROC-AUC:** ~0.73
- Interactive Confusion Matrix heatmap (True Positives, False Positives, True Negatives, False Negatives).
- Feature importance visualization and live model retraining capability.

### 6. Executive Overview Dashboard
- High-level KPI metric cards:
  - Total Flights (6,500)
  - On-Time Flights (4,125 / 63.5%)
  - Delayed Flights (2,375 / 36.5%)
  - Weather-Related Delays (742 / 31.2% of delays)
  - Average Delay Duration when delayed (~52 mins)
- Graphical charts:
  - Monthly Flight Delay and On-Time volume trend line.
  - Delay Primary Cause donut chart (Weather, Late Aircraft, NAS / Air Traffic, Carrier).
  - Airline delay rate comparisons.
  - Weather condition delay likelihood distribution.

---

## 📂 Project Directory Structure

```
Flight_Delay_Prediction/
├── app.py                             # Main Streamlit web application & multi-page navigation
├── config.py                          # Global configuration, directory paths, airport/airline constants
├── generate_dataset.py                # Standalone Big Data historical dataset generator
├── train_spark_model.py               # Standalone PySpark MLlib model training & evaluation script
├── requirements.txt                   # Project Python dependencies
├── README.md                          # Comprehensive documentation
├── data/
│   ├── flight_delay_dataset.csv       # 6,500 historical flight records
│   └── booking_records.csv            # Persistent passenger reservation storage
├── models/
│   ├── spark_rf_model/                # Serialized PySpark MLlib Pipeline model
│   ├── model_metrics.json             # Evaluator metrics, confusion matrix, feature importances
│   └── trained_rf_classifier.pkl      # Serialized inference pipeline artifact
└── modules/
    ├── __init__.py
    ├── spark_manager.py               # PySpark session builder, worker environment & Spark SQL
    ├── booking_module.py              # Module 1: Passenger reservation & boarding pass issuance
    ├── previous_flight_analysis.py    # Module 2: Historical analysis, delay patterns & Spark SQL
    ├── current_flight_prediction.py   # Module 3: Delay status prediction & duration estimation
    ├── weather_risk_module.py         # Module 4: Meteorological risk scoring & factor ranking
    ├── ml_training_module.py          # Module 5: PySpark MLlib pipeline & model evaluation
    └── dashboard_module.py            # Module 6: Executive KPI overview & delay trend charts
```

---

## ⚙️ Installation & Execution Guide

### 1. Prerequisites
- **Python:** Version 3.10 to 3.13 installed.
- **Java:** Java Runtime Environment (JRE/JDK) version 8, 11, 17, or higher (required by Apache Spark).

### 2. Install Dependencies
Open a terminal in the project directory and run:
```bash
pip install -r requirements.txt
```

### 3. Generate the Dataset
Create the historical flight dataset (6,500 records):
```bash
python generate_dataset.py
```

### 4. Train the PySpark MLlib Model
Train the Random Forest Classifier and generate evaluation metrics:
```bash
python train_spark_model.py
```

### 5. Launch the Streamlit Web Application
Run the Streamlit application:
```bash
streamlit run app.py
```
The application will open in your web browser at: `http://localhost:8501`.

---

## 📊 Dataset Schema

| Field | Data Type | Description |
| :--- | :--- | :--- |
| `flight_number` | String | Unique flight identifier (e.g., `DL1042`, `AA4521`) |
| `airline` | String | Commercial airline name |
| `source` | String | Origin 3-letter IATA airport code |
| `destination` | String | Destination 3-letter IATA airport code |
| `flight_date` | String | Scheduled departure date (`YYYY-MM-DD`) |
| `departure_time` | String | Scheduled departure time (`HH:MM`) |
| `departure_hour` | Integer | Scheduled departure hour (`0 - 23`) |
| `arrival_time` | String | Scheduled arrival time (`HH:MM`) |
| `actual_arrival_time` | String | Actual recorded arrival time (`HH:MM`) |
| `distance_miles` | Integer | Great-circle flight distance in miles |
| `temperature_c` | Float | Surface temperature in Celsius |
| `wind_speed_mph` | Float | Surface wind speed in miles per hour |
| `visibility_miles` | Float | Horizontal visibility in miles |
| `rainfall_mm` | Float | Precipitation accumulation in millimeters |
| `weather_condition` | String | Observed condition (`Clear`, `Fog`, `Snow`, `Thunderstorm`, etc.) |
| `delay_duration_mins` | Integer | Departure delay in minutes (`0` if on time) |
| `delay_reason` | String | Primary delay cause (`Weather`, `Carrier`, `Late Aircraft`, `NAS`, `None`) |
| `is_delayed` | Integer | Binary classification label (`0` = On-Time, `1` = Delayed >15 mins) |
| `data_source_type` | String | Explicit sample metadata classification |

---

## 🛡️ Academic & Ethical Disclosures
- **Sample Data Usage:** This project uses simulated Big Data historical flight records based on US commercial routes and FAA delay distributions. It is explicitly labeled for academic evaluation.
- **Privacy Protection:** Only sample demonstration passenger details (Name, Age, Email) are processed for reservation simulation. No private personal data or confidential credentials are collected or transmitted.
- **Live APIs:** To ensure offline portability and reliability for college laboratory evaluations, no paid or volatile third-party live APIs are required.
