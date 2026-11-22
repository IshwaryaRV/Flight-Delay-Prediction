"""
Module: Flight Delay Prediction
Predicts flight delay status, estimated duration, and probability
using the trained Random Forest Classifier.
"""

import joblib
import pandas as pd
import numpy as np
import streamlit as st
from config import SKLEARN_MODEL_PKL, AIRLINES, AIRPORTS, WEATHER_CONDITIONS

@st.cache_resource
def load_rf_model():
    """Loads the trained Random Forest inference pipeline."""
    if SKLEARN_MODEL_PKL.exists():
        try:
            return joblib.load(SKLEARN_MODEL_PKL)
        except Exception as e:
            st.error(f"Error loading model: {e}")
    return None

def estimate_delay_duration(is_delayed: int, prob_delayed: float, weather: str, wind_speed: float, dep_hour: int) -> dict:
    """Estimates flight delay duration based on conditions."""
    if is_delayed == 0:
        return {
            "range": "0 - 10 minutes",
            "point_estimate": round(np.random.uniform(0, 8), 0)
        }

    base_mins = 20 + (prob_delayed * 40)
    if weather in ["Thunderstorm", "Snow"]:
        base_mins += 35
    elif weather in ["Heavy Rain", "Fog"]:
        base_mins += 20
    elif weather == "Light Rain":
        base_mins += 8

    if wind_speed > 30:
        base_mins += 18
    elif wind_speed > 20:
        base_mins += 8

    if 16 <= dep_hour <= 20:
        base_mins += 15

    est_min = max(16, int(base_mins * 0.8))
    est_max = int(base_mins * 1.25)
    est_point = int((est_min + est_max) / 2)

    return {
        "range": f"{est_min} - {est_max} minutes",
        "point_estimate": est_point
    }

def render_current_flight_prediction():
    """Renders the simplified Flight Delay Prediction page."""
    st.subheader("Flight Delay Prediction")

    model = load_rf_model()
    if model is None:
        st.error("Trained model not found.")
        return

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Flight Details")
            airline = st.selectbox("Airline", list(AIRLINES.keys()))
            default_fl = f"{AIRLINES[airline]['code']}452"
            flight_number = st.text_input("Flight Number", value=default_fl)
            source = st.selectbox("Source Airport", list(AIRPORTS.keys()), index=0)
            dest_options = [c for c in list(AIRPORTS.keys()) if c != source]
            destination = st.selectbox("Destination Airport", dest_options, index=0)
            departure_hour = st.slider("Departure Hour", 0, 23, 10, format="%02d:00")
            distance = st.number_input("Distance (Miles)", min_value=100, max_value=4000, value=750, step=50)

        with col2:
            st.markdown("##### Weather Details")
            weather = st.selectbox("Weather Condition", WEATHER_CONDITIONS, index=0)
            temp = st.slider("Temperature (°C)", -20.0, 45.0, 22.0, step=0.5)
            wind = st.slider("Wind Speed (mph)", 0.0, 60.0, 10.0, step=0.5)
            vis = st.slider("Visibility (Miles)", 0.1, 10.0, 8.0, step=0.1)
            rain = st.slider("Rainfall (mm)", 0.0, 70.0, 0.0, step=1.0)

        predict_btn = st.form_submit_button("Predict Delay", type="primary")

    if predict_btn:
        input_dict = {
            "airline": [airline],
            "source": [source],
            "destination": [destination],
            "weather_condition": [weather],
            "departure_hour": [departure_hour],
            "distance_miles": [distance],
            "temperature_c": [temp],
            "wind_speed_mph": [wind],
            "visibility_miles": [vis],
            "rainfall_mm": [rain]
        }
        input_df = pd.DataFrame(input_dict)

        try:
            pred_class = int(model.predict(input_df)[0])
            pred_probs = model.predict_proba(input_df)[0]
            prob_delayed = float(pred_probs[1])

            est_duration = estimate_delay_duration(
                is_delayed=pred_class,
                prob_delayed=prob_delayed,
                weather=weather,
                wind_speed=wind,
                dep_hour=departure_hour
            )

            st.write("")
            st.markdown("##### Prediction Result")

            res_col1, res_col2, res_col3 = st.columns(3)

            if pred_class == 1:
                res_col1.error("Status: **DELAYED**")
            else:
                res_col1.success("Status: **ON TIME**")

            res_col2.info(f"Estimated Delay Duration: **{est_duration['range']}**")
            res_col3.metric("Delay Probability", f"{prob_delayed * 100:.1f}%")

        except Exception as e:
            st.error(f"Prediction error: {e}")
