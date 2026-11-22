"""
Module: Weather Risk
Evaluates weather risk based on temperature, wind speed,
visibility, rainfall, and weather condition.
"""

import json
import pandas as pd
import plotly.express as px
import streamlit as st
from config import WEATHER_CONDITIONS, RISK_LEVELS, METRICS_JSON

def calculate_weather_risk_score(temp: float, wind: float, vis: float, rain: float, weather: str) -> dict:
    """Computes meteorological delay risk score (0 - 100)."""
    score = 0.0

    # 1. Condition severity
    if weather == "Thunderstorm":
        score += 40
    elif weather == "Snow":
        score += 35
    elif weather == "Heavy Rain":
        score += 25
    elif weather == "Fog":
        score += 22
    elif weather == "Light Rain":
        score += 10
    elif weather == "Partly Cloudy":
        score += 4
    else:
        score += 0

    # 2. Wind speed (mph)
    if wind >= 35:
        score += 25
    elif wind >= 25:
        score += 16
    elif wind >= 15:
        score += 8
    else:
        score += 2

    # 3. Visibility (Miles)
    if vis <= 1.0:
        score += 25
    elif vis <= 3.0:
        score += 18
    elif vis <= 5.0:
        score += 8
    else:
        score += 1

    # 4. Rainfall (mm)
    if rain >= 30:
        score += 18
    elif rain >= 15:
        score += 10
    elif rain >= 5:
        score += 4

    # 5. Temperature extremes
    if temp <= -5.0:
        score += 10
    elif temp >= 38.0:
        score += 6

    final_score = min(max(int(score), 5), 100)

    if final_score < 30:
        level = "Low"
    elif final_score <= 65:
        level = "Medium"
    else:
        level = "High"

    return {
        "score": final_score,
        "level": level,
        "color": RISK_LEVELS[level]["color"],
        "description": RISK_LEVELS[level]["desc"]
    }

def render_weather_risk_module():
    """Renders the simplified Weather Risk page."""
    st.subheader("Weather Risk")

    with st.form("weather_risk_form"):
        col1, col2 = st.columns(2)
        with col1:
            weather = st.selectbox("Weather Condition", WEATHER_CONDITIONS, index=0)
            temp = st.slider("Temperature (°C)", -20.0, 45.0, 20.0, step=1.0)
            wind = st.slider("Wind Speed (mph)", 0.0, 60.0, 10.0, step=1.0)

        with col2:
            vis = st.slider("Visibility (Miles)", 0.1, 10.0, 8.0, step=0.1)
            rain = st.slider("Rainfall (mm)", 0.0, 70.0, 0.0, step=1.0)

        check_btn = st.form_submit_button("Check Risk", type="primary")

    if check_btn:
        res = calculate_weather_risk_score(temp, wind, vis, rain, weather)

        st.write("")
        st.markdown("##### Assessment Result")

        r1, r2, r3 = st.columns(3)
        if res["level"] == "High":
            r1.error(f"Risk Level: **{res['level']}**")
        elif res["level"] == "Medium":
            r1.warning(f"Risk Level: **{res['level']}**")
        else:
            r1.success(f"Risk Level: **{res['level']}**")

        r2.info(f"Risk Score: **{res['score']} / 100**")
        r3.markdown(f"**Explanation:** {res['description']}")

    # Factors contributing most to flight delays
    st.write("")
    st.markdown("##### Factors Contributing Most to Flight Delays")
    if METRICS_JSON.exists():
        try:
            with open(METRICS_JSON, "r") as f:
                metrics_data = json.load(f)
            importances = metrics_data.get("feature_importances", [])
            imp_df = pd.DataFrame(importances).head(6)
            imp_df["Importance (%)"] = imp_df["importance"] * 100

            fig_imp = px.bar(
                imp_df,
                x="Importance (%)",
                y="feature",
                orientation="h",
                labels={"Importance (%)": "Relative Importance (%)", "feature": "Factor"},
                color="Importance (%)",
                color_continuous_scale="Blues",
                text=imp_df["Importance (%)"].apply(lambda x: f"{x:.1f}%")
            )
            fig_imp.update_layout(
                yaxis=dict(autorange="reversed"),
                height=250,
                margin=dict(l=20, r=20, t=10, b=20),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception:
            pass
