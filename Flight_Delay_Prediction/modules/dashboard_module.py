"""
Module: Dashboard
Displays key flight statistics and essential analytical charts.
"""

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from config import DATASET_CSV

@st.cache_data
def load_dashboard_data() -> pd.DataFrame:
    """Loads historical dataset for dashboard analytics."""
    return pd.read_csv(DATASET_CSV)

def render_dashboard():
    """Renders the simplified Dashboard."""
    df = load_dashboard_data()
    if df.empty:
        st.warning("Historical dataset not found.")
        return

    st.subheader("Dashboard")

    # Calculate required KPIs
    total_flights = len(df)
    delayed_flights = int(df["is_delayed"].sum())
    ontime_flights = total_flights - delayed_flights
    weather_delays = int((df["delay_reason"] == "Weather").sum())
    avg_delay = df[df["is_delayed"] == 1]["delay_duration_mins"].mean()

    # Display the required KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Flights", f"{total_flights:,}")
    col2.metric("On-Time Flights", f"{ontime_flights:,}")
    col3.metric("Delayed Flights", f"{delayed_flights:,}")
    col4.metric("Weather Delays", f"{weather_delays:,}")
    col5.metric("Average Delay", f"{avg_delay:.0f} mins")

    st.write("")

    # Chart 1: Monthly Flight Delay
    df["month_year"] = pd.to_datetime(df["flight_date"]).dt.to_period("M").astype(str)
    monthly_trend = df.groupby("month_year").agg(
        total=("flight_number", "count"),
        delayed=("is_delayed", "sum")
    ).reset_index()
    monthly_trend["ontime"] = monthly_trend["total"] - monthly_trend["delayed"]

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(
        x=monthly_trend["month_year"],
        y=monthly_trend["ontime"],
        name="On-Time",
        marker_color="#2b6cb0"
    ))
    fig_trend.add_trace(go.Bar(
        x=monthly_trend["month_year"],
        y=monthly_trend["delayed"],
        name="Delayed",
        marker_color="#e53e3e"
    ))
    fig_trend.update_layout(
        title="Monthly Flight Delay",
        barmode="stack",
        xaxis_title="Month",
        yaxis_title="Number of Flights",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Row with Charts 2 and 3: Delay Reasons & Airline-wise Delay Comparison
    c1, c2 = st.columns(2)

    with c1:
        # Chart 2: Delay Reasons
        delay_causes = df[df["is_delayed"] == 1]["delay_reason"].value_counts().reset_index()
        delay_causes.columns = ["Reason", "Count"]

        fig_reasons = px.pie(
            delay_causes,
            values="Count",
            names="Reason",
            title="Delay Reasons",
            color_discrete_sequence=["#2b6cb0", "#4299e1", "#ed8936", "#e53e3e"]
        )
        fig_reasons.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_reasons, use_container_width=True)

    with c2:
        # Chart 3: Airline-wise Delay Comparison
        airline_delay = df.groupby("airline")["is_delayed"].agg(["count", "mean"]).reset_index()
        airline_delay.columns = ["Airline", "Total", "DelayRate"]
        airline_delay["DelayPct"] = airline_delay["DelayRate"] * 100
        airline_delay = airline_delay.sort_values(by="DelayPct", ascending=True)

        fig_airline = px.bar(
            airline_delay,
            x="DelayPct",
            y="Airline",
            orientation="h",
            title="Airline-wise Delay Comparison",
            labels={"DelayPct": "Delay Rate (%)", "Airline": "Airline"},
            color="DelayPct",
            color_continuous_scale="Blues",
            text=airline_delay["DelayPct"].apply(lambda x: f"{x:.1f}%")
        )
        fig_airline.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig_airline, use_container_width=True)
