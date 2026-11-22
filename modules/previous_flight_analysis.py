"""
Module: Previous Flight Analysis
Provides filtering on historical flights, record inspection,
and charts for delay rate and airline performance.
"""

import pandas as pd
import streamlit as st
import plotly.express as px
from config import DATASET_CSV

@st.cache_data
def load_historical_data() -> pd.DataFrame:
    """Loads historical dataset."""
    return pd.read_csv(DATASET_CSV)

def render_previous_flight_analysis():
    """Renders the simplified Previous Flight Analysis page."""
    st.subheader("Previous Flight Analysis")

    df = load_historical_data()
    if df.empty:
        st.warning("Historical flight dataset not found.")
        return

    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        airline_opts = ["All"] + sorted(df["airline"].dropna().unique().tolist())
        selected_airline = st.selectbox("Airline", airline_opts)
    with col2:
        src_opts = ["All"] + sorted(df["source"].dropna().unique().tolist())
        selected_src = st.selectbox("Source", src_opts)
    with col3:
        dst_opts = ["All"] + sorted(df["destination"].dropna().unique().tolist())
        selected_dst = st.selectbox("Destination", dst_opts)
    with col4:
        status_opts = ["All", "On-Time", "Delayed"]
        selected_status = st.selectbox("Delay Status", status_opts)

    # Filter data
    filtered_df = df.copy()
    if selected_airline != "All":
        filtered_df = filtered_df[filtered_df["airline"] == selected_airline]
    if selected_src != "All":
        filtered_df = filtered_df[filtered_df["source"] == selected_src]
    if selected_dst != "All":
        filtered_df = filtered_df[filtered_df["destination"] == selected_dst]
    if selected_status == "On-Time":
        filtered_df = filtered_df[filtered_df["is_delayed"] == 0]
    elif selected_status == "Delayed":
        filtered_df = filtered_df[filtered_df["is_delayed"] == 1]

    # Historical Records Table
    st.markdown("##### Historical Flight Records")
    display_cols = [
        "flight_number", "airline", "source", "destination", "flight_date",
        "departure_time", "arrival_time", "actual_arrival_time",
        "delay_duration_mins", "is_delayed"
    ]
    table_df = filtered_df[display_cols].copy().head(50)
    table_df["Status"] = table_df["is_delayed"].apply(lambda x: "Delayed" if x == 1 else "On Time")
    table_df = table_df.drop(columns=["is_delayed"]).rename(columns={
        "flight_number": "Flight No",
        "airline": "Airline",
        "source": "Source",
        "destination": "Destination",
        "flight_date": "Date",
        "departure_time": "Sched Dep",
        "arrival_time": "Sched Arr",
        "actual_arrival_time": "Actual Arr",
        "delay_duration_mins": "Delay (mins)"
    })

    st.dataframe(table_df, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(table_df)} of {len(filtered_df):,} matching flights.")

    st.write("")

    # Visualizations: Delay Rate chart and Airline-wise Delay chart
    c1, c2 = st.columns(2)

    with c1:
        # Delay Rate chart across departure hours
        hourly_stats = df.groupby("departure_hour").agg(
            total=("flight_number", "count"),
            delayed=("is_delayed", "sum")
        ).reset_index()
        hourly_stats["DelayRate"] = (hourly_stats["delayed"] / hourly_stats["total"]) * 100

        fig_delay_rate = px.line(
            hourly_stats,
            x="departure_hour",
            y="DelayRate",
            markers=True,
            title="Delay Rate by Departure Hour",
            labels={"departure_hour": "Departure Hour (0-23)", "DelayRate": "Delay Rate (%)"}
        )
        fig_delay_rate.update_traces(line_color="#2b6cb0", line_width=2.5)
        fig_delay_rate.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_delay_rate, use_container_width=True)

    with c2:
        # Airline-wise Delay chart
        airline_stats = df.groupby("airline")["is_delayed"].agg(["count", "mean"]).reset_index()
        airline_stats.columns = ["Airline", "Total", "DelayRate"]
        airline_stats["DelayPct"] = airline_stats["DelayRate"] * 100
        airline_stats = airline_stats.sort_values(by="DelayPct", ascending=True)

        fig_airline = px.bar(
            airline_stats,
            x="DelayPct",
            y="Airline",
            orientation="h",
            title="Airline-wise Delay Comparison",
            labels={"DelayPct": "Delay Rate (%)", "Airline": "Airline"},
            color="DelayPct",
            color_continuous_scale="Blues",
            text=airline_stats["DelayPct"].apply(lambda x: f"{x:.1f}%")
        )
        fig_airline.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig_airline, use_container_width=True)
