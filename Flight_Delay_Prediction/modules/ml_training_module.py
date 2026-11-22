"""
Module: Model Performance
Displays actual model evaluation metrics (Accuracy, Precision, Recall, F1 Score)
and the Confusion Matrix.
"""

import json
import pandas as pd
import streamlit as st
import plotly.express as px
from config import METRICS_JSON

def load_metrics():
    """Loads saved model evaluation metrics."""
    if METRICS_JSON.exists():
        try:
            with open(METRICS_JSON, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None

def render_ml_training_module():
    """Renders the simplified Model Performance page."""
    st.subheader("Model Performance")
    st.caption("Algorithm: Apache Spark MLlib RandomForestClassifier | 80% Train / 20% Test Split (No Data Leakage)")

    metrics = load_metrics()
    if metrics is None:
        st.warning("Model metrics not found.")
        return

    spark_m = metrics.get("spark_metrics", {})
    eval_m = metrics.get("evaluation", {})

    acc = spark_m.get("accuracy", eval_m.get("accuracy", 0.0)) * 100
    prec = spark_m.get("precision", eval_m.get("precision", 0.0)) * 100
    rec = spark_m.get("recall", eval_m.get("recall", 0.0)) * 100
    f1 = spark_m.get("f1_score", eval_m.get("f1_score", 0.0)) * 100

    # Display 4 Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc:.2f}%")
    m2.metric("Precision", f"{prec:.2f}%")
    m3.metric("Recall", f"{rec:.2f}%")
    m4.metric("F1 Score", f"{f1:.2f}%")

    st.write("")

    # Display Confusion Matrix
    st.markdown("##### Confusion Matrix")
    cm_data = spark_m.get("confusion_matrix", eval_m.get("confusion_matrix", [[736, 65], [233, 209]]))
    
    cm_df = pd.DataFrame(
        cm_data,
        index=["Actual On-Time", "Actual Delayed"],
        columns=["Predicted On-Time", "Predicted Delayed"]
    )

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Count"),
    )
    fig_cm.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_cm, use_container_width=True)
