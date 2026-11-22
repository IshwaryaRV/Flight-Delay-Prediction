"""
Main Application: Flight Delay Prediction Using Big Data
Clean, simple, and neat college student project interface.
"""

import streamlit as st
from modules.dashboard_module import render_dashboard
from modules.booking_module import render_booking_module
from modules.previous_flight_analysis import render_previous_flight_analysis
from modules.current_flight_prediction import render_current_flight_prediction
from modules.weather_risk_module import render_weather_risk_module
from modules.ml_training_module import render_ml_training_module

# Page Configuration
st.set_page_config(
    page_title="Flight Delay Prediction Using Big Data",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, simple CSS with white background and blue accents
st.markdown("""
<style>
    /* White background */
    .stApp {
        background-color: #ffffff;
        color: #1a202c;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    }
    
    /* Header title */
    .project-header {
        padding: 10px 0px 14px 0px;
        border-bottom: 2px solid #2b6cb0;
        margin-bottom: 20px;
    }
    .project-header h2 {
        color: #2b6cb0;
        margin: 0;
        font-size: 24px;
        font-weight: 600;
    }
    
    /* Clean sidebar */
    [data-testid="stSidebar"] {
        background-color: #f7fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Primary buttons */
    .stButton > button {
        background-color: #2b6cb0;
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 500;
        padding: 6px 16px;
    }
    .stButton > button:hover {
        background-color: #2c5282;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Top Header
    st.markdown("""
    <div class="project-header">
        <h2>Flight Delay Prediction Using Big Data</h2>
    </div>
    """, unsafe_allow_html=True)

    # Simplified Sidebar
    with st.sidebar:
        st.markdown("### Navigation")
        pages = [
            "Dashboard",
            "Flight Booking",
            "Previous Flight Analysis",
            "Flight Delay Prediction",
            "Weather Risk",
            "Model Performance"
        ]

        if "nav_page" not in st.session_state:
            st.session_state["nav_page"] = "Dashboard"

        # Safe index lookup
        try:
            curr_idx = pages.index(st.session_state["nav_page"])
        except ValueError:
            curr_idx = 0

        selected_page = st.radio(
            "Go to:",
            pages,
            index=curr_idx,
            label_visibility="collapsed"
        )
        st.session_state["nav_page"] = selected_page

    # Page Routing
    if selected_page == "Dashboard":
        render_dashboard()
    elif selected_page == "Flight Booking":
        render_booking_module()
    elif selected_page == "Previous Flight Analysis":
        render_previous_flight_analysis()
    elif selected_page == "Flight Delay Prediction":
        render_current_flight_prediction()
    elif selected_page == "Weather Risk":
        render_weather_risk_module()
    elif selected_page == "Model Performance":
        render_ml_training_module()

if __name__ == "__main__":
    main()
