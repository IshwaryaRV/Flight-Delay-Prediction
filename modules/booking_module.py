"""
Module: Flight Booking
Collects passenger information, registers a flight reservation,
and displays a simple booking confirmation.
"""

import random
import string
from datetime import datetime, date
import pandas as pd
import streamlit as st
from config import BOOKINGS_CSV, AIRLINES, AIRPORTS

def generate_pnr() -> str:
    """Generates a 6-character alphanumeric booking reference."""
    chars = string.ascii_uppercase + string.digits
    return "BK" + "".join(random.choices(chars, k=4))

def load_bookings() -> pd.DataFrame:
    """Loads existing booking records from CSV."""
    if BOOKINGS_CSV.exists():
        try:
            return pd.read_csv(BOOKINGS_CSV)
        except Exception:
            pass
    return pd.DataFrame(columns=[
        "booking_id", "passenger_name", "passenger_age", "passenger_email",
        "flight_number", "airline", "source", "destination", "travel_date",
        "departure_time", "seat_number", "booking_status", "booking_timestamp"
    ])

def save_booking(record: dict) -> bool:
    """Appends a new booking record to CSV."""
    try:
        df = load_bookings()
        new_row = pd.DataFrame([record])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(BOOKINGS_CSV, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving booking record: {e}")
        return False

def render_booking_module():
    """Renders the simplified Flight Booking page."""
    st.subheader("Flight Booking")

    with st.form("flight_booking_form"):
        col1, col2 = st.columns(2)
        with col1:
            p_name = st.text_input("Passenger Name", placeholder="Enter full name")
            p_age = st.number_input("Age", min_value=1, max_value=110, value=25)
            p_email = st.text_input("Email", placeholder="Enter email address")
            travel_date = st.date_input("Travel Date", min_value=date.today())

        with col2:
            airline_selected = st.selectbox("Airline", list(AIRLINES.keys()))
            default_fl = f"{AIRLINES[airline_selected]['code']}{random.randint(100, 999)}"
            flight_no = st.text_input("Flight Number", value=default_fl)
            src_code = st.selectbox("Source", list(AIRPORTS.keys()), index=0)
            dest_options = [c for c in list(AIRPORTS.keys()) if c != src_code]
            dst_code = st.selectbox("Destination", dest_options, index=0)

        book_btn = st.form_submit_button("Book Flight", type="primary")

    if book_btn:
        if not p_name.strip():
            st.error("Please enter passenger name.")
        elif not p_email.strip() or "@" not in p_email:
            st.error("Please enter a valid email address.")
        elif src_code == dst_code:
            st.error("Source and destination cannot be the same airport.")
        else:
            pnr = generate_pnr()
            record = {
                "booking_id": pnr,
                "passenger_name": p_name.strip(),
                "passenger_age": int(p_age),
                "passenger_email": p_email.strip(),
                "flight_number": flight_no.strip().upper(),
                "airline": airline_selected,
                "source": src_code,
                "destination": dst_code,
                "travel_date": travel_date.strftime("%Y-%m-%d"),
                "departure_time": "10:00",
                "seat_number": f"{random.randint(1, 30)}{random.choice(['A','B','C','D'])}",
                "booking_status": "Confirmed",
                "booking_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            if save_booking(record):
                st.success("Booking Successful!")
                st.markdown(f"""
                **Booking Confirmation Details:**
                - **Booking ID (PNR):** {pnr}
                - **Passenger Name:** {p_name.strip()} (Age: {int(p_age)})
                - **Flight Number:** {flight_no.strip().upper()} ({airline_selected})
                - **Route:** {src_code} ➔ {dst_code}
                - **Travel Date:** {travel_date.strftime("%Y-%m-%d")}
                - **Status:** Confirmed
                """)

    # Display recent bookings
    df_all = load_bookings()
    if not df_all.empty:
        st.write("")
        st.markdown("##### Recent Bookings")
        display_df = df_all[[
            "booking_id", "passenger_name", "flight_number", "airline",
            "source", "destination", "travel_date", "booking_status"
        ]].rename(columns={
            "booking_id": "PNR",
            "passenger_name": "Passenger",
            "flight_number": "Flight",
            "airline": "Airline",
            "source": "From",
            "destination": "To",
            "travel_date": "Date",
            "booking_status": "Status"
        }).tail(5)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
