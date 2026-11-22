"""
Dataset Generator for 'Prediction of Flight Delays Using Big Data'

This script synthesizes a realistic, high-fidelity historical flight operations dataset
modeling major US commercial routes, realistic meteorological parameters, operational
delays, and delay status.

Clearly designated as Sample / Academic Historical Data for Demonstration.
"""

import os
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from config import DATASET_CSV, AIRLINES, AIRPORTS, WEATHER_CONDITIONS

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_flight_dataset(num_records: int = 6000) -> pd.DataFrame:
    """
    Generates a realistic historical dataset for flight delay prediction.
    Features:
      - flight_number: e.g. AA1024, DL452
      - airline: e.g. Delta Air Lines, American Airlines
      - source: Origin airport code (e.g. JFK)
      - destination: Destination airport code (e.g. LAX)
      - departure_time: Scheduled departure timestamp (HH:MM)
      - departure_hour: Scheduled departure hour (0-23)
      - arrival_time: Scheduled arrival timestamp (HH:MM)
      - distance_miles: Flight distance between source and destination
      - temperature_c: Ambient temperature in Celsius (-10 to 40)
      - wind_speed_mph: Surface wind speed in mph (0 to 60)
      - visibility_miles: Visibility in miles (0.5 to 10.0)
      - rainfall_mm: Precipitation in mm (0 to 65)
      - weather_condition: Clear, Cloudy, Fog, Rain, Thunderstorm, Snow
      - delay_duration_mins: Actual departure delay in minutes (0 if on-time)
      - delay_reason: None, Weather, Carrier, Late Aircraft, Air Traffic (NAS)
      - is_delayed: Binary label (0 = On Time [<=15 mins], 1 = Delayed [>15 mins])
      - data_source_type: Explicit tag distinguishing Sample from Real data
    """
    airline_names = list(AIRLINES.keys())
    airport_codes = list(AIRPORTS.keys())
    
    # Distance lookup approximations between airports
    base_distances = {
        ("JFK", "LAX"): 2475, ("LAX", "JFK"): 2475,
        ("ORD", "DFW"): 802,  ("DFW", "ORD"): 802,
        ("ATL", "MIA"): 594,  ("MIA", "ATL"): 594,
        ("DEN", "SFO"): 967,  ("SFO", "DEN"): 967,
        ("SEA", "LAX"): 954,  ("LAX", "SEA"): 954,
        ("BOS", "ORD"): 867,  ("ORD", "BOS"): 867,
        ("ATL", "JFK"): 760,  ("JFK", "ATL"): 760,
        ("DFW", "LAX"): 1235, ("LAX", "DFW"): 1235,
        ("SFO", "SEA"): 679,  ("SEA", "SFO"): 679,
        ("DEN", "ORD"): 888,  ("ORD", "DEN"): 888,
    }

    records = []
    start_date = datetime(2025, 1, 1)

    for i in range(num_records):
        airline = random.choice(airline_names)
        code = AIRLINES[airline]["code"]
        fl_num = f"{code}{random.randint(100, 9999)}"
        
        # Pick source and destination (ensure distinct)
        src = random.choice(airport_codes)
        dst = random.choice([a for a in airport_codes if a != src])
        
        # Approximate distance
        dist = base_distances.get((src, dst), random.randint(450, 2600))
        
        # Scheduled time
        dep_hour = random.choices(
            population=list(range(24)),
            weights=[1, 1, 1, 1, 2, 5, 8, 9, 8, 7, 7, 8, 8, 7, 8, 9, 9, 8, 7, 6, 5, 4, 2, 1],
            k=1
        )[0]
        dep_minute = random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
        
        flight_date = start_date + timedelta(days=random.randint(0, 360))
        sched_dep = flight_date.replace(hour=dep_hour, minute=dep_minute)
        
        # Flight duration roughly 1 hour per 500 miles + 35 mins taxi/climb
        duration_mins = int((dist / 500.0) * 60 + 35)
        sched_arr = sched_dep + timedelta(minutes=duration_mins)
        
        # Weather generation
        month = sched_dep.month
        # Winter months colder and more snow
        if month in [12, 1, 2]:
            weather = random.choices(
                WEATHER_CONDITIONS,
                weights=[30, 25, 10, 10, 5, 5, 15],
                k=1
            )[0]
            temp = round(random.uniform(-8.0, 15.0), 1)
        elif month in [6, 7, 8]:
            weather = random.choices(
                WEATHER_CONDITIONS,
                weights=[45, 30, 3, 10, 8, 4, 0],
                k=1
            )[0]
            temp = round(random.uniform(18.0, 38.0), 1)
        else:
            weather = random.choices(
                WEATHER_CONDITIONS,
                weights=[40, 30, 8, 12, 5, 4, 1],
                k=1
            )[0]
            temp = round(random.uniform(5.0, 26.0), 1)

        # Weather impacts
        if weather in ["Thunderstorm", "Snow"]:
            wind_speed = round(random.uniform(18.0, 48.0), 1)
            visibility = round(random.uniform(0.5, 3.5), 1)
            rainfall = round(random.uniform(15.0, 55.0), 1) if weather == "Thunderstorm" else 0.0
        elif weather in ["Heavy Rain", "Fog"]:
            wind_speed = round(random.uniform(10.0, 32.0), 1)
            visibility = round(random.uniform(1.0, 4.0), 1)
            rainfall = round(random.uniform(10.0, 35.0), 1) if weather == "Heavy Rain" else round(random.uniform(0.0, 2.0), 1)
        elif weather == "Light Rain":
            wind_speed = round(random.uniform(8.0, 22.0), 1)
            visibility = round(random.uniform(3.5, 7.5), 1)
            rainfall = round(random.uniform(2.0, 10.0), 1)
        else:  # Clear or Partly Cloudy
            wind_speed = round(random.uniform(3.0, 18.0), 1)
            visibility = round(random.uniform(7.0, 10.0), 1)
            rainfall = 0.0

        # Delay probability calculation based on operational reality
        # Baseline delay chance: ~18%
        delay_score = 0.18
        
        # Peak congestion hours (16:00 - 20:00)
        if 16 <= dep_hour <= 20:
            delay_score += 0.16
        elif 7 <= dep_hour <= 9:
            delay_score += 0.08
        elif dep_hour >= 21:
            delay_score += 0.12  # Cumulative cascading delay

        # Weather contribution
        if weather == "Thunderstorm":
            delay_score += 0.45
        elif weather == "Snow":
            delay_score += 0.40
        elif weather == "Heavy Rain":
            delay_score += 0.28
        elif weather == "Fog":
            delay_score += 0.22
        elif weather == "Light Rain":
            delay_score += 0.09
            
        if wind_speed > 25:
            delay_score += 0.18
        if visibility < 3.0:
            delay_score += 0.15
            
        # Specific airline slight variance
        if airline in ["Spirit Airlines", "JetBlue Airways"]:
            delay_score += 0.06
        elif airline in ["Delta Air Lines", "Alaska Airlines"]:
            delay_score -= 0.04

        delay_prob = min(max(delay_score, 0.05), 0.95)
        is_delayed = 1 if random.random() < delay_prob else 0

        if is_delayed == 1:
            # Delay duration > 15 mins (FAA standard delay definition)
            if weather in ["Thunderstorm", "Snow", "Heavy Rain"]:
                delay_mins = int(random.triangular(25, 180, 65))
                reason = "Weather"
            elif random.random() < 0.40:
                delay_mins = int(random.triangular(16, 120, 35))
                reason = "Late Aircraft"
            elif random.random() < 0.35:
                delay_mins = int(random.triangular(16, 90, 30))
                reason = "Carrier"
            else:
                delay_mins = int(random.triangular(16, 80, 25))
                reason = "Air Traffic (NAS)"
        else:
            delay_mins = int(random.randint(-15, 14)) # early arrivals up to -15 mins
            reason = "None"

        actual_arr = sched_arr + timedelta(minutes=max(delay_mins, 0))

        records.append({
            "flight_number": fl_num,
            "airline": airline,
            "source": src,
            "destination": dst,
            "flight_date": sched_dep.strftime("%Y-%m-%d"),
            "departure_time": sched_dep.strftime("%H:%M"),
            "departure_hour": dep_hour,
            "arrival_time": sched_arr.strftime("%H:%M"),
            "actual_arrival_time": actual_arr.strftime("%H:%M"),
            "distance_miles": dist,
            "temperature_c": temp,
            "wind_speed_mph": wind_speed,
            "visibility_miles": visibility,
            "rainfall_mm": rainfall,
            "weather_condition": weather,
            "delay_duration_mins": max(delay_mins, 0),
            "delay_reason": reason,
            "is_delayed": is_delayed,
            "data_source_type": "Simulated Big Data Historical Flight Operations Archive (Sample Dataset for College Project)"
        })

    df = pd.DataFrame(records)
    return df

def main():
    print("Generating simulated historical flight dataset...")
    df = generate_flight_dataset(num_records=6500)
    df.to_csv(DATASET_CSV, index=False)
    print(f"Successfully generated {len(df)} records at: {DATASET_CSV}")
    print("Dataset Summary:")
    print(f"Total Flights: {len(df)}")
    print(f"Delayed Flights: {df['is_delayed'].sum()} ({df['is_delayed'].mean() * 100:.1f}%)")
    print(f"On-Time Flights: {(1 - df['is_delayed']).sum()} ({(1 - df['is_delayed']).mean() * 100:.1f}%)")
    print("\nDelay Reason Breakdown:")
    print(df['delay_reason'].value_counts())

if __name__ == "__main__":
    main()
