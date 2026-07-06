"""Shared data-loading and feature-engineering helpers.

Used by both `train_models.py` (offline training) and `app.py` (serving).
Mirrors the preprocessing done in the Phase 1 / Phase 2 notebooks so the
web app and the notebooks agree on features.
"""
import os
import re
import numpy as np
import pandas as pd

# cleaned_final_train.csv lives one level up, next to the notebooks.
HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "..", "cleaned_final_train.csv")

# The five weather metrics, each present as Max/Avg/Min in the dataset.
WEATHER_METRICS = [
    "Temperature (F)", "Dew Point (F)", "Humidity (%)",
    "Wind Speed (mph)", "Pressure (in)",
]

NUMERIC_FEATURES = [
    "Temperature (F)_Max", "Temperature (F)_Avg", "Temperature (F)_Min",
    "Dew Point (F)_Max", "Dew Point (F)_Avg", "Dew Point (F)_Min",
    "Humidity (%)_Max", "Humidity (%)_Avg", "Humidity (%)_Min",
    "Wind Speed (mph)_Max", "Wind Speed (mph)_Avg", "Wind Speed (mph)_Min",
    "Pressure (in)_Max", "Pressure (in)_Avg", "Pressure (in)_Min",
    "Hour_of_Day", "Week_Number", "Scheduled Hour", "Scheduled Weekday",
    "Scheduled Month", "Weather Severity", "IsWeekend", "PeakHour", "Season",
    "Avg Departure Delay (Airport)",
]
CATEGORICAL_FEATURES = [
    "Departure_IATA", "Arrival_IATA", "Airline_IATA", "Day_of_Week", "Month_of_Year",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Delay severity buckets (minutes).
DELAY_BINS = [-1, 0, 45, 175, float("inf")]
DELAY_LABELS = ["No Delay", "Short Delay", "Moderate Delay", "Long Delay"]


def assign_season(month):
    return {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
            6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}[int(month)]


def clean_columns(df):
    """Strip the mojibake degree sign from weather headers, collapse spaces."""
    df.columns = [re.sub(r"\s+", " ", re.sub(r"[^\x00-\x7F]", "", c)).strip()
                  for c in df.columns]
    return df


def load_and_prepare(path=CSV_PATH):
    df = pd.read_csv(path)
    df = clean_columns(df)
    df["Scheduled Time"] = pd.to_datetime(df["Departure_Scheduled"], errors="coerce")
    df = df.rename(columns={"Departure_Delay_Minutes": "Delay (minutes)"})
    df = df.dropna(subset=["Scheduled Time", "Delay (minutes)"]).reset_index(drop=True)
    return df


def add_features(df):
    df = df.copy()
    df["Weather Severity"] = (
        df["Wind Speed (mph)_Max"] * 0.4
        + (100 - df["Humidity (%)_Avg"]) * 0.3
        + (df["Temperature (F)_Max"] - df["Temperature (F)_Min"]) * 0.3
    )
    hour = df["Scheduled Time"].dt.hour
    df["IsWeekend"] = (df["Scheduled Time"].dt.weekday >= 5).astype(int)
    df["PeakHour"] = (hour.between(6, 9) | hour.between(17, 20)).astype(int)
    df["Season"] = df["Scheduled Time"].dt.month.map(assign_season)
    df["Scheduled Hour"] = hour
    df["Scheduled Weekday"] = df["Scheduled Time"].dt.weekday
    df["Scheduled Month"] = df["Scheduled Time"].dt.month
    df["Avg Departure Delay (Airport)"] = (
        df.groupby("Departure_IATA")["Delay (minutes)"].transform("mean")
    )
    return df


def make_preprocessor():
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def build_feature_row(payload, opts):
    """Turn a UI payload dict into a single-row DataFrame with ALL_FEATURES.

    payload keys (all optional except the categoricals fall back to defaults):
        departure_iata, arrival_iata, airline_iata  -> categorical codes
        year, month, day, hour                       -> scheduled datetime parts
        temp_avg, dew_avg, humidity_avg, wind_avg, pressure_avg -> weather sliders
    `opts` is the options dict produced by train_models.py (medians, spreads,
    airport delay lookup, weather metric key mapping).
    """
    row = {}

    # --- Weather: user supplies the "avg"; Max/Min derived via dataset spreads.
    slider_key = {
        "Temperature (F)": "temp_avg", "Dew Point (F)": "dew_avg",
        "Humidity (%)": "humidity_avg", "Wind Speed (mph)": "wind_avg",
        "Pressure (in)": "pressure_avg",
    }
    for metric in WEATHER_METRICS:
        stats = opts["weather"][metric]
        raw = payload.get(slider_key[metric])
        avg = float(raw) if raw is not None else float(stats["avg_median"])
        row[f"{metric}_Avg"] = avg
        row[f"{metric}_Max"] = avg + stats["up_spread"]
        row[f"{metric}_Min"] = avg - stats["down_spread"]

    row["Weather Severity"] = (
        row["Wind Speed (mph)_Max"] * 0.4
        + (100 - row["Humidity (%)_Avg"]) * 0.3
        + (row["Temperature (F)_Max"] - row["Temperature (F)_Min"]) * 0.3
    )

    # --- Temporal features from the chosen schedule (coalesce None -> default;
    #     hour is checked explicitly so a valid 0 is not treated as missing).
    def _int(key, default):
        v = payload.get(key)
        return int(v) if v is not None else default
    month = _int("month", 7)
    day = _int("day", 15)
    hour = _int("hour", 12)
    year = _int("year", 2023)
    ts = pd.Timestamp(year=year, month=month, day=day, hour=hour)
    row["Hour_of_Day"] = hour
    row["Scheduled Hour"] = hour
    row["Scheduled Weekday"] = ts.weekday()
    row["Scheduled Month"] = month
    row["Week_Number"] = int(ts.isocalendar().week)
    row["IsWeekend"] = int(ts.weekday() >= 5)
    row["PeakHour"] = int((6 <= hour <= 9) or (17 <= hour <= 20))
    row["Season"] = assign_season(month)

    # --- Categoricals (None or empty string -> first known value).
    dep = payload.get("departure_iata") or opts["departure_airports"][0]
    arr = payload.get("arrival_iata") or opts["arrival_airports"][0]
    airline = payload.get("airline_iata") or opts["airlines"][0]
    row["Departure_IATA"] = dep
    row["Arrival_IATA"] = arr
    row["Airline_IATA"] = airline
    row["Day_of_Week"] = ts.day_name()
    row["Month_of_Year"] = ts.month_name()

    # Airport reputation feature (historical average delay at that airport).
    row["Avg Departure Delay (Airport)"] = float(
        opts["airport_avg_delay"].get(dep, opts["global_avg_delay"])
    )

    return pd.DataFrame([row])[ALL_FEATURES]
