"""Train the delay-prediction models and export everything the web app needs.

Outputs (written to webapp/models/):
    binary_model.pkl        RandomForest on-time-vs-delayed classifier
    multiclass_model.pkl    RandomForest delay-severity classifier
    regression_model.pkl    RandomForest exact-delay regressor
    options.json            dropdown values, slider ranges, airport delay lookup
    metrics.json            held-out metrics + feature importances

Run once before starting the app:  python train_models.py
"""
import os
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             mean_absolute_error, mean_squared_error, r2_score)

import pipeline_utils as pu

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def grouped_importances(pipe):
    """Aggregate one-hot feature importances back to the original columns."""
    pre = pipe.named_steps["pre"]
    rf = pipe.named_steps["clf"]
    names = pre.get_feature_names_out()
    imp = rf.feature_importances_
    groups = {}
    for name, val in zip(names, imp):
        # names look like "num__Weather Severity" or "cat__Airline_IATA_sv"
        body = name.split("__", 1)[1]
        base = body
        for col in pu.CATEGORICAL_FEATURES:
            if body.startswith(col + "_"):
                base = col
                break
        groups[base] = groups.get(base, 0.0) + float(val)
    top = sorted(groups.items(), key=lambda kv: kv[1], reverse=True)
    return [{"feature": k, "importance": round(v, 4)} for k, v in top]


def main():
    print("Loading data ...")
    df = pu.add_features(pu.load_and_prepare())
    for c in pu.NUMERIC_FEATURES:
        df[c] = df[c].fillna(df[c].median())
    for c in pu.CATEGORICAL_FEATURES:
        df[c] = df[c].fillna("unknown")

    X = df[pu.ALL_FEATURES]
    metrics = {"n_rows": int(len(df))}

    # ---- Binary classifier -------------------------------------------------
    print("Training binary classifier ...")
    y = (df["Delay (minutes)"] > 0).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    binary = Pipeline([("pre", pu.make_preprocessor()),
                       ("clf", RandomForestClassifier(n_estimators=100, max_depth=20,
                                                      random_state=42, n_jobs=-1))]).fit(Xtr, ytr)
    yp = binary.predict(Xte)
    metrics["binary"] = {
        "accuracy": round(accuracy_score(yte, yp), 4),
        "precision": round(precision_score(yte, yp), 4),
        "recall": round(recall_score(yte, yp), 4),
        "f1": round(f1_score(yte, yp), 4),
        "importances": grouped_importances(binary),
    }
    joblib.dump(binary, os.path.join(MODEL_DIR, "binary_model.pkl"))

    # ---- Multi-class classifier -------------------------------------------
    print("Training multi-class classifier ...")
    bucket = pd.cut(df["Delay (minutes)"], bins=pu.DELAY_BINS,
                    labels=pu.DELAY_LABELS).astype(str)
    present = [c for c in pu.DELAY_LABELS if c in set(bucket)]
    Xtr, Xte, ytr, yte = train_test_split(X, bucket, test_size=0.2,
                                          stratify=bucket, random_state=42)
    multi = Pipeline([("pre", pu.make_preprocessor()),
                      ("clf", RandomForestClassifier(n_estimators=100, max_depth=20,
                                                     random_state=42, n_jobs=-1))]).fit(Xtr, ytr)
    yp = multi.predict(Xte)
    metrics["multiclass"] = {
        "accuracy": round(accuracy_score(yte, yp), 4),
        "macro_f1": round(f1_score(yte, yp, average="macro"), 4),
        "classes": present,
    }
    joblib.dump(multi, os.path.join(MODEL_DIR, "multiclass_model.pkl"))

    # ---- Regressor ---------------------------------------------------------
    print("Training regressor ...")
    yreg = df["Delay (minutes)"].astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, yreg, test_size=0.2, random_state=42)
    reg = Pipeline([("pre", pu.make_preprocessor()),
                    ("clf", RandomForestRegressor(n_estimators=60, max_depth=14,
                                                  random_state=42, n_jobs=-1))]).fit(Xtr, ytr)
    yp = reg.predict(Xte)
    metrics["regression"] = {
        "mae": round(mean_absolute_error(yte, yp), 3),
        "rmse": round(mean_squared_error(yte, yp) ** 0.5, 3),
        "r2": round(r2_score(yte, yp), 3),
    }
    joblib.dump(reg, os.path.join(MODEL_DIR, "regression_model.pkl"))

    # ---- Options for the UI -----------------------------------------------
    print("Building UI options ...")
    airport_avg = df.groupby("Departure_IATA")["Delay (minutes)"].mean().round(2).to_dict()

    def top_values(col, n=200):
        return sorted(v for v in df[col].dropna().unique().tolist())[:n]

    weather = {}
    for metric in pu.WEATHER_METRICS:
        avg = df[f"{metric}_Avg"]
        weather[metric] = {
            "min": round(float(df[f"{metric}_Min"].quantile(0.01)), 2),
            "max": round(float(df[f"{metric}_Max"].quantile(0.99)), 2),
            "avg_median": round(float(avg.median()), 2),
            "up_spread": round(float((df[f"{metric}_Max"] - avg).median()), 2),
            "down_spread": round(float((avg - df[f"{metric}_Min"]).median()), 2),
        }

    options = {
        "departure_airports": top_values("Departure_IATA"),
        "arrival_airports": top_values("Arrival_IATA"),
        "airlines": top_values("Airline_IATA"),
        "airport_avg_delay": {str(k): float(v) for k, v in airport_avg.items()},
        "global_avg_delay": round(float(df["Delay (minutes)"].mean()), 2),
        "weather": weather,
        "delay_share_ontime": round(float((df["Delay (minutes)"] == 0).mean()), 4),
        "delay_describe": {k: round(float(v), 2)
                           for k, v in df["Delay (minutes)"].describe().items()},
    }

    with open(os.path.join(MODEL_DIR, "options.json"), "w", encoding="utf-8") as f:
        json.dump(options, f, indent=2)
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved models + options + metrics to", MODEL_DIR)
    print("Binary:", metrics["binary"]["accuracy"], "acc |",
          "Multi:", metrics["multiclass"]["accuracy"], "acc |",
          "Reg R2:", metrics["regression"]["r2"])


if __name__ == "__main__":
    main()
