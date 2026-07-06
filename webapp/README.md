# Flight Delay Predictor — Web UI

An interactive web app for the Flight Delay Prediction project. Describe a
flight (route, airline, schedule, weather) and three trained models estimate:

- **Will it be delayed?** — RandomForest binary classifier
- **How severe?** — RandomForest multi-class classifier (No / Short / Moderate / Long delay)
- **How many minutes?** — RandomForest regressor

The page also shows each model's held-out performance and the top features that
drive the binary prediction.

## How it works

| File | Purpose |
|------|---------|
| `pipeline_utils.py` | Shared data loading + feature engineering (matches the Phase 1/2 notebooks) |
| `train_models.py`   | Trains the 3 models and writes `models/*.pkl`, `options.json`, `metrics.json` |
| `app.py`            | Standard-library HTTP server: serves the UI and the `/meta` + `/predict` endpoints |

The web server uses only the Python standard library (`http.server`) — no Flask
or FastAPI required. It needs the same scientific stack as the notebooks.

## Run it

From this `webapp/` folder:

```bash
# 1. Install dependencies (already present if you ran the notebooks)
pip install -r requirements.txt

# 2. Train the models once — creates the models/ folder (~150 MB, git-ignored)
python train_models.py

# 3. Start the app, then open the printed URL
python app.py            # http://127.0.0.1:8000
# python app.py 9000     # to use a different port
```

`train_models.py` reads `../cleaned_final_train.csv`, so keep this folder next to
the dataset.

## API

- `GET /` — the web UI
- `GET /meta` — dropdown options + model metrics (JSON)
- `POST /predict` — body: `{departure_iata, arrival_iata, airline_iata, year, month, day, hour, temp_avg, dew_avg, humidity_avg, wind_avg, pressure_avg}` (all optional; missing values fall back to dataset medians). Returns delay probability, severity class + probabilities, and predicted minutes.

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"departure_iata":"khi","arrival_iata":"ruh","airline_iata":"sv","month":12,"day":25,"hour":18,"temp_avg":60,"wind_avg":25,"humidity_avg":80}'
```
