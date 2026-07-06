"""Web app for the Flight Delay Prediction project (standard-library server).

Serves a single-page UI where a user describes a flight (route, airline,
schedule, weather) and gets three predictions from the trained models:
    - Will it be delayed?         (binary classifier)
    - How severe?                 (multi-class classifier)
    - How many minutes?           (regressor)

Run:
    python train_models.py     # once, to create webapp/models/*
    python app.py               # then open http://127.0.0.1:8000

Uses only the Python standard library for serving (http.server), so it needs
no web framework beyond scikit-learn / pandas / joblib.
"""
import os
import sys
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import joblib

import pipeline_utils as pu

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")


def _load(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        raise RuntimeError(
            f"Missing {name}. Run `python train_models.py` first to create the models."
        )
    return joblib.load(path)


BINARY = _load("binary_model.pkl")
MULTI = _load("multiclass_model.pkl")
REG = _load("regression_model.pkl")
with open(os.path.join(MODEL_DIR, "options.json"), encoding="utf-8") as f:
    OPTIONS = json.load(f)
with open(os.path.join(MODEL_DIR, "metrics.json"), encoding="utf-8") as f:
    METRICS = json.load(f)

# Accepted request fields and their defaults (also caps what a client can set).
DEFAULTS = {
    "departure_iata": None, "arrival_iata": None, "airline_iata": None,
    "year": 2023, "month": 7, "day": 15, "hour": 12,
    "temp_avg": None, "dew_avg": None, "humidity_avg": None,
    "wind_avg": None, "pressure_avg": None,
}


def run_prediction(payload):
    req = dict(DEFAULTS)
    req.update({k: v for k, v in (payload or {}).items() if k in DEFAULTS})
    row = pu.build_feature_row(req, OPTIONS)

    p_delay = float(BINARY.predict_proba(row)[0][1])
    classes = MULTI.named_steps["clf"].classes_
    probs = MULTI.predict_proba(row)[0]
    sev_proba = {str(c): round(float(p), 3) for c, p in zip(classes, probs)}
    severity = str(MULTI.predict(row)[0])
    minutes = max(0.0, float(REG.predict(row)[0]))

    return {
        "delay_probability": round(p_delay, 3),
        "is_delayed": bool(p_delay >= 0.5),
        "severity": severity,
        "severity_probabilities": sev_proba,
        "predicted_minutes": round(minutes, 1),
        "echo": req,
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            self._send(200, HTML_PAGE, "text/html; charset=utf-8")
        elif self.path == "/meta":
            self._send(200, json.dumps({"options": OPTIONS, "metrics": METRICS}),
                       "application/json")
        else:
            self._send(404, json.dumps({"error": "not found"}), "application/json")

    def do_POST(self):
        if self.path != "/predict":
            self._send(404, json.dumps({"error": "not found"}), "application/json")
            return
        try:
            n = int(self.headers.get("Content-Length", 0) or 0)
            payload = json.loads(self.rfile.read(n) or b"{}")
            self._send(200, json.dumps(run_prediction(payload)), "application/json")
        except Exception as e:  # surface prediction/parse errors to the client
            self._send(400, json.dumps({"error": str(e)}), "application/json")

    def log_message(self, *args):
        pass  # keep the console quiet


def main(host="127.0.0.1", port=8000):
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"Flight Delay Predictor running at http://{host}:{port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        srv.shutdown()


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Flight Delay Predictor</title>
<style>
  :root{
    --bg:#0b1220; --card:#131c31; --card2:#0f1728; --line:#243354;
    --text:#e8eefc; --muted:#93a4c8; --accent:#4f8cff; --accent2:#22d3ee;
    --good:#22c55e; --warn:#f59e0b; --bad:#ef4444; --radius:14px;
  }
  *{box-sizing:border-box}
  body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
       background:radial-gradient(1000px 500px at 85% -20%,#1b2b4d,transparent 60%),var(--bg);color:var(--text)}
  .wrap{max-width:1080px;margin:0 auto;padding:28px 20px 60px}
  header h1{margin:0 0 4px;font-size:26px;letter-spacing:.2px}
  header p{margin:0;color:var(--muted)}
  .grid{display:grid;grid-template-columns:1.15fr .85fr;gap:20px;margin-top:22px}
  @media(max-width:860px){.grid{grid-template-columns:1fr}}
  .card{background:linear-gradient(180deg,var(--card),var(--card2));border:1px solid var(--line);
        border-radius:var(--radius);padding:20px}
  .card h2{margin:0 0 14px;font-size:16px;color:#cfe0ff}
  label{display:block;font-size:12px;color:var(--muted);margin:12px 0 6px}
  select,input[type=number],input[type=date]{width:100%;padding:10px 12px;border-radius:10px;
        border:1px solid var(--line);background:#0c1424;color:var(--text);font-size:14px}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  .row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
  .slider{display:flex;align-items:center;gap:10px}
  input[type=range]{flex:1;accent-color:var(--accent)}
  .sval{min-width:52px;text-align:right;font-variant-numeric:tabular-nums;color:#cfe0ff}
  button{margin-top:18px;width:100%;padding:13px;border:0;border-radius:11px;cursor:pointer;
        background:linear-gradient(90deg,var(--accent),var(--accent2));color:#04122b;font-weight:700;font-size:15px}
  button:active{transform:translateY(1px)}
  .result{display:none}
  .verdict{display:flex;align-items:center;gap:14px;padding:16px;border-radius:12px;margin-bottom:14px;
          border:1px solid var(--line);background:#0c1424}
  .dot{width:14px;height:14px;border-radius:50%}
  .verdict b{font-size:18px}
  .metrics{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:6px}
  .metric{background:#0c1424;border:1px solid var(--line);border-radius:10px;padding:12px;text-align:center}
  .metric .v{font-size:20px;font-weight:700}
  .metric .k{font-size:11px;color:var(--muted);margin-top:2px}
  .bars{margin-top:8px}
  .bar{display:flex;align-items:center;gap:10px;margin:7px 0;font-size:13px}
  .bar .name{width:150px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .bar .track{flex:1;height:9px;background:#0c1424;border-radius:6px;overflow:hidden;border:1px solid var(--line)}
  .bar .fill{height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2))}
  .bar .pct{width:44px;text-align:right;color:#cfe0ff;font-variant-numeric:tabular-nums}
  .muted{color:var(--muted);font-size:12px}
  .pill{display:inline-block;padding:3px 9px;border-radius:999px;font-size:12px;border:1px solid var(--line);
        background:#0c1424;color:var(--muted);margin-left:8px}
  .foot{margin-top:24px;color:var(--muted);font-size:12px;text-align:center}
  .spin{display:inline-block;width:16px;height:16px;border:2px solid #04122b;border-top-color:transparent;
        border-radius:50%;animation:s .7s linear infinite;vertical-align:-3px}
  @keyframes s{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>✈️ Flight Delay Predictor</h1>
    <p>Describe a departure and the trained models estimate whether it will be delayed, how badly, and by how many minutes.</p>
  </header>

  <div class="grid">
    <!-- INPUT -->
    <div class="card">
      <h2>Flight & conditions</h2>
      <div class="row">
        <div><label>Departure airport</label><select id="departure_iata"></select></div>
        <div><label>Arrival airport</label><select id="arrival_iata"></select></div>
      </div>
      <div class="row">
        <div><label>Airline</label><select id="airline_iata"></select></div>
        <div><label>Departure date</label><input type="date" id="date" value="2023-07-15"/></div>
      </div>
      <label>Departure hour: <span id="hourlab">12:00</span></label>
      <div class="slider"><input type="range" id="hour" min="0" max="23" value="12"/><span class="sval" id="hourval">12</span></div>

      <h2 style="margin-top:20px">Weather</h2>
      <div id="weather"></div>

      <button id="go" onclick="predict()">Predict delay</button>
      <p class="muted" id="hint" style="margin-top:10px"></p>
    </div>

    <!-- OUTPUT -->
    <div class="card">
      <h2>Prediction</h2>
      <div id="empty" class="muted">Fill in the flight details and hit <b>Predict delay</b>.</div>
      <div id="result" class="result">
        <div class="verdict">
          <span class="dot" id="vdot"></span>
          <div>
            <div><b id="vtext">—</b><span class="pill" id="vsev"></span></div>
            <div class="muted" id="vprob"></div>
          </div>
        </div>
        <div class="metrics">
          <div class="metric"><div class="v" id="rmin">—</div><div class="k">predicted minutes</div></div>
          <div class="metric"><div class="v" id="rprob">—</div><div class="k">delay probability</div></div>
          <div class="metric"><div class="v" id="rsev">—</div><div class="k">severity class</div></div>
        </div>
        <div class="bars" id="sevbars"></div>
      </div>
    </div>
  </div>

  <!-- MODEL CARD -->
  <div class="card" style="margin-top:20px">
    <h2>About the model <span class="muted" id="nrows"></span></h2>
    <div class="metrics" id="modelmetrics"></div>
    <h2 style="margin:18px 0 6px">What drives the prediction? <span class="muted">(top features, binary model)</span></h2>
    <div class="bars" id="importances"></div>
  </div>

  <div class="foot">RandomForest models trained on <span id="rowsfoot">the</span> cleaned flight + weather dataset · Phase 1 & 2 pipeline.</div>
</div>

<script>
let OPT=null, MET=null;

function opt(sel, values, def){
  const el=document.getElementById(sel);
  el.innerHTML="";
  values.forEach(v=>{const o=document.createElement("option");o.value=v;o.textContent=String(v).toUpperCase();el.appendChild(o);});
  if(def!==undefined) el.value=def;
}

async function boot(){
  const r=await fetch("/meta"); const j=await r.json();
  OPT=j.options; MET=j.metrics;

  opt("departure_iata", OPT.departure_airports);
  opt("arrival_iata", OPT.arrival_airports, OPT.arrival_airports[1]||OPT.arrival_airports[0]);
  opt("airline_iata", OPT.airlines);

  // weather sliders
  const labels={"Temperature (F)":["temp_avg","°F"],"Dew Point (F)":["dew_avg","°F"],
                "Humidity (%)":["humidity_avg","%"],"Wind Speed (mph)":["wind_avg","mph"],
                "Pressure (in)":["pressure_avg","in"]};
  const box=document.getElementById("weather");
  Object.keys(labels).forEach(metric=>{
    const [key,unit]=labels[metric]; const s=OPT.weather[metric];
    const wrap=document.createElement("div");
    wrap.innerHTML=`<label>${metric.replace(" (F)","").replace(" (in)","")} <span class="muted">(${unit})</span></label>
      <div class="slider"><input type="range" id="${key}" min="${s.min}" max="${s.max}" step="0.1" value="${s.avg_median}"/>
      <span class="sval" id="${key}v">${s.avg_median}</span></div>`;
    box.appendChild(wrap);
    wrap.querySelector("input").addEventListener("input",e=>{document.getElementById(key+"v").textContent=(+e.target.value).toFixed(1);});
  });

  document.getElementById("hour").addEventListener("input",e=>{
    document.getElementById("hourval").textContent=e.target.value;
    document.getElementById("hourlab").textContent=String(e.target.value).padStart(2,"0")+":00";
  });

  // model metrics
  const m=MET; const mm=document.getElementById("modelmetrics");
  mm.innerHTML=`
    <div class="metric"><div class="v">${(m.binary.accuracy*100).toFixed(1)}%</div><div class="k">Binary accuracy</div></div>
    <div class="metric"><div class="v">${m.binary.f1}</div><div class="k">Binary F1</div></div>
    <div class="metric"><div class="v">${(m.multiclass.accuracy*100).toFixed(1)}%</div><div class="k">Multi-class accuracy</div></div>
    <div class="metric"><div class="v">${m.regression.mae}</div><div class="k">Regression MAE (min)</div></div>
    <div class="metric"><div class="v">${m.regression.rmse}</div><div class="k">Regression RMSE (min)</div></div>
    <div class="metric"><div class="v">${m.regression.r2}</div><div class="k">Regression R²</div></div>`;
  document.getElementById("nrows").textContent="· trained on "+m.n_rows.toLocaleString()+" flights";
  document.getElementById("rowsfoot").textContent=m.n_rows.toLocaleString();

  // importances
  const imp=m.binary.importances.slice(0,10);
  const max=Math.max(...imp.map(d=>d.importance));
  document.getElementById("importances").innerHTML=imp.map(d=>`
    <div class="bar"><div class="name">${d.feature}</div>
    <div class="track"><div class="fill" style="width:${(d.importance/max*100).toFixed(0)}%"></div></div>
    <div class="pct">${(d.importance*100).toFixed(1)}%</div></div>`).join("");
}

async function predict(){
  const btn=document.getElementById("go"); btn.disabled=true;
  btn.innerHTML='<span class="spin"></span> Predicting…';
  const d=document.getElementById("date").value.split("-");
  const body={
    departure_iata:document.getElementById("departure_iata").value,
    arrival_iata:document.getElementById("arrival_iata").value,
    airline_iata:document.getElementById("airline_iata").value,
    year:+d[0], month:+d[1], day:+d[2], hour:+document.getElementById("hour").value,
    temp_avg:+document.getElementById("temp_avg").value,
    dew_avg:+document.getElementById("dew_avg").value,
    humidity_avg:+document.getElementById("humidity_avg").value,
    wind_avg:+document.getElementById("wind_avg").value,
    pressure_avg:+document.getElementById("pressure_avg").value,
  };
  try{
    const r=await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    const j=await r.json();
    render(j);
  }catch(e){ document.getElementById("hint").textContent="Error: "+e; }
  btn.disabled=false; btn.textContent="Predict delay";
}

function render(j){
  document.getElementById("empty").style.display="none";
  document.getElementById("result").style.display="block";
  const delayed=j.is_delayed;
  const color=delayed?(j.predicted_minutes>45?"var(--bad)":"var(--warn)"):"var(--good)";
  document.getElementById("vdot").style.background=color;
  document.getElementById("vtext").textContent=delayed?"Likely DELAYED":"Likely ON-TIME";
  document.getElementById("vsev").textContent=j.severity;
  document.getElementById("vprob").textContent=`Model is ${(Math.max(j.delay_probability,1-j.delay_probability)*100).toFixed(0)}% confident`;
  document.getElementById("rmin").textContent=j.predicted_minutes+" min";
  document.getElementById("rprob").textContent=(j.delay_probability*100).toFixed(0)+"%";
  document.getElementById("rsev").textContent=j.severity.replace(" Delay","");
  const sp=j.severity_probabilities||{}; const keys=Object.keys(sp);
  const max=Math.max(0.0001,...keys.map(k=>sp[k]));
  document.getElementById("sevbars").innerHTML=keys.map(k=>`
    <div class="bar"><div class="name">${k}</div>
    <div class="track"><div class="fill" style="width:${(sp[k]/max*100).toFixed(0)}%"></div></div>
    <div class="pct">${(sp[k]*100).toFixed(0)}%</div></div>`).join("");
}

boot();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    main(port=port)
