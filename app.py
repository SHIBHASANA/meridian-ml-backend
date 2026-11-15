"""Meridian ML API (Flask). Run locally: python app.py   Production: gunicorn app:app"""
import os, math, datetime, joblib
from flask import Flask, request, jsonify
from model import FEATURES, BOUNDS, analyze

app = Flask(__name__)
PATH = os.path.join(os.path.dirname(__file__), "models", "meridian_iforest.joblib")
ART = joblib.load(PATH) if os.path.exists(PATH) else None
ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")]
DISCLAIMER = ("Research and decision-support demonstration. Model trained on synthetic data. The score is an unusualness "
              "measure, not a diagnosis or a medical risk probability.")

@app.after_request
def cors(r):
    o = request.headers.get("Origin", "")
    if "*" in ORIGINS or o in ORIGINS:
        r.headers["Access-Control-Allow-Origin"] = "*" if "*" in ORIGINS else o
        r.headers["Access-Control-Allow-Headers"] = "Content-Type"
        r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return r

@app.route("/api/health")
def health():
    return jsonify(status="ok" if ART else "model_missing", model_loaded=bool(ART),
                   version=ART["version"] if ART else None, trained_at=ART["trained_at"] if ART else None,
                   data=ART["data"] if ART else None, features=FEATURES)

@app.route("/api/analyze", methods=["POST", "OPTIONS"])
def analyze_route():
    if request.method == "OPTIONS": return ("", 204)
    if not ART: return jsonify(error="model not loaded; run train_model.py"), 503
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get("features"), dict):
        return jsonify(error="body must be JSON with a 'features' object"), 400
    f, errs = body["features"], []
    for k in FEATURES:
        v = f.get(k)
        if v is None: errs.append(f"{k}: missing"); continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v): errs.append(f"{k}: must be a finite number"); continue
        lo, hi = BOUNDS[k]
        if not lo <= v <= hi: errs.append(f"{k}: {v} outside plausible range {lo} to {hi}")
    if errs: return jsonify(error="invalid input", errors=errs), 422
    out = analyze(ART, f)
    out.update(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(), crew=str(body.get("crew", ""))[:60],
               model={"version": ART["version"], "trained_on": "synthetic data"}, disclaimer=DISCLAIMER)
    return jsonify(out)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
