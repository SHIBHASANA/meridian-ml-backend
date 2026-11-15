"""Meridian ML core: feature spec, SYNTHETIC data generator, pipeline, explanations.
All training data here is synthetic and generated from illustrative reference ranges.
It is NOT real astronaut data and NOT clinical data."""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["hr", "spo2", "temp", "sys", "rr", "sleep", "stress", "hyd"]
LABELS = {"hr": "Heart rate", "spo2": "Blood oxygen", "temp": "Core temperature",
          "sys": "Systolic pressure", "rr": "Respiration", "sleep": "Sleep (recent avg)",
          "stress": "Stress (recent avg)", "hyd": "Hydration index"}
# plausibility bounds for input validation (reject impossible values)
BOUNDS = {"hr": (20, 240), "spo2": (50, 100), "temp": (30, 43), "sys": (50, 260),
          "rr": (4, 60), "sleep": (0, 16), "stress": (1, 10), "hyd": (0, 100)}
# illustrative "typical healthy adult" mean/sd used ONLY to generate synthetic data
NORMAL = {"hr": (68, 8), "spo2": (97.5, .8), "temp": (36.8, .25), "sys": (117, 9),
          "rr": (15, 1.8), "sleep": (7.0, .7), "stress": (4.0, 1.3), "hyd": (80, 7)}

def _clip(df):
    for i, f in enumerate(FEATURES):
        df[:, i] = np.clip(df[:, i], *BOUNDS[f])
    return df

def make_normal(n, rng):
    X = np.column_stack([rng.normal(*NORMAL[f], n) for f in FEATURES])
    return _clip(X)

def make_anomalies(n, rng, kind):
    """Synthetic anomaly patterns defined by the developer (not real events)."""
    X = make_normal(n, rng); c = {f: i for i, f in enumerate(FEATURES)}
    def put(f, m, s): X[:, c[f]] = rng.normal(m, s, n)
    if kind == "cardiac_like":   put("hr", 135, 12); put("sys", 150, 10); put("rr", 23, 2); put("spo2", 95.5, 1)
    elif kind == "fever_dehydration_like": put("temp", 38.7, .3); put("hyd", 46, 5); put("hr", 95, 8)
    elif kind == "hypoxia_like": put("spo2", 91, 1.5); put("rr", 26, 3); put("hr", 100, 10)
    elif kind == "sleep_stress_like": put("sleep", 4.8, .5); put("stress", 8, .8); put("hr", 76, 8)
    elif kind == "subtle_drift_like": put("hr", 94, 3); put("rr", 19, 1); put("spo2", 96.2, .4)  # each value still near normal range
    else: raise ValueError(kind)
    return _clip(X)

KINDS = ["cardiac_like", "fever_dehydration_like", "hypoxia_like", "sleep_stress_like", "subtle_drift_like"]

def build_pipeline(seed=42):
    return Pipeline([("scale", StandardScaler()),
                     ("iso", IsolationForest(n_estimators=300, contamination="auto", random_state=seed))])

def vec(feats):
    return np.array([[float(feats[f]) for f in FEATURES]])

def classify(dec, art):
    if dec < art["thr_significant"]: return 2
    if dec < art["thr_unusual"]: return 1
    return 0

def analyze(art, feats):
    pipe, x = art["pipeline"], vec(feats)
    dec = float(pipe.decision_function(x)[0])
    cls = classify(dec, art)
    pct = float(np.mean(art["val_scores"] <= dec) * 100)
    contrib = []
    for i, f in enumerate(FEATURES):   # model-based: how much would the score improve if this feature were typical?
        y = x.copy(); y[0, i] = art["median"][i]
        gain = float(pipe.decision_function(y)[0]) - dec
        z = (x[0, i] - art["mean"][i]) / art["std"][i]
        contrib.append({"feature": f, "label": LABELS[f], "value": round(float(x[0, i]), 2),
                        "training_mean": round(float(art["mean"][i]), 2), "z": round(float(z), 2),
                        "direction": "high" if z > 0 else "low", "score_effect": round(gain, 4)})
    contrib.sort(key=lambda d: d["score_effect"], reverse=True)
    return {"classification": ["normal_pattern", "unusual_pattern", "significant_anomaly"][cls],
            "label": ["Normal pattern", "Unusual pattern", "Significant anomaly"][cls],
            "score": round(dec, 4), "normality_percentile": round(pct, 1),
            "contributors": [c for c in contrib if c["score_effect"] > 0.002][:3],
            "thresholds": {"unusual_below": round(art["thr_unusual"], 4), "significant_below": round(art["thr_significant"], 4)}}
