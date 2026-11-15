"""Train and evaluate the Isolation Forest on SYNTHETIC data. Run: python train_model.py"""
import json, datetime, numpy as np, joblib
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from model import *

rng = np.random.default_rng(42)
Xtr = make_normal(6000, rng)                      # training: normal patterns only (unsupervised)
Xval = make_normal(2000, rng)                     # validation normals: used to set thresholds
Xte_n = make_normal(2000, rng)                    # held-out test normals
Xte_a = {k: make_anomalies(500, rng, k) for k in KINDS}   # held-out synthetic anomalies

pipe = build_pipeline().fit(Xtr)
val = pipe.decision_function(Xval)
art = {"pipeline": pipe, "thr_unusual": float(np.percentile(val, 1.0)),      # ~1% false-positive rate on validation normals
       "thr_significant": float(np.percentile(val, 0.1)),                     # ~0.1%
       "val_scores": np.sort(val), "mean": Xtr.mean(0), "std": Xtr.std(0), "median": np.median(Xtr, 0),
       "features": FEATURES, "version": "1.0.0", "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "data": "synthetic (generated from illustrative reference ranges; not real astronaut or clinical data)"}

def evaluate(thr, kinds):
    Xa = np.vstack([Xte_a[k] for k in kinds]); X = np.vstack([Xte_n, Xa])
    y = np.r_[np.zeros(len(Xte_n)), np.ones(len(Xa))]
    d = pipe.decision_function(X); p = (d < thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p).ravel()
    return {"precision": round(precision_score(y, p), 3), "recall": round(recall_score(y, p), 3), "f1": round(f1_score(y, p), 3),
            "false_positive_rate": round(fp / (fp + tn), 4), "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "roc_auc": round(roc_auc_score(y, -d), 3)}

main = [k for k in KINDS if k != "subtle_drift_like"]
m = {"n_train_normal": len(Xtr), "n_test_normal": len(Xte_n), "n_test_anomaly_per_kind": 500,
     "unusual_threshold": evaluate(art["thr_unusual"], main), "significant_threshold": evaluate(art["thr_significant"], main),
     "per_kind_recall_at_unusual": {k: round(float((pipe.decision_function(Xte_a[k]) < art["thr_unusual"]).mean()), 3) for k in KINDS},
     "note": "Anomaly test cases are synthetic patterns written by the developer, so these figures are optimistic and say nothing about performance on real crews. subtle_drift_like is reported separately and excluded from the headline metrics."}
joblib.dump(art, "models/meridian_iforest.joblib"); json.dump(m, open("models/metrics.json", "w"), indent=2)
print(json.dumps(m, indent=2))
