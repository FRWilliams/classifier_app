# app.py
from flask import Flask, request, jsonify
from pathlib import Path
import os
import pandas as pd
import numpy as np
import joblib

# IMPORTANT: ensure the symbol path used in the pickle exists
import preprocessing  # must be importable BEFORE joblib.load()

try:
    import cloudpickle as cp  # optional fallback for notebook pickles
except Exception:
    cp = None

print(">>> APP STARTED, loading model lazily…")

app = Flask(__name__)
try:
    app.json.ensure_ascii = False  # Flask 2.3+
except Exception:
    app.config["JSON_AS_ASCII"] = False

BASE_DIR = Path(__file__).resolve().parent
PIPE_PATH = BASE_DIR / "model" / "income_pipeline.pkl"
THRESHOLD_PATH = BASE_DIR / "model" / "threshold.txt"

pipeline = None
last_load_error = None

# Use file or env for threshold (defaults to 0.5)
def load_threshold():
    env_thr = os.getenv("THRESHOLD")
    if env_thr:
        try:
            return float(env_thr)
        except ValueError:
            pass
    if THRESHOLD_PATH.exists():
        try:
            return float(THRESHOLD_PATH.read_text().strip())
        except ValueError:
            pass
    return 0.5

DECISION_THRESHOLD = load_threshold()

def get_pipeline():
    """Lazy-load the trained pipeline, capture first error for /health."""
    global pipeline, last_load_error
    if pipeline is not None:
        return pipeline
    try:
        pipeline = joblib.load(PIPE_PATH)
        last_load_error = None
    except Exception as e1:
        if cp is not None:
            try:
                with open(PIPE_PATH, "rb") as f:
                    pipeline = cp.load(f)
                    last_load_error = None
            except Exception as e2:
                pipeline = None
                last_load_error = f"joblib: {repr(e1)} | cloudpickle: {repr(e2)}"
                app.logger.exception(f"Failed to load pipeline from {PIPE_PATH}")
                raise
        else:
            pipeline = None
            last_load_error = repr(e1)
            app.logger.exception(f"Failed to load pipeline from {PIPE_PATH}")
            raise
    return pipeline

# Contracts
REQUIRED_COLUMNS = [
    "TEN","RAC1P","CIT","SCHL","BLD","HUPAC","COW","MAR","SEX","VEH","WKL",
    "AGEP","NPF","GRPIP","WKHP"
]
NUMERIC_COLUMNS = ["AGEP","NPF","GRPIP","WKHP"]

@app.route("/", methods=["GET"])
def home():
    return (
        "<h2>Income Classifier API</h2>"
        "<p>GET <code>/health</code> for model status.</p>"
        "<p>POST <code>/predict</code> with JSON (single object or list of objects).</p>",
        200,
    )

@app.route("/health", methods=["GET"])
def health():
    try:
        get_pipeline()
        return jsonify(status="ok", model_path=str(PIPE_PATH), threshold=DECISION_THRESHOLD), 200
    except Exception:
        return jsonify(status="pipeline_not_loaded", model_path=str(PIPE_PATH), error=last_load_error), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        pl = get_pipeline()
    except Exception:
        return jsonify(error=f"Pipeline not loaded from {PIPE_PATH}", details=last_load_error), 500

    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify(error="Invalid JSON. Send an object or a list of objects."), 400
    if payload is None:
        return jsonify(error="Empty JSON payload."), 400

    if isinstance(payload, dict):
        df = pd.DataFrame([payload]); single = True
    elif isinstance(payload, list) and all(isinstance(x, dict) for x in payload):
        df = pd.DataFrame(payload); single = False
    else:
        return jsonify(error="Payload must be a JSON object or a list of JSON objects."), 400

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return jsonify(error=f"Missing required keys: {missing}"), 400

    for c in NUMERIC_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    try:
        probs = pl.predict_proba(df)[:, 1]
        thr = DECISION_THRESHOLD
        preds = (probs >= thr).astype(int)
    except Exception as e:
        return jsonify(error=f"Inference failed: {e}"), 500

    label_map = {0: "Income ≤ 50K", 1: "Income > 50K"}
    if single:
        conf = float(probs[0]) if preds[0] == 1 else float(1 - probs[0])
        return jsonify({
            "prediction": label_map.get(int(preds[0]), str(int(preds[0]))),
            "probability_income_gt_50k": float(probs[0]),
            "confidence_percent": round(conf * 100, 2),
            "threshold_used": thr,
        }), 200

    results = []
    for p, pr in zip(preds, probs):
        conf = float(pr) if p == 1 else float(1 - pr)
        results.append({
            "prediction": label_map.get(int(p), str(int(p))),
            "probability_income_gt_50k": float(pr),
            "confidence_percent": round(conf * 100, 2),
            "threshold_used": thr,
        })
    return jsonify(results), 200

if __name__ == "__main__":
    # REQUIRED for Render
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
