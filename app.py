# app.py
from flask import Flask, request, jsonify
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from preprocessing import mapping_impute  # ✅ Modular import

print(">>> APP STARTED, loading model…")

app = Flask(__name__)

# --- Ensure UTF-8 in JSON responses (so "≤" renders, not \u2264) ---
try:
    app.json.ensure_ascii = False
except Exception:
    app.config["JSON_AS_ASCII"] = False

# Required columns for validation
REQUIRED_COLUMNS = [
    "TEN","RAC1P","CIT","SCHL","BLD","HUPAC","COW","MAR","SEX","VEH","WKL",
    "AGEP","NPF","GRPIP","WKHP"
]
NUMERIC_COLUMNS = ["AGEP","NPF","GRPIP","WKHP"]

# =============================================================================
# Model loading (lazy) + paths
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
PIPE_PATH = BASE_DIR / "model" / "income_pipeline.pkl"

pipeline = None
last_load_error = None

def get_pipeline():
    global pipeline, last_load_error
    if pipeline is None:
        try:
            pipeline = joblib.load(PIPE_PATH)
            last_load_error = None
        except Exception as e:
            pipeline = None
            last_load_error = repr(e)
            app.logger.exception(f"Failed to load pipeline from {PIPE_PATH}")
            raise
    return pipeline

# =============================================================================
# Routes
# =============================================================================

@app.route("/", methods=["GET"])
def home():
    return (
        "<h2>Income Classifier API</h2>"
        "<p>POST <code>/predict</code> with JSON (single object or list of objects).</p>"
        "<p>GET <code>/health</code> for model status.</p>",
        200,
    )

@app.route("/health", methods=["GET"])
def health():
    try:
        get_pipeline()
        return jsonify(status="ok", model_path=str(PIPE_PATH)), 200
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
        threshold = 0.6648
        preds = (probs >= threshold).astype(int)
    except Exception as e:
        return jsonify(error=f"Inference failed: {e}"), 500

    label_map = {0: "Income ≤ 50K", 1: "Income > 50K"}
    if single:
        conf = float(probs[0]) if preds[0] == 1 else float(1 - probs[0])
        out = {
            "prediction": label_map.get(int(preds[0]), str(int(preds[0]))),
            "probability_income_gt_50k": float(probs[0]),
            "confidence_percent": round(conf * 100, 2),
        }
        return jsonify(out), 200

    results = []
    for p, pr in zip(preds, probs):
        conf = float(pr) if p == 1 else float(1 - pr)
        results.append({
            "prediction": label_map.get(int(p), str(int(p))),
            "probability_income_gt_50k": float(pr),
            "confidence_percent": round(conf * 100, 2),
        })
    return jsonify(results), 200

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
