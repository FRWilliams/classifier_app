# app.py
from flask import Flask, request, jsonify
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)

# --- Ensure UTF-8 (so "≤" renders, not \u2264) ---
try:
    # Flask 2.3+
    app.json.ensure_ascii = False
except Exception:
    # Flask <= 2.2
    app.config["JSON_AS_ASCII"] = False

# --- Paths & model load ---
BASE_DIR = Path(__file__).resolve().parent
PIPE_PATH = BASE_DIR / "model" / "income_pipeline.pkl"

try:
    pipeline = joblib.load(PIPE_PATH)
except Exception as e:
    pipeline = None
    app.logger.error(f"Failed to load pipeline from {PIPE_PATH}: {e}")

# Columns the pipeline expects (raw ACS fields; strings or codes both OK)
REQUIRED_COLUMNS = [
    "TEN","RAC1P","CIT","SCHL","BLD","HUPAC","COW","MAR","SEX","VEH","WKL",
    "AGEP","NPF","GRPIP","WKHP"
]
NUMERIC_COLUMNS = ["AGEP","NPF","GRPIP","WKHP"]  # we’ll coerce these just in case

@app.route("/", methods=["GET"])
def home():
    return (
        "<h2>Income Classifier API</h2>"
        "<p>POST <code>/predict</code> with JSON (single object or list of objects).</p>"
        "<p>Try <code>/health</code> to check status.</p>",
        200,
    )

@app.route("/health", methods=["GET"])
def health():
    status = "ok" if pipeline is not None else "pipeline_not_loaded"
    return jsonify(status=status), (200 if pipeline is not None else 500)

@app.route("/predict", methods=["POST"])
def predict():
    if pipeline is None:
        return jsonify(error=f"Pipeline not loaded from {PIPE_PATH}"), 500

    payload = request.get_json(force=True)
    if payload is None:
        return jsonify(error="Empty or invalid JSON payload."), 400

    # Normalize input -> DataFrame
    if isinstance(payload, dict):
        df = pd.DataFrame([payload])
        single = True
    elif isinstance(payload, list) and all(isinstance(x, dict) for x in payload):
        df = pd.DataFrame(payload)
        single = False
    else:
        return jsonify(error="Payload must be a JSON object or a list of JSON objects."), 400

    # Check required columns; helpful error if any are missing
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return jsonify(error=f"Missing required keys: {missing}"), 400

    # Coerce numerics just in case they arrive as strings
    for c in NUMERIC_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Inference via single pipeline (mapping + impute + OHE + XGB)
    try:
        probs = pipeline.predict_proba(df)[:, 1]
        preds = (probs >= 0.5).astype(int)
    except Exception as e:
        return jsonify(error=f"Inference failed: {e}"), 500

    label_map = {0: "Income ≤ 50K", 1: "Income > 50K"}
    results = []
    for i, p in enumerate(preds):
        conf = probs[i] if p == 1 else 1 - probs[i]
        results.append({
            "prediction": label_map.get(int(p), str(p)),
            "probability_income_gt_50k": float(probs[i]),
            "confidence_percent": round(float(conf) * 100, 2),
        })

    return jsonify(results[0] if single else results), 200

if __name__ == "__main__":
    # Use host="0.0.0.0" when deploying in containers/Render; keep 127.0.0.1 for local-only
    app.run(host="127.0.0.1", port=5000, debug=False)
