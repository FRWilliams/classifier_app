# app.py
from flask import Flask, request, jsonify
from pathlib import Path
import os
import pandas as pd
import numpy as np
import joblib

# Make sure the module name used in the pickle exists at import time
import preprocessing  # registers preprocessing.mapping_impute for unpickling

try:
    import cloudpickle as cp  # optional fallback
except Exception:
    cp = None

# Optional: allow calling API from a browser app on another origin
# from flask_cors import CORS

app = Flask(__name__)
# CORS(app)

# --- Ensure UTF-8 in JSON ---
try:
    app.json.ensure_ascii = False  # Flask 2.3+
except Exception:
    app.config["JSON_AS_ASCII"] = False

# --- Paths & files ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
PIPE_PATH = MODEL_DIR / "income_pipeline.pkl"
THRESHOLD_PATH = MODEL_DIR / "threshold.txt"

# --- Lazy model load ---
pipeline = None
last_load_error = None
decision_threshold = None

def ensure_model_file():
    """
    Ensure income_pipeline.pkl exists; if not, try downloading it from MODEL_URL.
    """
    if PIPE_PATH.exists():
        return
    url = os.getenv("MODEL_URL")
    if not url:
        raise FileNotFoundError(f"Model file missing at {PIPE_PATH} and no MODEL_URL set")
    import requests
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PIPE_PATH.with_suffix(".tmp")
    with requests.get(url, stream=True, timeout=90) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                if chunk:
                    f.write(chunk)
    tmp.replace(PIPE_PATH)

def load_threshold():
    """
    Load decision threshold from file or env; fallback to 0.5.
    """
    env_thr = os.getenv("THRESHOLD")
    if env_thr:
        try:
            return float(env_thr)
        except ValueError:
            pass
    if THRESHOLD_PATH.exists():
        txt = THRESHOLD_PATH.read_text().strip()
        try:
            return float(txt)
        except ValueError:
            pass
    return 0.5

def get_pipeline():
    """
    Load the trained pipeline (first time only). Captures any load error.
    """
    global pipeline, last_load_error, decision_threshold
    if pipeline is not None:
        return pipeline
    try:
        ensure_model_file()
        # First, try joblib
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
                raise
        else:
            pipeline = None
            last_load_error = repr(e1)
            raise
    # Threshold (defer until after successful load)
    decision_threshold = load_threshold()
    return pipeline

# --- Contracts ---
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
        return jsonify(status="ok",
                       model_path=str(PIPE_PATH),
                       threshold=decision_threshold), 200
    except Exception:
        return jsonify(status="pipeline_not_loaded",
                       model_path=str(PIPE_PATH),
                       error=last_load_error), 500

@app.route("/predict", methods=["POST"])
def predict():
    # Ensure model is loaded
    try:
        pl = get_pipeline()
    except Exception:
        return jsonify(error=f"Pipeline not loaded from {PIPE_PATH}",
                       details=last_load_error), 500

    # Parse JSON
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify(error="Invalid JSON. Send an object or a list of objects."), 400

    if payload is None:
        return jsonify(error="Empty JSON payload."), 400

    # Normalize → DataFrame
    if isinstance(payload, dict):
        df = pd.DataFrame([payload]); single = True
    elif isinstance(payload, list) and all(isinstance(x, dict) for x in payload):
        df = pd.DataFrame(payload); single = False
    else:
        return jsonify(error="Payload must be a JSON object or a list of JSON objects."), 400

    # Validate keys
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return jsonify(error=f"Missing required keys: {missing}"), 400

    # Coerce numeric columns
    for c in NUMERIC_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Predict
    try:
        probs = pl.predict_proba(df)[:, 1]
        thr = decision_threshold if decision_threshold is not None else 0.5
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
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return "No file uploaded", 400

        try:
            payload = json.load(file)
        except Exception:
            return "Invalid JSON format", 400

        # Normalize → DataFrame
        if isinstance(payload, dict):
            df = pd.DataFrame([payload])
        elif isinstance(payload, list) and all(isinstance(x, dict) for x in payload):
            df = pd.DataFrame(payload)
        else:
            return "Payload must be a JSON object or a list of JSON objects.", 400
        
        # Validate required columns
        missing = [c for c in REQUIRED_COLUMNS if c not in payload.columns]
        if missing:
            return f"Missing required keys: {missing}", 400

        for c in NUMERIC_COLUMNS:
            if c in payload.columns:
                payload[c] = pd.to_numeric(payload[c], errors="coerce")

        try:
            pl = get_pipeline()
            probs = pl.predict_proba(payload)[:, 1]
            threshold = 0.6648
            preds = (probs >= threshold).astype(int)
        except Exception as e:
            return f"Inference failed: {e}", 500

        label_map = {0: "Income ≤ 50K", 1: "Income > 50K"}
        results = []
        for p, pr in zip(preds, probs):
            conf = pr if p == 1 else 1 - pr
            results.append({
                "prediction": label_map.get(int(p), str(int(p))),
                "probability_income_gt_50k": round(pr, 4),
                "confidence_percent": round(conf * 100, 2),
                "threshold_used": threshold
            })

        return jsonify(results)

    # GET request: show upload form
    return '''
        <h2>Upload JSON File for Prediction</h2>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="file" accept=".json">
            <input type="submit" value="Predict">
        </form>
    '''


if __name__ == "__main__":
    # 0.0.0.0 is required on Render
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
