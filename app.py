# app.py
from flask import Flask, request, jsonify
from pathlib import Path
import os
import pandas as pd
import numpy as np
import joblib

print(">>> APP STARTED, loading model lazily…")

app = Flask(__name__)
try:
    app.json.ensure_ascii = False  # Flask 2.3+
except Exception:
    app.config["JSON_AS_ASCII"] = False

# ------------------ PREPROCESSING (as in your original) ------------------
marital_status_map = {1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated", 5: "Never married"}
citizenship_map    = {1: "Born in US", 2: "Born in Territory", 3: "Born abroad to US parents", 4: "Naturalized", 5: "Not a citizen"}
class_of_worker_map= {0: "Not Applicable", 1: "Private for-profit", 2: "Private nonprofit", 3: "Local government",
                      4: "State government", 5: "Self-employed"}
sex_map            = {1: "Male", 2: "Female"}
education_map      = {
    0:"N/A",1:"No schooling",2:"Pre-K to Grade 4",3:"Pre-K to Grade 4",4:"Pre-K to Grade 4",
    5:"Pre-K to Grade 4",6:"Pre-K to Grade 4",7:"Pre-K to Grade 4",8:"Grade 5-8",9:"Grade 5-8",
    10:"Grade 5-8",11:"Grade 5-8",12:"Grade 9-12 (no diploma)",13:"Grade 9-12 (no diploma)",
    14:"Grade 9-12 (no diploma)",15:"Grade 9-12 (no diploma)",16:"High School Graduate",17:"High School Graduate",
    18:"Some College",19:"Some College",20:"Associate's",21:"Bachelor's",22:"Graduate Degree",23:"Graduate Degree"
}
race_map           = {1:"White",2:"Black",3:"American Indian",4:"Alaska Native",5:"Tribes Specified",
                      6:"Asian",7:"Pacific Islander",8:"Other",9:"Two or More Races"}
tenure_map         = {0:"N/A",1:"Owned with mortgage or loan (include home equity loans)",2:"Owned Free And Clear",
                      3:"Rented",4:"Occupied without payment of rent"}
building_map       = {0:"N/A",1:"Mobile Home or Trailer",2:"One-family house detached",3:"One-family house attached",
                      4:"2 Apartments",5:"3-4 Apartments",6:"5-9 Apartments",7:"10-19 Apartments",
                      8:"20-49 Apartments",9:"50 or More Apartments",10:"Boat, RV, van, etc."}
children_map       = {0:"N/A",1:"With children under 6 years only",2:"With children 6 to 17 years only",
                      3:"With children under 6 years and 6 to 17 years",4:"No children"}
vehicle_map        = {-1:"N/A",0:"No vehicles",1:"1 vehicle",2:"2 vehicles",3:"3 vehicles",
                      4:"4 vehicles",5:"5 vehicles",6:"6 or more vehicles"}

REQUIRED_COLUMNS = ["TEN","RAC1P","CIT","SCHL","BLD","HUPAC","COW","MAR","SEX","VEH","WKL","AGEP","NPF","GRPIP","WKHP"]
NUMERIC_COLUMNS  = ["AGEP","NPF","GRPIP","WKHP"]

def _map_if_numeric(series: pd.Series, mapping: dict) -> pd.Series:
    def _maybe_map(v):
        try:
            iv = int(str(v)) if str(v).isdigit() else v
            return mapping.get(iv, mapping.get(v, v))
        except Exception:
            return mapping.get(v, v)
    return series.map(_maybe_map)

def mapping_impute(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    if "MAR"   in X: X["MAR"]   = _map_if_numeric(X["MAR"],   marital_status_map)
    if "CIT"   in X: X["CIT"]   = _map_if_numeric(X["CIT"],   citizenship_map)
    if "COW"   in X: X["COW"]   = _map_if_numeric(X["COW"],   class_of_worker_map)
    if "SEX"   in X: X["SEX"]   = _map_if_numeric(X["SEX"],   sex_map)
    if "SCHL"  in X: X["SCHL"]  = _map_if_numeric(X["SCHL"],  education_map)
    if "RAC1P" in X: X["RAC1P"] = _map_if_numeric(X["RAC1P"], race_map)
    if "TEN"   in X: X["TEN"]   = _map_if_numeric(X["TEN"],   tenure_map)
    if "BLD"   in X: X["BLD"]   = _map_if_numeric(X["BLD"],   building_map)
    if "HUPAC" in X: X["HUPAC"] = _map_if_numeric(X["HUPAC"], children_map)
    if "VEH"   in X: X["VEH"]   = _map_if_numeric(X["VEH"],   vehicle_map)
    X.drop(columns=["ST"], errors="ignore", inplace=True)
    for c in NUMERIC_COLUMNS:
        if c in X:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    # simple imputations (match your original)
    obj_cols = X.select_dtypes(include="object").columns
    for c in obj_cols:
        mode = X[c].mode(dropna=True)
        X[c] = X[c].fillna(mode.iloc[0] if not mode.empty else "Unknown")
    num_cols = X.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        med = X[c].median()
        X[c] = X[c].fillna(0 if pd.isna(med) else med)
    return X
# ------------------ END PREPROCESSING ------------------

BASE_DIR = Path(__file__).resolve().parent
PIPE_PATH = BASE_DIR / "model" / "income_pipeline.pkl"
THRESHOLD_PATH = BASE_DIR / "model" / "threshold.txt"

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

def load_threshold():
    if THRESHOLD_PATH.exists():
        try:
            return float(THRESHOLD_PATH.read_text().strip())
        except Exception:
            pass
    return 0.6648  # your original working threshold

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
        return jsonify(status="ok", model_path=str(PIPE_PATH), threshold=load_threshold()), 200
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
        thr = load_threshold()
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
    # IMPORTANT for Render
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
