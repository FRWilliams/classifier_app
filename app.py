# app.py
from flask import Flask, request, jsonify
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

print(">>> APP STARTED, loading model…")

app = Flask(__name__)

# --- Ensure UTF-8 in JSON responses (so "≤" renders, not \u2264) ---
try:
    # Flask 2.3+
    app.json.ensure_ascii = False
except Exception:
    # Flask <= 2.2
    app.config["JSON_AS_ASCII"] = False


# =============================================================================
# Custom preprocessing used during training
# IMPORTANT: These names must exist BEFORE loading the pickle so unpickling works.
# If you trained with different names or dicts, paste the exact ones you used.
# =============================================================================

# Categorical code -> label maps (PLACEHOLDERS; replace with your real mappings)
# ---------- Mapping dicts (codes -> strings) ----------
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


# Helper used by mapping_impute
def _map_if_numeric(series: pd.Series, mapping: dict) -> pd.Series:
    """
    If the values are numeric codes (or numeric-looking strings), map via dict.
    Otherwise, return values (or mapped if keys already match).
    Unknowns pass through unchanged.
    """
    if not isinstance(series, pd.Series):
        return series
    # Try to map ints first (common when codes are integers)
    def _maybe_map(v):
        try:
            # treat digit-like strings as ints, e.g., "10" -> 10
            iv = int(str(v)) if str(v).isdigit() else v
            return mapping.get(iv, mapping.get(v, v))
        except Exception:
            return mapping.get(v, v)
    return series.map(_maybe_map)

# Columns the pipeline expects (raw ACS fields; strings or codes both OK)
REQUIRED_COLUMNS = [
    "TEN","RAC1P","CIT","SCHL","BLD","HUPAC","COW","MAR","SEX","VEH","WKL",
    "AGEP","NPF","GRPIP","WKHP"
]
# The training code referenced a global named `NUMERIC`; keep it identical here
NUMERIC = ["AGEP","NPF","GRPIP","WKHP"]
# We also use this in request validation
NUMERIC_COLUMNS = NUMERIC

def mapping_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Map numeric codes -> strings for categorical columns (prevents treating codes as ordinal).
    - Drop 'ST' if present.
    - Impute: object cols by mode; numeric cols by median.
    - Ensure declared numeric columns are numeric dtype.
    """
    X = df.copy()

    # Map codes -> strings for categoricals (idempotent if already strings)
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

    # Drop unused if present
    X.drop(columns=["ST"], errors="ignore", inplace=True)

    # Ensure numeric cols really numeric (in case read as strings)
    for c in NUMERIC:
        if c in X:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # Impute objects by mode
    obj_cols = X.select_dtypes(include="object").columns
    for c in obj_cols:
        mode = X[c].mode(dropna=True)
        X[c] = X[c].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    # Impute numerics by median
    num_cols = X.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        med = X[c].median()
        X[c] = X[c].fillna(0 if pd.isna(med) else med)

    return X


# =============================================================================
# Model loading (lazy) + paths
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
PIPE_PATH = BASE_DIR / "model" / "income_pipeline.pkl"

pipeline = None
last_load_error = None

def get_pipeline():
    """Lazy-load the trained pipeline, capturing the first error for /health."""
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
    # Ensure model is available
    try:
        pl = get_pipeline()
    except Exception:
        return jsonify(error=f"Pipeline not loaded from {PIPE_PATH}", details=last_load_error), 500

    # Parse JSON
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify(error="Invalid JSON. Send an object or a list of objects."), 400

    if payload is None:
        return jsonify(error="Empty JSON payload."), 400

    # Normalize payload -> DataFrame
    if isinstance(payload, dict):
        df = pd.DataFrame([payload]); single = True
    elif isinstance(payload, list) and all(isinstance(x, dict) for x in payload):
        df = pd.DataFrame(payload); single = False
    else:
        return jsonify(error="Payload must be a JSON object or a list of JSON objects."), 400

    # Validate required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return jsonify(error=f"Missing required keys: {missing}"), 400

    # Coerce numeric columns
    for c in NUMERIC_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Predict
    try:
        # If your pipeline includes the mapping_impute step internally,
        # just call predict_proba directly as below.
        probs = pl.predict_proba(df)[:, 1]  # probability of Income > 50K
        threshold = 0.6648  # adjust if you choose a different operating point
        preds = (probs >= threshold).astype(int)
    except Exception as e:
        return jsonify(error=f"Inference failed: {e}"), 500

    # Format response
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
    # Use host="0.0.0.0" for containers/Render; keep 127.0.0.1 for local-only
    app.run(host="127.0.0.1", port=5000, debug=False)
