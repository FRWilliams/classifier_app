# # train_pipeline_xgb.py
# # Build one sklearn Pipeline (mapping-safe + impute + OHE + XGBoost)
# # Input: data/train_data.csv  (may contain either raw ACS codes OR mapped strings)
# # Output: model/income_pipeline.pkl

# from pathlib import Path
# import numpy as np
# import pandas as pd
# import joblib

# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import roc_auc_score, classification_report

# from xgboost import XGBClassifier

# # ----------------------------
# # Paths
# # ----------------------------
# BASE_DIR = Path(__file__).resolve().parent
# DATA_PATH = BASE_DIR / "data" / "ACSPUMS1Y2022_Georgia_Data.csv"   # <-- put your training CSV here
# MODEL_DIR = BASE_DIR / "model"
# MODEL_DIR.mkdir(exist_ok=True)

# # ----------------------------
# # Columns & Target
# # ----------------------------
# # Raw feature columns we expect in the CSV (either as codes OR mapped strings)
# CATEGORICAL = ["TEN","RAC1P","CIT","SCHL","BLD","HUPAC","COW","MAR","SEX","VEH","WKL"]
# NUMERIC     = ["AGEP","NPF","GRPIP","WKHP"]   # GRPIP stays numeric per your note
# ALL_FEATURES = CATEGORICAL + NUMERIC

# TARGET_NAME = "income_>50K"   # we will create this from WAGP > 50000

# # ----------------------------
# # Mapping dicts (used only if your CSV still has numeric codes)
# # If your CSV is already mapped to strings, this step leaves values as-is.
# # ----------------------------
# marital_status_map = {1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated", 5: "Never married"}
# citizenship_map    = {1: "Born in US", 2: "Born in Territory", 3: "Born abroad to US parents", 4: "Naturalized", 5: "Not a citizen"}
# class_of_worker_map= {0: "Not Applicable", 1: "Private for-profit", 2: "Private nonprofit", 3: "Local government",
#                       4: "State government", 5: "Self-employed"}
# sex_map            = {1: "Male", 2: "Female"}
# education_map      = {
#     0:"N/A",1:"No schooling",2:"Pre-K to Grade 4",3:"Pre-K to Grade 4",4:"Pre-K to Grade 4",
#     5:"Pre-K to Grade 4",6:"Pre-K to Grade 4",7:"Pre-K to Grade 4",8:"Grade 5-8",9:"Grade 5-8",
#     10:"Grade 5-8",11:"Grade 5-8",12:"Grade 9-12 (no diploma)",13:"Grade 9-12 (no diploma)",
#     14:"Grade 9-12 (no diploma)",15:"Grade 9-12 (no diploma)",16:"High School Graduate",17:"High School Graduate",
#     18:"Some College",19:"Some College",20:"Associate's",21:"Bachelor's",22:"Graduate Degree",23:"Graduate Degree"
# }
# race_map           = {1:"White",2:"Black",3:"American Indian",4:"Alaska Native",5:"Tribes Specified",
#                       6:"Asian",7:"Pacific Islander",8:"Other",9:"Two or More Races"}
# tenure_map         = {0:"N/A",1:"Owned with mortgage or loan (include home equity loans)",2:"Owned Free And Clear",
#                       3:"Rented",4:"Occupied without payment of rent"}
# building_map       = {0:"N/A",1:"Mobile Home or Trailer",2:"One-family house detached",3:"One-family house attached",
#                       4:"2 Apartments",5:"3-4 Apartments",6:"5-9 Apartments",7:"10-19 Apartments",
#                       8:"20-49 Apartments",9:"50 or More Apartments",10:"Boat, RV, van, etc."}
# children_map       = {0:"N/A",1:"With children under 6 years only",2:"With children 6 to 17 years only",
#                       3:"With children under 6 years and 6 to 17 years",4:"No children"}
# vehicle_map        = {-1:"N/A",0:"No vehicles",1:"1 vehicle",2:"2 vehicles",3:"3 vehicles",
#                       4:"4 vehicles",5:"5 vehicles",6:"6 or more vehicles"}

# def _map_if_numeric(series, mapping):
#     """If the series looks numeric/coded, map to strings; if already strings, return unchanged."""
#     if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
#         return series.map(mapping)
#     return series

# def map_codes_to_strings(df: pd.DataFrame) -> pd.DataFrame:
#     """Make the training robust: if values are numeric codes, map them to strings.
#        If already strings (your 'after mappings' table), leave as-is."""
#     X = df.copy()
#     if "MAR"  in X: X["MAR"]  = _map_if_numeric(X["MAR"],  marital_status_map)
#     if "CIT"  in X: X["CIT"]  = _map_if_numeric(X["CIT"],  citizenship_map)
#     if "COW"  in X: X["COW"]  = _map_if_numeric(X["COW"],  class_of_worker_map)
#     if "SEX"  in X: X["SEX"]  = _map_if_numeric(X["SEX"],  sex_map)
#     if "SCHL" in X: X["SCHL"] = _map_if_numeric(X["SCHL"], education_map)
#     if "RAC1P"in X: X["RAC1P"]= _map_if_numeric(X["RAC1P"],race_map)
#     if "TEN"  in X: X["TEN"]  = _map_if_numeric(X["TEN"],  tenure_map)
#     if "BLD"  in X: X["BLD"]  = _map_if_numeric(X["BLD"],  building_map)
#     if "HUPAC"in X: X["HUPAC"]= _map_if_numeric(X["HUPAC"],children_map)
#     if "VEH"  in X: X["VEH"]  = _map_if_numeric(X["VEH"],  vehicle_map)
#     # Drop unused if present
#     X.drop(columns=["ST"], errors="ignore", inplace=True)
#     return X

# def mapping_impute(df: pd.DataFrame) -> pd.DataFrame:
#     """Your grouped imputation style:
#        - Apply robust mapping (if needed)
#        - Impute object (mode) and numeric (median)
#     """
#     X = map_codes_to_strings(df)

#     # Object -> mode
#     obj_cols = X.select_dtypes(include="object").columns
#     for c in obj_cols:
#         mode = X[c].mode(dropna=True)
#         X[c] = X[c].fillna(mode.iloc[0] if not mode.empty else "Unknown")

#     # Numeric -> median
#     num_cols = X.select_dtypes(include=[np.number]).columns
#     for c in num_cols:
#         med = X[c].median()
#         X[c] = X[c].fillna(0 if np.isnan(med) else med)

#     return X

# # Wrap into a sklearn transformer so it can live inside a Pipeline
# mapper = FunctionTransformer(mapping_impute, feature_names_out="one-to-one", validate=False)

# # ----------------------------
# # Preprocessor: OHE cats, impute guardrails
# # ----------------------------
# # NOTE: If you're on sklearn <1.2 use OneHotEncoder(sparse=False)
# try:
#     ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
# except TypeError:
#     ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

# cat_pipe = Pipeline([
#     ("impute", SimpleImputer(strategy="most_frequent")),
#     ("ohe", ohe)
# ])

# num_pipe = Pipeline([
#     ("impute", SimpleImputer(strategy="median"))
# ])

# pre = ColumnTransformer(
#     transformers=[
#         ("cat", cat_pipe, CATEGORICAL),
#         ("num", num_pipe, NUMERIC)
#     ],
#     remainder="drop"
# )

# # ----------------------------
# # Load data
# # ----------------------------
# df = pd.read_csv(DATA_PATH)

# # Create target from WAGP > 50000 (1 => >50K, 0 => <=50K)
# if TARGET_NAME not in df.columns:
#     if "WAGP" not in df.columns:
#         raise ValueError("train_data.csv must contain WAGP to derive the target or include income_>50K directly.")
#     df[TARGET_NAME] = (df["WAGP"].astype(float) > 50000).astype(int)

# # Features = ALL_FEATURES; drop WAGP to avoid leakage
# missing = set(ALL_FEATURES + [TARGET_NAME]) - set(df.columns)
# if missing:
#     raise ValueError(f"train_data.csv missing required columns: {missing}")

# X = df[ALL_FEATURES].copy()
# y = df[TARGET_NAME].astype(int).copy()

# # ----------------------------
# # Split & handle class imbalance
# # ----------------------------
# X_train, X_val, y_train, y_val = train_test_split(
#     X, y, test_size=0.20, random_state=42, stratify=y
# )

# pos = int((y_train == 1).sum())
# neg = int((y_train == 0).sum())
# scale_pos_weight = neg / max(pos, 1)

# # ----------------------------
# # XGBoost model (your best params)
# # ----------------------------
# xgb = XGBClassifier(
#     random_state=42,
#     eval_metric="logloss",
#     n_estimators=100,     # from your grid's best params
#     max_depth=6,         # from your grid's best params
#     n_jobs=-1,
#     scale_pos_weight=scale_pos_weight
# )

# # ----------------------------
# # Full Pipeline: mapping+impute -> OHE -> XGB
# # ----------------------------
# pipe = Pipeline([
#     ("map_impute", mapper),
#     ("pre", pre),
#     ("model", xgb)
# ])

# # ----------------------------
# # Train, Evaluate, Save
# # ----------------------------
# pipe.fit(X_train, y_train)

# y_prob = pipe.predict_proba(X_val)[:, 1]
# y_pred = (y_prob >= 0.7).astype(int)

# print(f"Validation ROC-AUC: {roc_auc_score(y_val, y_prob):.4f}")
# print("Classification Report (0=<=50K, 1=>50K):")
# print(classification_report(y_val, y_pred, digits=3))

# out_path = MODEL_DIR / "income_pipeline.pkl"
# joblib.dump(pipe, out_path)
# print("Saved pipeline to:", out_path)


#@title Using Pipelines to Improve Model Performance 
# teach_pipelines_cv.py
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ACSPUMS1Y2022_Georgia_Data.csv"   # <-- put your training CSV here
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# ---------- Columns ----------
CATEGORICAL = ["TEN","RAC1P","CIT","SCHL","BLD","HUPAC","COW","MAR","SEX","VEH","WKL"]
NUMERIC     = ["AGEP","NPF","GRPIP","WKHP"]  # GRPIP stays numeric per your note
ALL_FEATS   = CATEGORICAL + NUMERIC
TARGET      = "income_>50K"

# ---------- Load & target creation ----------
df = pd.read_csv(DATA_PATH)
if TARGET not in df.columns:
    if "WAGP" not in df.columns:
        raise ValueError("Need WAGP in CSV to derive target or include income_>50K directly.")
    df[TARGET] = (df["WAGP"].astype(float) > 50000).astype(int)

missing = set(ALL_FEATS + [TARGET]) - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}")

X = df[ALL_FEATS].copy()
y = df[TARGET].astype(int).copy()

# ---------- Train/validation split (CV happens inside train only) ----------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# ---------- Shared preprocessing ----------
# - Impute categoricals with mode and OHE (ignore unknowns)
# - Impute numerics with median
# - Scale numerics for Logistic Regression only (we'll add scaler in that pipeline)
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe", ohe)
])
num_pipe_noscale = Pipeline([
    ("impute", SimpleImputer(strategy="median"))
])
pre_base = ColumnTransformer([
    ("cat", cat_pipe, CATEGORICAL),
    ("num", num_pipe_noscale, NUMERIC)
])

# For Logistic Regression we add scaling on numerics after the ColumnTransformer:
num_pipe_scaled = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])
pre_for_lr = ColumnTransformer([
    ("cat", cat_pipe, CATEGORICAL),
    ("num", num_pipe_scaled, NUMERIC)
])

# ---------- Class imbalance helper ----------
pos = int((y_train == 1).sum())
neg = int((y_train == 0).sum())
scale_pos_weight = neg / max(pos, 1)  # for XGB

# ---------- Pipelines ----------
pipe_lr = Pipeline([
    ("pre", pre_for_lr),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, n_jobs=-1))
])

pipe_rf = Pipeline([
    ("pre", pre_base),
    ("clf", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1))
])

pipe_xgb = Pipeline([
    ("pre", pre_base),
    ("clf", XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        n_estimators=100,   # start from your best
        max_depth=6,        # start from your best
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1
    ))
])

# ---------- Grids (keep modest for class) ----------
grid_lr  = {"clf__C": [0.1, 1, 10]}
grid_rf  = {"clf__n_estimators": [200, 300],
            "clf__max_depth": [None, 20, 30]}
grid_xgb = {"clf__n_estimators": [100, 200],
            "clf__max_depth": [6, 9]}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scorer = "roc_auc"   # better than accuracy for imbalanced problems

def run_grid(name, pipe, grid):
    print(f"\n=== {name} ===")
    gs = GridSearchCV(pipe, grid, cv=cv, scoring=scorer, n_jobs=-1)
    gs.fit(X_train, y_train)
    print("Best", scorer, ":", gs.best_score_)
    print("Best Params:", gs.best_params_)
    # Evaluate on hold-out validation (same preprocessing via pipeline)
    y_prob = gs.best_estimator_.predict_proba(X_val)[:, 1]
    print("Hold-out ROC-AUC:", roc_auc_score(y_val, y_prob))
    return gs, y_prob

gs_lr,  prob_lr  = run_grid("Logistic Regression", pipe_lr, grid_lr)
gs_rf,  prob_rf  = run_grid("Random Forest",       pipe_rf, grid_rf)
gs_xgb, prob_xgb = run_grid("XGBoost",             pipe_xgb, grid_xgb)

# ---------- Pick best by hold-out ROC-AUC ----------
candidates = [
    ("logreg", gs_lr,  prob_lr),
    ("rf",     gs_rf,  prob_rf),
    ("xgb",    gs_xgb, prob_xgb),
]
best_name, best_gs, best_prob = max(
    candidates, key=lambda t: roc_auc_score(y_val, t[2])
)
print(f"\n🏆 Selected model: {best_name}")

# ---------- Choose a decision threshold on validation ----------
prec, rec, thr = precision_recall_curve(y_val, best_prob)
# Example: maximize F1
f1 = (2*prec[:-1]*rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
best_idx = np.argmax(f1)
best_thr = float(thr[best_idx])
print(f"Chosen threshold (max F1): {best_thr:.4f} | P={prec[best_idx]:.3f}, R={rec[best_idx]:.3f}")

# Evaluate at chosen threshold
y_pred = (best_prob >= best_thr).astype(int)
print("\nClassification Report at chosen threshold:")
print(classification_report(y_val, y_pred, digits=3))

# ---------- Save the winning pipeline + threshold ----------
final_pipe = best_gs.best_estimator_     # this includes preprocessing + model
pipe_path = MODEL_DIR / "income_pipeline.pkl"
joblib.dump(final_pipe, pipe_path)


print("\nSaved:")
print("  Pipeline ->", pipe_path)
print("  Threshold->", Path)
