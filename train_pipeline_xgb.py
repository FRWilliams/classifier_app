# train_pipeline_xgb.py
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Import stable preprocessing symbols
from preprocessing import mapping_impute, NUMERIC as NUMERIC_FROM_PRE

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ACSPUMS1Y2022_Georgia_Data.csv"   # <-- your training CSV here
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# ---------- Columns ----------
CATEGORICAL = ["TEN","RAC1P","CIT","SCHL","BLD","HUPAC","COW","MAR","SEX","VEH","WKL"]
NUMERIC     = list(NUMERIC_FROM_PRE)
ALL_FEATS   = CATEGORICAL + NUMERIC
TARGET      = "income_>50K"

# ---------- Mapper step (calls into preprocessing.mapping_impute) ----------
mapper = FunctionTransformer(mapping_impute, feature_names_out="one-to-one", validate=False)

def main():
    # ---------- Load + derive target ----------
    df_raw = pd.read_csv(DATA_PATH)
    df = df_raw.loc[df_raw["AGEP"] >= 16].copy()

    if TARGET not in df.columns:
        if "WAGP" not in df.columns:
            raise ValueError("Need WAGP in CSV to derive target or include income_>50K directly.")
        df[TARGET] = (df["WAGP"].astype(float) > 50000).astype(int)

    missing = set(ALL_FEATS + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    X = df[ALL_FEATS].copy()
    y = df[TARGET].astype(int).copy()

    # ---------- Split ----------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # ---------- Preprocessing ----------
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", ohe)])
    num_pipe_noscale = Pipeline([("impute", SimpleImputer(strategy="median"))])

    pre_base = ColumnTransformer([
        ("cat", cat_pipe, CATEGORICAL),
        ("num", num_pipe_noscale, NUMERIC),
    ])

    num_pipe_scaled = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    pre_for_lr = ColumnTransformer([
        ("cat", cat_pipe, CATEGORICAL),
        ("num", num_pipe_scaled, NUMERIC),
    ])

    # ---------- Class imbalance (XGB) ----------
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1)

    # ---------- Pipelines ----------
    pipe_lr = Pipeline([
        ("map_impute", mapper),
        ("pre", pre_for_lr),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, n_jobs=-1)),
    ])

    pipe_rf = Pipeline([
        ("map_impute", mapper),
        ("pre", pre_base),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1)),
    ])

    pipe_xgb = Pipeline([
        ("map_impute", mapper),
        ("pre", pre_base),
        ("clf", XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            n_estimators=100,
            max_depth=6,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            n_jobs=1,
        )),
    ])

    # ----------- Grids ----------
    grid_lr  = {"clf__C": [0.1, 1, 10]}
    grid_rf  = {"clf__n_estimators": [200, 300], "clf__max_depth": [None, 20, 30]}
    grid_xgb = {"clf__n_estimators": [100, 200], "clf__max_depth": [6, 9]}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = "roc_auc"

    def run_grid(name, pipe, grid):
        print(f"\n=== {name} ===")
        gs = GridSearchCV(pipe, grid, cv=cv, scoring=scorer, n_jobs=-1)
        gs.fit(X_train, y_train)
        print("Best", scorer, ":", gs.best_score_)
        print("Best Params:", gs.best_params_)
        y_prob = gs.best_estimator_.predict_proba(X_val)[:, 1]
        print("Hold-out ROC-AUC:", roc_auc_score(y_val, y_prob))
        return gs, y_prob

    gs_lr,  prob_lr  = run_grid("Logistic Regression", pipe_lr, grid_lr)
    gs_rf,  prob_rf  = run_grid("Random Forest",       pipe_rf, grid_rf)
    gs_xgb, prob_xgb = run_grid("XGBoost",             pipe_xgb, grid_xgb)

    # ---------- Pick best by hold-out ROC-AUC ----------
    candidates = [("logreg", gs_lr, prob_lr), ("rf", gs_rf, prob_rf), ("xgb", gs_xgb, prob_xgb)]
    best_name, best_gs, best_prob = max(candidates, key=lambda t: roc_auc_score(y_val, t[2]))
    print(f"\n🏆 Selected model: {best_name}")

    # ---------- Choose decision threshold (max F1) ----------
    prec, rec, thr = precision_recall_curve(y_val, best_prob)
    f1 = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    best_idx = int(np.argmax(f1))
    best_thr = float(thr[best_idx])
    print(f"Chosen threshold (max F1): {best_thr:.4f} | P={prec[best_idx]:.3f}, R={rec[best_idx]:.3f}")

    print("\nClassification Report @ chosen threshold:")
    y_pred = (best_prob >= best_thr).astype(int)
    print(classification_report(y_val, y_pred, digits=3))

    # ---------- Save artifacts ----------
    final_pipe = best_gs.best_estimator_  # includes mapping_impute + preprocessing + model
    pipe_path = MODEL_DIR / "income_pipeline.pkl"
    joblib.dump(final_pipe, pipe_path)

    thr_path = MODEL_DIR / "threshold.txt"
    thr_path.write_text(str(best_thr))

    print("\nSaved:")
    print("  Pipeline  ->", pipe_path)
    print("  Threshold ->", thr_path)

if __name__ == "__main__":
    main()
