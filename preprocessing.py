# preprocessing.py
import pandas as pd
import numpy as np

# ---- Categorical & numeric schema (keep consistent with training) ----
CATEGORICAL = ["TEN","RAC1P","CIT","SCHL","BLD","HUPAC","COW","MAR","SEX","VEH","WKL"]
NUMERIC     = ["AGEP","NPF","GRPIP","WKHP"]

# ---- Mappings (same as your earlier ones) ----
marital_status_map  = {1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated", 5: "Never married"}
citizenship_map     = {1: "Born in US", 2: "Born in Territory", 3: "Born abroad to US parents", 4: "Naturalized", 5: "Not a citizen"}
class_of_worker_map = {0: "Not Applicable", 1: "Private for-profit", 2: "Private nonprofit", 3: "Local government",
                       4: "State government", 5: "Self-employed"}
sex_map             = {1: "Male", 2: "Female"}
education_map       = {
    0:"N/A",1:"No schooling",2:"Pre-K to Grade 4",3:"Pre-K to Grade 4",4:"Pre-K to Grade 4",
    5:"Pre-K to Grade 4",6:"Pre-K to Grade 4",7:"Pre-K to Grade 4",8:"Grade 5-8",9:"Grade 5-8",
    10:"Grade 5-8",11:"Grade 5-8",12:"Grade 9-12 (no diploma)",13:"Grade 9-12 (no diploma)",
    14:"Grade 9-12 (no diploma)",15:"Grade 9-12 (no diploma)",16:"High School Graduate",17:"High School Graduate",
    18:"Some College",19:"Some College",20:"Associate's",21:"Bachelor's",22:"Graduate Degree",23:"Graduate Degree"
}
race_map            = {1:"White",2:"Black",3:"American Indian",4:"Alaska Native",5:"Tribes Specified",
                       6:"Asian",7:"Pacific Islander",8:"Other",9:"Two or More Races"}
tenure_map          = {0:"N/A",1:"Owned with mortgage or loan (include home equity loans)",2:"Owned Free And Clear",
                       3:"Rented",4:"Occupied without payment of rent"}
building_map        = {0:"N/A",1:"Mobile Home or Trailer",2:"One-family house detached",3:"One-family house attached",
                       4:"2 Apartments",5:"3-4 Apartments",6:"5-9 Apartments",7:"10-19 Apartments",
                       8:"20-49 Apartments",9:"50 or More Apartments",10:"Boat, RV, van, etc."}
children_map        = {0:"N/A",1:"With children under 6 years only",2:"With children 6 to 17 years only",
                       3:"With children under 6 years and 6 to 17 years",4:"No children"}
vehicle_map         = {-1:"N/A",0:"No vehicles",1:"1 vehicle",2:"2 vehicles",3:"3 vehicles",
                       4:"4 vehicles",5:"5 vehicles",6:"6 or more vehicles"}

def _is_intlike(v) -> bool:
    s = str(v)
    if s.startswith(("+", "-")):
        s = s[1:]
    return s.isdigit()

def _map_if_numeric(series: pd.Series, mapping: dict) -> pd.Series:
    """
    Map numeric codes to human-readable strings.
    If a value isn't in the mapping, return it as a **string** (not int),
    and preserve NaN as NaN. This guarantees a uniform 'str or NaN' dtype.
    """
    def _map_one(v):
        if pd.isna(v):
            return np.nan
        # try int-like first (e.g., "12" or 12)
        if _is_intlike(v):
            iv = int(str(v))
            if iv in mapping:
                return mapping[iv]
        # then try direct mapping on the original value
        if v in mapping:
            return mapping[v]
        # fallback: force to string to avoid int/str mixes
        return str(v)

    return series.map(_map_one).astype("object")  # object dtype with python str/NaN

def mapping_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Map numeric codes -> strings for categorical cols (uniform str/NaN)
    - Drop 'ST' if present
    - Ensure NUMERIC cols are numeric
    - Impute: object cols by mode; numeric cols by median
    """
    X = df.copy()

    # Apply mappings (now guarantees strings for categories)
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

    # Ensure ALL categorical columns are strings (preserve NaN)
    for c in CATEGORICAL:
        if c in X:
            X[c] = X[c].apply(lambda v: v if pd.isna(v) else str(v))

    # Drop unused
    X.drop(columns=["ST"], errors="ignore", inplace=True)

    # Ensure numeric columns are numeric
    for c in NUMERIC:
        if c in X:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # Impute object by mode
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

__all__ = ["mapping_impute", "NUMERIC", "CATEGORICAL"]
