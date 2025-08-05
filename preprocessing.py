import pandas as pd
import numpy as np
import joblib

# Load expected features at the top of your script
feature_list = joblib.load("model/model_features.pkl")

def preprocess_data(df):
    # === 1. Define Mapping Dictionaries ===
    marital_status_map = {1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated", 5: "Never married"}
    citizenship_map = {1: "Born in US", 2: "Born in Territory", 3: "Born abroad to US parents", 4: "Naturalized", 5: "Not a citizen"}
    class_of_worker_map = {0: "Not Applicable", 1: "Private for-profit", 2: "Private nonprofit", 3: "Local government", 4: "State government", 5: "Self-employed"}
    sex_map = {1: "Male", 2: "Female"}
    education_map = {0: "N/A", 1: "No schooling", 2: "Pre-K to Grade 4", 3: "Pre-K to Grade 4", 4: "Pre-K to Grade 4",
                    5: "Pre-K to Grade 4", 6: "Pre-K to Grade 4", 7: "Pre-K to Grade 4", 8: "Grade 5-8", 9: "Grade 5-8",
                    10: "Grade 5-8", 11: "Grade 5-8", 12: "Grade 9-12 (no diploma)", 13: "Grade 9-12 (no diploma)",
                    14: "Grade 9-12 (no diploma)", 15: "Grade 9-12 (no diploma)", 16: "High School Graduate", 17: "High School Graduate",
                    18: "Some College", 19: "Some College", 20: "Associate's", 21: "Bachelor's", 22: "Graduate Degree",
                    23: "Graduate Degree"}                    
    race_map = {1: "White", 2: "Black", 3: "American Indian", 4: "Alaska Native", 5: "Tribes Specified", 6: "Asian", 7: "Pacific Islander", 8: "Other", 9: "Two or More Races"}
    tenure_map = {0: "N/A", 1: "Owned with mortgage or loan (include home equity loans)", 2: "Owned Free And Clear", 3: "Rented", 4: "Occupied without payment of rent"}
    building_map = {0: "N/A", 1: "Mobile Home or Trailer", 2: "One-family house detached", 3: "One-family house attached",
                    4: "2 Apartments", 5: "3-4 Apartments", 6: "5-9 Apartments", 7: "10-19 Apartments",
                    8: "20-49 Apartments", 9: "50 or More Apartments", 10: "Boat, RV, van, etc."}
    children_map = {0: "N/A", 1: "With children under 6 years only", 2: "With children 6 to 17 years only",
                    3: "With children under 6 years and 6 to 17 years", 4: "No children"}
    vehicle_map = {-1: "N/A", 0: "No vehicles", 1: "1 vehicle", 2: "2 vehicles", 3: "3 vehicles",
                   4: "4 vehicles", 5: "5 vehicles", 6: "6 or more vehicles"}

    # === 2. Apply Mappings ===
    df["MAR"] = df.get("MAR", np.nan).map(marital_status_map)
    df["CIT"] = df.get("CIT", np.nan).map(citizenship_map)
    df["COW"] = df.get("COW", np.nan).map(class_of_worker_map)
    df["SEX"] = df.get("SEX", np.nan).map(sex_map)
    df["SCHL"] = df.get("SCHL", np.nan).map(education_map)
    df["RAC1P"] = df.get("RAC1P", np.nan).map(race_map)
    df["TEN"] = df.get("TEN", np.nan).map(tenure_map)
    df["BLD"] = df.get("BLD", np.nan).map(building_map)
    df["HUPAC"] = df.get("HUPAC", np.nan).map(children_map)
    df["VEH"] = df.get("VEH", np.nan).map(vehicle_map)

    # === 3. Impute missing values ===
    df_imputed = df.copy()

    # Categorical columns (string-based)
    object_cols = df_imputed.select_dtypes(include='object').columns.tolist()
    for col in object_cols:
        mode = df_imputed[col].mode()
        df_imputed[col].fillna(mode[0] if not mode.empty else "Unknown", inplace=True)

    # Numeric columns
    numeric_cols = df_imputed.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        median = df_imputed[col].median()
        df_imputed[col].fillna(median if not np.isnan(median) else 0, inplace=True)

    # === 4. Drop unused columns ===
    df_imputed.drop(columns=["ST"], errors="ignore", inplace=True)

    # === 5. Encode categorical features ===
    df_encoded = pd.get_dummies(df_imputed, drop_first=True).astype(int)

    # === 6. Align with model features ===
    aligned_df = pd.DataFrame(columns=feature_list)
    aligned_df.loc[0] = 0  # initialize all to 0

    for col in df_encoded.columns:
        if col in aligned_df.columns:
            aligned_df.at[0, col] = df_encoded[col].values[0]

    return aligned_df