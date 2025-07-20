import pandas as pd
import numpy as np

def preprocess_data(df):
    # ============================
    # 1. Define Mapping Dictionaries
    # ============================

    marital_status_map = {
        1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated", 5: "Never married"
    }

    citizenship_map = {
        1: "Born in US", 2: "Born in Territory", 3: "Born abroad to US parents",
        4: "Naturalized", 5: "Not a citizen"
    }

    class_of_worker_map = {
        0: "Not Applicable", 1: "Private for-profit", 2: "Private nonprofit",
        3: "Local government", 4: "State government", 5: "Self-employed"
    }

    sex_map = {1: "Male", 2: "Female"}

    education_map = {
        0: "N/A", 1: "No schooling", 2: "Pre-K to Grade 4", 3: "Pre-K to Grade 4",
        4: "Pre-K to Grade 4", 5: "Pre-K to Grade 4", 6: "Grade 5-8", 7: "Grade 5-8",
        8: "Grade 5-8", 9: "Grade 5-8", 10: "Grade 9-12 (no diploma)",
        11: "Grade 9-12 (no diploma)", 12: "Grade 9-12 (no diploma)",
        13: "Grade 9-12 (no diploma)", 14: "Grade 9-12 (no diploma)",
        15: "Grade 9-12 (no diploma)", 16: "High School Graduate",
        17: "High School Graduate", 18: "Some College", 19: "Some College",
        20: "Associate's", 21: "Bachelor's", 22: "Graduate Degree", 23: "Graduate Degree"
    }

    race_map = {
        1: "White", 2: "Black", 3: "American Indian", 4: "Alaska Native",
        5: "Tribes Specified", 6: "Asian", 7: "Pacific Islander", 8: "Other", 9: "Two or More Races"
    }

    tenure_map = {
        0: "N/A", 1: "Owned with mortgage or loan (include home equity loans)",
        2: "Owned Free And Clear", 3: "Rented", 4: "Occupied without payment of rent"
    }

    building_map = {
        0: "N/A", 1: "Mobile Home or Trailer", 2: "One-family house detached",
        3: "One-family house attached", 4: "2 Apartments", 5: "3-4 Apartments",
        6: "5-9 Apartments", 7: "10-19 Apartments", 8: "20-49 Apartments",
        9: "50 or More Apartments", 10: "Boat, RV, van, etc."
    }

    children_map = {
        0: "N/A", 1: "With children under 6 years only", 2: "With children 6 to 17 years only",
        3: "With children under 6 years and 6 to 17 years", 4: "No children"
    }

    vehicle_map = {
        -1: "N/A", 0: "No vehicles", 1: "1 vehicle", 2: "2 vehicles",
        3: "3 vehicles", 4: "4 vehicles", 5: "5 vehicles", 6: "6 or more vehicles"
    }

    # ============================
    # 2. Apply the Mapping
    # ============================

    df["MAR"] = df["MAR"].map(marital_status_map)
    df["CIT"] = df["CIT"].map(citizenship_map)
    df["COW"] = df["COW"].map(class_of_worker_map)
    df["SEX"] = df["SEX"].map(sex_map)
    df["SCHL"] = df["SCHL"].map(education_map)
    df["RAC1P"] = df["RAC1P"].map(race_map)
    df["TEN"] = df["TEN"].map(tenure_map)
    df["BLD"] = df["BLD"].map(building_map)
    df["HUPAC"] = df["HUPAC"].map(children_map)
    df["VEH"] = df["VEH"].map(vehicle_map)

    # ============================
    # 3. Create income label
    # ============================
    df["income"] = np.where(df["WAGP"] > 50000, ">50K", "<=50K")

    # ============================
    # 4. Filter records
    # ============================
    df_filtered = df[
        (df["AGEP"] >= 16) &
        (df["WKHP"] > 0) &
        (df["COW"].notna()) &
        (df["WAGP"] > 0) &
        (df["AGEP"].notna()) &
        (df["WKHP"].notna()) &
        (df["SEX"].notna()) &
        (df["RAC1P"].notna())
    ].copy()

    # ============================
    # 5. Impute missing values
    # ============================

    # Create WAGP_BIN for group-based imputation
    df_filtered["WAGP_BIN"] = pd.qcut(df_filtered["WAGP"], q=4, duplicates="drop")

    # Impute categorical (object) columns
    object_cols = df_filtered.select_dtypes(include='object').columns.tolist()
    for col in object_cols:
        df_filtered[col] = df_filtered.groupby(
            ["COW", "WAGP_BIN", "RAC1P", "SEX"]
        )[col].transform(
            lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x.fillna("Unknown")
        )

    # Impute numeric columns using group-wise median
    numeric_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        df_filtered[col] = df_filtered.groupby(["COW", "WAGP_BIN", "RAC1P", "SEX"])[col].transform(
            lambda x: x.fillna(x.median()) if x.notnull().any() else x.fillna(0)
        )


    # Drop helper columns
    df_filtered.drop(columns=["WAGP_BIN", "WAGP", "ST"], errors='ignore', inplace=True)

    return df_filtered
