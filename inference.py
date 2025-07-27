from preprocessing import preprocess_data
import joblib
import pandas as pd
import json


# Load JSON payload from file
with open("test_payload.json", "r") as file:
    payload = json.load(file)

# Convert to DataFrame
raw_df = pd.DataFrame([payload])

# Load model and features
model = joblib.load("model/income_model.pkl")
features = joblib.load("model/model_features.pkl")

# Preprocess and encode
cleaned = preprocess_data(raw_df)
encoded = pd.get_dummies(cleaned, drop_first=True).astype(int)

# Align with training feature order
aligned_df = pd.DataFrame(columns=features)
aligned_df.loc[0] = 0
for col in encoded.columns:
    if col in aligned_df.columns:
        aligned_df.at[0, col] = encoded[col].values[0]

# Make prediction
prediction = model.predict(aligned_df)[0]
print("Predicted Income Above 50K?" , bool(prediction))
