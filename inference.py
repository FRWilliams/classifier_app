from preprocessing import preprocess_input
import joblib
import pandas as pd

# Load raw user data
new_data = pd.read_csv("new_input.csv")

# Preprocess data using same logic as training
X_cleaned = preprocess_input(new_data)

# Load trained model and features
model = joblib.load("model/income_model.pkl")
features = joblib.load("model/model_features.pkl")

# Ensure correct column order
X_cleaned = X_cleaned[features]

# Predict
predictions = model.predict(X_cleaned)
