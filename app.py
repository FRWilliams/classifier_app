from flask import Flask, request, jsonify
import pandas as pd
import joblib
from preprocessing import preprocess_data  # Assuming your preprocessing script is named preprocessing.py

app = Flask(__name__)

# Load model and expected features
model = joblib.load("model\income_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse incoming JSON payload
        payload = request.get_json(force=True)

        if not payload:
            raise ValueError("Empty or invalid input payload.")

        # Convert to DataFrame and preprocess
        input_df = pd.DataFrame([payload])
        processed_df = preprocess_data(input_df)

        # Run prediction
        prediction = model.predict(processed_df)[0]

        # Convert prediction to readable label
        label_map = {0: "Income ≤ 50K", 1: "Income > 50K"}
        readable_output = label_map.get(int(prediction), "Unknown")

        return jsonify({"prediction": readable_output})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)

