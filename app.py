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
        payload = request.get_json(force=True)
        if not payload:
            raise ValueError("Empty or invalid input payload.")

        input_df = pd.DataFrame([payload])
        processed_df = preprocess_data(input_df)

        # Predict class and confidence
        predicted_class = model.predict(processed_df)[0]
        proba = model.predict_proba(processed_df)[0]

        label_map = {0: "Income ≤ 50K", 1: "Income > 50K"}
        readable_output = label_map.get(int(predicted_class), "Unknown")
        confidence_score = round(float(proba[int(predicted_class)]) * 100, 2)

        return jsonify({
            "prediction": readable_output,
            "confidence_percent": f"{confidence_score}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400
if __name__ == "__main__":
    app.run(port=5000)