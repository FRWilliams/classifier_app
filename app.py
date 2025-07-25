# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from preprocessing import preprocess_data

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        cleaned = preprocess_data(df)
        model = joblib.load("model/income_model.pkl")
        features = joblib.load("model/model_features.pkl")
        prediction = model.predict(cleaned[features])
        return jsonify({"prediction": prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)

