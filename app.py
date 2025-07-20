# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('income_model.pkl')

# Load expected feature names
with open('model_features.pkl', 'rb') as f:
    feature_names = joblib.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return "Income Classification API is live!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse incoming JSON
        incoming_data = request.get_json()

        # Convert to DataFrame and align with training features
        input_df = pd.DataFrame([incoming_data])
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Predict
        prediction = model.predict(input_df)[0]
        predicted_class = '>50K' if prediction == 1 else '<=50K'

        return jsonify({'prediction': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

