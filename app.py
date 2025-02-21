import pickle
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load('model/logistic_regression_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Breast Cancer Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)

        # Scale features
        scaled_features = scaler.transform(features)

        # Predict
        prediction = model.predict(scaled_features)
        probability = model.predict_proba(scaled_features)[:, 1]

        result = {
            'prediction': int(prediction[0]),  # 0 = Benign, 1 = Malignant
            'probability': float(probability[0])
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
