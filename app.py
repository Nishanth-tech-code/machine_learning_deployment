
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('student_performance_model.joblib')
scaler = joblib.load('student_performance_scaler.joblib')

weights = model['weights']
bias = model['bias']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Define feature names in the correct order as used during training
        feature_names = [
            'Hours Studied',
            'Previous Scores',
            'Sleep Hours',
            'Sample Question Papers Practiced',
            'Extracurricular_Encoded'
        ]

        # Create a DataFrame from the input data to ensure correct feature order for scaling
        input_df = pd.DataFrame([data], columns=feature_names)

        # Scale the input data using the pre-fitted scaler
        scaled_input = scaler.transform(input_df)

        # Make prediction using the loaded weights and bias
        prediction = np.dot(scaled_input, weights) + bias

        return jsonify({'Performance Index Prediction': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # For local testing, you can run:
    # app.run(debug=True)
    # For deployment, consider using a production-ready WSGI server like Gunicorn
    print("To run the app, save this as 'app.py' and execute 'python app.py' or use a WSGI server.")
