from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('models/models.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    # Parse the input JSON
    data = request.get_json(force=True)
    prediction = model.predict([data['features']])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
