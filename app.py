from flask import Flask, jsonify
import os
import re

app = Flask(__name__)

# Path to accuracy.txt
ACCURACY_FILE_PATH = os.path.join('results', 'bestaccuracy.txt')


@app.route('/')
def home():
    return "Welcome to the ML Model API. Use the /predict endpoint to read accuracy."


@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Read the accuracy value from the file
        with open(ACCURACY_FILE_PATH, 'r') as file:
            content = file.read().strip()

        # Extract the floating-point value using regex
        match = re.search(r'Accuracy:\s*([0-9]*\.[0-9]+)', content)
        if not match:
            return jsonify({"error": "Invalid format in accuracy.txt"}), 400

        # Convert the extracted value to a float
        accuracy = float(match.group(1))

        # Perform any required conversion (e.g., scale to percentage)
        accuracy_percentage = accuracy * 100

        return jsonify({
            "accuracy": accuracy,
            "accuracy_percentage": f"{accuracy_percentage:.2f}%"
        })

    except FileNotFoundError:
        return jsonify({"error": "accuracy.txt not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
