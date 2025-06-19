from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from frontend

# Load the model with a safe path
try:
    model_path = os.path.join(os.path.dirname(__file__), "credit_scoring_model.pkl")
    print("📦 Loading model from:", model_path)
    model = joblib.load(model_path)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None  # Avoid crashing on startup

# Define expected input feature names (13 total)
FEATURE_NAMES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
    "DelinquencyRatio",
    "IncomePerDependent",
    "TotalPastDue"
]

@app.route("/", methods=["GET"])
def home():
    return "✅ Credit Scoring Model API is running."

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()

        # Prepare input values in the expected order
        input_data = [data.get(feat, 0) for feat in FEATURE_NAMES]
        input_array = np.array([input_data])

        prediction = model.predict(input_array)[0]
        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        print("❌ Error during prediction:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ["PORT"])  # Render requires this
    print(f"🚀 Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
