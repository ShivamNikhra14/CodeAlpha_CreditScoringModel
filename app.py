from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the saved model
model = joblib.load("credit_scoring_model.pkl")

# Define the expected feature names (same order as used during training)
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

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_data = [data.get(feat, 0) for feat in FEATURE_NAMES]  # default to 0 if missing
        prediction = model.predict([input_data])[0]
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/", methods=["GET"])
def home():
    return "Credit Scoring Model API is running."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Required for Render
    app.run(host="0.0.0.0", port=port, debug=True)
