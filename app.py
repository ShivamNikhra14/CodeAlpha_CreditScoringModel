from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
try:
    model_path = os.path.join(os.path.dirname(__file__), "credit_scoring_model.pkl")
    print("📦 Loading model from:", model_path)
    model = joblib.load(model_path)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

# Input features expected by the model
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

        # Default missing values to 0
        for feat in FEATURE_NAMES:
            if feat not in data:
                data[feat] = 0

        # Calculate engineered features
        total_past_due = (
            data["NumberOfTime30-59DaysPastDueNotWorse"] +
            data["NumberOfTimes90DaysLate"] +
            data["NumberOfTime60-89DaysPastDueNotWorse"]
        )

        open_credit = max(data["NumberOfOpenCreditLinesAndLoans"], 1)
        dependents = data["NumberOfDependents"] + 1

        data["DelinquencyRatio"] = total_past_due / open_credit
        data["IncomePerDependent"] = data["MonthlyIncome"] / dependents
        data["TotalPastDue"] = total_past_due

        # Prepare ordered input array
        input_data = [data[feat] for feat in FEATURE_NAMES]
        input_array = np.array([input_data])

        prediction = model.predict(input_array)[0]
        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        print("❌ Error during prediction:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ["PORT"])
    print(f"🚀 Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
