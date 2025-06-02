# CodeAlpha_CreditScoringModel

A machine learning project to predict an individual's creditworthiness based on historical financial data using classification algorithms, with a focus on Decision Tree Classifier.

## 📌 Project Objective

The goal of this project is to build a model that predicts whether an individual is likely to default on a loan using past financial records. This helps financial institutions assess risk and make informed lending decisions.

## 🧠 Algorithms Explored

- Logistic Regression (Exploratory)
- Random Forest Classifier (Exploratory)
- ✅ Decision Tree Classifier (Final model)

## 📊 Dataset

The dataset used is from the "Give Me Some Credit" Kaggle competition, which includes:

- **Target column:** `SeriousDlqin2yrs` (1 = default, 0 = no default)
- **Features include:**  
  - `RevolvingUtilizationOfUnsecuredLines`
  - `DebtRatio`
  - `MonthlyIncome`
  - `NumberOfOpenCreditLinesAndLoans`
  - `NumberOfTimes90DaysLate`
  - `NumberOfDependents`
  - And more

### 📂 Files

- `cs-training.csv`: Training dataset
- `cs-test.csv`: Test dataset (not used for final model evaluation)
- `credit_scoring_model.ipynb`: Notebook with complete preprocessing, training, and evaluation
- `credit_scoring_model.pkl`: Saved trained model

## 🛠️ Feature Engineering

New features created to improve model performance:

1. **DelinquencyRatio** – Ratio of late payments to total open credit lines
2. **IncomePerDependent** – Monthly income adjusted for number of dependents
3. **TotalPastDue** – Combined count of all past due payments

## ⚙️ Model Pipeline

- **Preprocessing**:  
  - Median imputation for missing values  
  - Standard scaling

- **Model**:  
  - `DecisionTreeClassifier` with hyperparameter tuning via `GridSearchCV`

## 📈 Evaluation Metrics

Evaluated using:
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC AUC Score

### 🧪 Best Results
- ROC AUC Score: **~0.93**
- Improved F1-score for minority class after feature engineering and hyperparameter tuning.

## 💾 Model Saving

Model saved using `joblib`:

```python
import joblib
joblib.dump(best_model, 'credit_scoring_model.pkl')

