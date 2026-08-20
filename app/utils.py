from pathlib import Path

import joblib
import pandas as pd


APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = APP_DIR / "model.pkl"
TRANSFORMER_PATH = APP_DIR / "transformer.pkl"


model = joblib.load(MODEL_PATH)
transformer = joblib.load(TRANSFORMER_PATH)


FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "tenure_years",
    "spend_per_month"
]


def prepare_customer(customer):
    """
    Convert one customer JSON object
    into the DataFrame expected by the model.
    """

    row = {}

    for column in FEATURE_COLUMNS:
        row[column] = customer.get(column, None)

    data = pd.DataFrame([row])

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce"
    )

    return data


def predict_customer(customer):
    """
    Generate churn probability and prediction
    for one customer.
    """

    data = prepare_customer(customer)

    transformed_data = transformer.transform(data)

    probability = float(
        model.predict_proba(transformed_data)[0, 1]
    )

    if probability >= 0.5:
        prediction = "Yes"
    else:
        prediction = "No"

    return probability, prediction