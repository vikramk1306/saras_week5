import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Load dataset
df = pd.read_csv("gold_churn_data.csv")

# Create X and y
X = df.drop("Churn", axis=1)
y = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# Remove columns that are not model features
X = X.drop(
    columns=["customerID", "Unnamed: 0"],
    errors="ignore"
)

# Convert TotalCharges to numeric
X["TotalCharges"] = pd.to_numeric(
    X["TotalCharges"],
    errors="coerce"
)

# Identify columns
categorical_cols = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_cols = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            SimpleImputer(strategy="mean"),
            numerical_cols
        ),
        (
            "cat",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                )
            ]),
            categorical_cols
        )
    ]
)


# Transform training data
X_transformed = preprocessor.fit_transform(X)


# Train model
model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

model.fit(
    X_transformed,
    y
)


# Save model
joblib.dump(
    model,
    "app/model.pkl"
)

# Save transformer
joblib.dump(
    preprocessor,
    "app/transformer.pkl"
)


print("Model trained successfully!")
print("Model saved to: app/model.pkl")
print("Transformer saved to: app/transformer.pkl")
print("Training rows:", len(X))
print("Features after preprocessing:", X_transformed.shape[1])
