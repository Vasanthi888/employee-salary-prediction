import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load your dataset
df = pd.read_csv("data/employees.csv")

# Features and Target
X = df.drop("Salary", axis=1)
y = df["Salary"]

# Define categorical and numeric columns
categorical_features = ["Education Level", "Job Role", "Company Size", "Location"]
numeric_features = ["Experience"]

# Preprocessing: OneHotEncoding for categoricals
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
], remainder="passthrough")  # Keep numeric as-is

# Build ML Pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train the model
model.fit(X, y)

# Save the trained model
joblib.dump(model, "models/salary_model.pkl")

print("✅ Model trained and saved successfully to 'models/salary_model.pkl'")
