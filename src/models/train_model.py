import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from pathlib import Path

from src.features.feature_engineering import feature_engineering
from src.utils.split_data import split_data
from src.utils.model_helpers import run_grid_search, save_model

# Load data
df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Feature engineering
X, y = feature_engineering(df)

# Train/test split
X_train, X_test, y_train, y_test = split_data(X, y)

# Define parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2']
}

# Initialize model
rf = RandomForestClassifier(class_weight='balanced', random_state=42)

# Run GridSearch
grid_search = run_grid_search(rf, param_grid_rf, X_train, y_train)

# Best model and predictions
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluation
f1 = f1_score(y_test, y_pred)
print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"F1 Score: {f1:.3f}")
print(classification_report(y_test, y_pred))

# Save best model
save_model(best_model, "outputs/models/random_forest_tuned.pkl")
