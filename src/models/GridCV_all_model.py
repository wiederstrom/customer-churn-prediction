import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report

from src.features.feature_engineering import feature_engineering
from src.utils.split_data import split_data
from src.utils.model_helpers import run_grid_search, save_model

df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
X, y = feature_engineering(df)
X_train, X_test, y_train, y_test = split_data(X, y)

rf_parameters={
    'n_estimators': [50,100,150,200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2,5,10],
    'max_features': ['sqrt', 'log2'] 
}

xgb_parameters={
    'max_depth': [3,5,7],
    'learning_rate': [0.1, 0.01, 0.001],
    'subsample': [0.5, 0.7, 1]

}

svm_parameters={
    'kernels': ['linear', 'rbf', 'poly'],
    'gammas': [0.1, 1, 10, 100],
    'C': [0.1, 1, 10, 100, 1000],
    'degrees': [0, 1, 2, 3, 4, 5, 6]

}

rf_model = RandomForestClassifier(random_state=42)
xgb_model = XGBClassifier()
svm_model = svm.SVC()



models = {
    'Random Forest': {
        'model': rf_model,
        'param': rf_parameters
        },
    'XGBoost': {
        'model': xgb_model,
        'param': xgb_parameters
    },
    'Support Vector Machine': {
        'model': svm_model,
        'param': svm_parameters
    }
}


for name, (model, params) in models.items():
    print(f"\nRunning GridSearchCV for: {name}")
    grid = run_grid_search(model, params, X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    print(f"Best params: {grid.best_params_}")
    print(f"F1 score: {f1:.3f}")
    print(classification_report(y_test, y_pred))

    save_model(best_model, f"outputs/models/{name.lower().replace(' ', '_')}_tuned.pkl")