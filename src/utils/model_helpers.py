import joblib
from pathlib import Path
from sklearn.model_selection import GridSearchCV

def run_grid_search(model, param_grid, X_train, y_train, cv=5, scoring='f1'):
    """
    Runs a GridSearchCV with the given model and parameters.

    Parameters:
    - model: estimator object
    - param_grid: dictionary of parameters to search
    - X_train: training feature data
    - y_train: training labels
    - cv: cross-validation folds
    - scoring: metric to optimize

    Returns:
    - Trained GridSearchCV object
    """
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_train, y_train)
    return grid_search

def save_model(model, path):
    """Saves the model to the specified path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Saved model to: {path}")
