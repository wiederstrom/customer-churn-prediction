from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.2, stratify=True, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Parameters:
    - X: pandas DataFrame, feature matrix
    - y: pandas Series, target vector
    - test_size: float, proportion of dataset to include in test split (default=0.2)
    - stratify: bool, whether to stratify split based on target (default=True)
    - random_state: int, controls shuffling applied to the data before split

    Returns:
    - X_train, X_test, y_train, y_test: split feature matrices and targets
    """
    stratify_arg = y if stratify else None
    return train_test_split(
        X, y, test_size=test_size,
        stratify=stratify_arg,
        random_state=random_state
    )
