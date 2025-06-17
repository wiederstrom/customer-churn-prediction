import pandas as pd
from sklearn.preprocessing import StandardScaler

def feature_engineering(df):
    """
    Perform feature engineering on the Telco Customer Churn dataset.

    Steps:
    - Drop irrelevant columns
    - Convert TotalCharges to numeric
    - Encode target variable
    - One-hot encode categorical features
    - Scale numerical features

    Returns:
    - X: Features DataFrame
    - y: Target Series
    """
    df = df.copy()

    # Drop customerID
    df.drop('customerID', axis=1, inplace=True)

    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # One-hot encode categorical variables
    categorical_cols = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Scale numeric features
    scaler = StandardScaler()
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Split features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    return X, y
