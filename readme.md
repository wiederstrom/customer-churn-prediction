# 💡 Customer Churn Prediction

This project trains and evaluates multiple machine learning models to predict customer churn in a subscription-based business using the Telco Customer Churn dataset.

## 🎯 Objective

The goal is to identify customers who are likely to cancel their subscription (churn), allowing the company to take preventive actions.

The project includes:
- Data cleaning and exploration
- Feature engineering
- Model training with hyperparameter tuning (GridSearchCV)
- Model evaluation and visualization
- Modular pipeline structure for reproducibility

## 📦 Dataset

Source: [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset includes:
- Demographics: Gender, SeniorCitizen, Partner, Dependents
- Services: Phone, Internet, Streaming, etc.
- Account info: Tenure, Contract, PaymentMethod, MonthlyCharges
- Target: `Churn`

## 🧠 Models Used

- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Machine (SVM)

Each model is trained with `GridSearchCV` and evaluated using:
- F1 Score
- ROC AUC
- Confusion Matrix
- Classification Report

## 🧱 Project Structure
```
customer-churn-prediction/
├── data/
│   └── raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_model_evaluation.ipynb
├── outputs/
│   └── models/  # Saved .pkl model files
├── src/
│   ├── features/feature_engineering.py
│   ├── models/GridCV_all_model.py
│   └── utils/
│       ├── split_data.py
│       └── model_helpers.py
├── README.md
└── requirements.txt
```

## 👤 Author
**Erik Lunde Wiederstrøm**  
Bachelor in Applied Data Science (2025)  
[LinkedIn →](https://linkedin.com/in/wiederstrom)

