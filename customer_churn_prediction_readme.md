# 📉 Customer Churn Prediction

This project builds a machine learning model to predict customer churn using the popular Telco Customer Churn dataset. It simulates a real-world business case where the goal is to identify customers at risk of leaving so retention strategies can be applied.

## 💡 Objective

Churn prediction is crucial for subscription-based businesses. By identifying customers likely to leave, companies can reduce churn through targeted actions. In this project, we:

- Explore and clean customer data
- Engineer relevant features
- Train a classification model to predict churn
- Evaluate model performance
- Interpret key drivers of churn using feature importance

## 🗂️ Dataset

The dataset is publicly available on Kaggle:\
🔗 [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

It contains 7,000+ customer records with demographic, account, and usage information.

Key features include:

- Demographics: gender, senior citizen, dependents
- Services: phone, internet, streaming
- Contract type, tenure, payment method
- Churn (target variable)

## 🚀 Getting Started

1. Clone this repository:

```bash
git clone https://github.com/wiederstrom/customer-churn-prediction.git
cd customer-churn-prediction
```

2. (Optional) Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the notebook:

```bash
jupyter notebook churn_prediction.ipynb
```

## 📊 Model and Evaluation

We tested models including:

- Logistic Regression
- Random Forest
- Gradient Boosted Trees (XGBoost or similar)

Metrics used:

- Accuracy
- Precision / Recall / F1-score
- ROC AUC

We also include a confusion matrix and key feature importance visualizations.

## 🧠 Insights

- Customers with month-to-month contracts were more likely to churn.
- Electronic check payment method had higher churn rates.
- Longer tenure generally indicated lower churn probability.

## 📁 Project Structure

```
customer-churn-prediction/
├── churn_prediction.ipynb     # Main notebook
├── data/                      # Contains the dataset (not tracked)
├── models/                    # (Optional) trained model files
├── requirements.txt
├── README.md
└── .gitignore
```

## ✍️ Author

**Erik Lunde Wiederstrøm**\
Bachelor in Applied Data Science, 2025\
[LinkedIn](https://linkedin.com/in/wiederstrom)

##

