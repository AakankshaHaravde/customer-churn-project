# 📊 Customer Churn Prediction Dashboard

A Machine Learning and Business Intelligence project developed using Python, Streamlit, Scikit-learn, XGBoost, and Power BI.

This project analyzes telecom customer data, visualizes customer churn behavior, compares machine learning models, predicts churn probability, and provides interactive dashboards for business insights.

---

# 🚀 Features

- Customer Churn Analysis
- Interactive Streamlit Dashboard
- Power BI Dashboard
- Churn Prediction System
- Multiple Machine Learning Model Comparison
- ROC Curve Comparison
- Confusion Matrix Visualization
- Feature Importance Analysis
- Customer Risk Segmentation

---

# 🛠️ Technologies Used

## Programming & ML

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib

## Visualization

- Matplotlib
- Seaborn
- Streamlit
- Power BI

---

# 📂 Dataset

Dataset Used:
Telco Customer Churn Dataset

The dataset contains:

- Customer demographics
- Internet services
- Contract details
- Payment methods
- Monthly charges
- Churn status

---

# 🤖 Machine Learning Models

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest
- XGBoost

---

# 📈 Model Accuracy

| Model | Accuracy |
|---|---|
| Logistic Regression | 77% |
| KNN | 71.9% |
| Random Forest | 75.1% |
| XGBoost | 76% |

---

# 📊 Streamlit Dashboard Features

## Overview

- Dataset Preview
- Dataset Shape
- Churn Distribution

## Analysis

- Churn vs Contract
- Churn vs Internet Service
- Churn vs Payment Method
- Monthly Charges Analysis
- Correlation Heatmap
- ROC Curve Comparison
- Confusion Matrix Heatmaps
- Feature Importance

## Prediction

Users can:

- Enter customer details
- Predict churn probability
- Identify high-risk customers

---

# 📊 Power BI Dashboard

The project also includes a Power BI dashboard for business intelligence and visual analytics.

Dashboard includes:

- Customer Churn Overview
- Contract Type Analysis
- Internet Service Analysis
- Payment Method Analysis
- Churn Rate Visualization
- Customer Risk Insights
- Model Accuracy Comparison

Power BI helps transform machine learning outputs into business-focused insights for better decision-making.

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/your-username/customer-churn-dashboard.git
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Streamlit Application

```bash
streamlit run app.py
```

---

# 📌 Key Insights

- Month-to-month contract customers have higher churn rates.
- Fiber optic users show higher churn probability.
- Customers with high monthly charges are more likely to churn.
- Long-term contracts significantly reduce churn.

---

# 📁 Project Structure

```bash
customer-churn-dashboard/
│
├── app.py
├── README.md
├── requirements.txt
├── customer_churn_dashboard.pbix
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   ├── churn_model.pkl
│   └── scaler.pkl
│
└── outputs/
```

---

# 👨‍💻 Author

Ak

MBA in AI & ML | Computer Engineer

---

# ⭐ Future Improvements

- Deploy on Streamlit Cloud
- Add SHAP Explainability
- Improve UI/UX
- Add Real-time Database Integration
- Add Customer Retention Recommendation System
