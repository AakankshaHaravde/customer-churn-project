from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# MODELS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ----------------------------
# PATH SETUP
# ----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# ----------------------------
# LOAD FILES
# ----------------------------

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

df = pd.read_csv(DATA_PATH)

# ----------------------------
# CLEAN DATA
# ----------------------------

if df["TotalCharges"].dtype == "object":
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

df = df.dropna()

# ----------------------------
# PREPARE DATA
# ----------------------------

df_model = df.copy()

df_model['Churn'] = df_model['Churn'].map({
    'Yes': 1,
    'No': 0
})

# Drop unnecessary columns
if 'customerID' in df_model.columns:
    df_model.drop('customerID', axis=1, inplace=True)

# Encoding
df_encoded = pd.get_dummies(df_model, drop_first=True)

# Features + Target
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scale
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------
# TRAIN MODELS
# ----------------------------

# Logistic Regression
lr_model = LogisticRegression(max_iter=2000)

lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
y_prob_lr = lr_model.predict_proba(X_test)[:,1]

# KNN
knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)
y_prob_knn = knn_model.predict_proba(X_test)[:,1]

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:,1]

# XGBoost
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:,1]

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="Customer Churn Dashboard",
    layout="wide"
)

st.title("Customer Churn Prediction Dashboard")

# ----------------------------
# SIDEBAR
# ----------------------------

section = st.sidebar.radio(
    "Choose Section",
    ["Overview", "Analysis", "Prediction"]
)

# ----------------------------
# OVERVIEW
# ----------------------------

if section == "Overview":

    st.subheader("Dataset Overview")

    st.write(df.head())

    st.write("Dataset Shape:", df.shape)

    st.subheader("Churn Distribution")

    st.write(df["Churn"].value_counts())

# ----------------------------
# ANALYSIS
# ----------------------------

elif section == "Analysis":

    st.title("Customer Churn Analysis")

    # ----------------------------
    # BASIC GRAPHS
    # ----------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Churn vs Contract")

        fig, ax = plt.subplots()

        sns.countplot(
            x="Contract",
            hue="Churn",
            data=df,
            ax=ax
        )

        plt.xticks(rotation=45)

        st.pyplot(fig)

    with col2:

        st.subheader("Churn vs Internet Service")

        fig, ax = plt.subplots()

        sns.countplot(
            x="InternetService",
            hue="Churn",
            data=df,
            ax=ax
        )

        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("Churn vs Payment Method")

        fig, ax = plt.subplots()

        sns.countplot(
            x="PaymentMethod",
            hue="Churn",
            data=df,
            ax=ax
        )

        plt.xticks(rotation=45)

        st.pyplot(fig)

    with col4:

        st.subheader("Monthly Charges vs Churn")

        fig, ax = plt.subplots()

        sns.boxplot(
            x="Churn",
            y="MonthlyCharges",
            data=df,
            ax=ax
        )

        st.pyplot(fig)

    # ----------------------------
    # TENURE GROUP
    # ----------------------------

    st.subheader("Tenure Group vs Churn")

    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0,12,24,48,72],
        labels=['0-1 yr','1-2 yr','2-4 yr','4-6 yr']
    )

    temp = pd.crosstab(
        df['tenure_group'],
        df['Churn'],
        normalize='index'
    ) * 100

    fig, ax = plt.subplots()

    temp.plot(kind='bar', ax=ax)

    plt.xticks(rotation=0)

    st.pyplot(fig)

    # ----------------------------
    # PIE + COUNT
    # ----------------------------

    col5, col6 = st.columns(2)

    with col5:

        st.subheader("Churn Distribution Pie Chart")

        churn_counts = df["Churn"].value_counts()

        fig, ax = plt.subplots()

        ax.pie(
            churn_counts,
            labels=churn_counts.index,
            autopct='%1.1f%%'
        )

        st.pyplot(fig)

    with col6:

        st.subheader("Churn Count")

        fig, ax = plt.subplots()

        sns.countplot(
            x="Churn",
            data=df,
            ax=ax
        )

        st.pyplot(fig)

    # ----------------------------
    # CORRELATION HEATMAP
    # ----------------------------

    st.header("Correlation Heatmap")

    temp_df = df.copy()

    temp_df['Churn'] = temp_df['Churn'].map({
        'Yes':1,
        'No':0
    })

    num_cols = [
        'tenure',
        'MonthlyCharges',
        'TotalCharges',
        'SeniorCitizen',
        'Churn'
    ]

    corr = temp_df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(8,6))

    sns.heatmap(
        corr,
        annot=True,
        cmap='Blues',
        fmt='.2f',
        ax=ax
    )

    st.pyplot(fig)

    # ----------------------------
    # RISK SEGMENTATION
    # ----------------------------

    st.header("Customer Risk Segmentation")

    def segment_customer(row):

        if (
            row['Contract'] == 'Month-to-month'
            and row['tenure'] < 12
            and row['MonthlyCharges'] > 70
        ):
            return 'High Risk'

        elif row['tenure'] < 24:
            return 'Medium Risk'

        else:
            return 'Low Risk'

    temp_df['RiskSegment'] = temp_df.apply(
        segment_customer,
        axis=1
    )

    fig, ax = plt.subplots()

    sns.countplot(
        x='RiskSegment',
        data=temp_df,
        order=['High Risk','Medium Risk','Low Risk'],
        ax=ax
    )

    st.pyplot(fig)

    # ----------------------------
    # CONFUSION MATRICES
    # ----------------------------

    st.header("Confusion Matrix Heatmaps")

    col7, col8 = st.columns(2)

    with col7:

        cm_lr = confusion_matrix(y_test, y_pred_lr)

        fig, ax = plt.subplots()

        sns.heatmap(
            cm_lr,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax
        )

        ax.set_title("Logistic Regression")

        st.pyplot(fig)

    with col8:

        cm_knn = confusion_matrix(y_test, y_pred_knn)

        fig, ax = plt.subplots()

        sns.heatmap(
            cm_knn,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax
        )

        ax.set_title("KNN")

        st.pyplot(fig)

    col9, col10 = st.columns(2)

    with col9:

        cm_rf = confusion_matrix(y_test, y_pred_rf)

        fig, ax = plt.subplots()

        sns.heatmap(
            cm_rf,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax
        )

        ax.set_title("Random Forest")

        st.pyplot(fig)

    with col10:

        cm_xgb = confusion_matrix(y_test, y_pred_xgb)

        fig, ax = plt.subplots()

        sns.heatmap(
            cm_xgb,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax
        )

        ax.set_title("XGBoost")

        st.pyplot(fig)

    # ----------------------------
    # ROC CURVES
    # ----------------------------

    st.header("ROC Curve Comparison")

    fig, ax = plt.subplots(figsize=(7,5))

    # Logistic
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    auc_lr = auc(fpr_lr, tpr_lr)

    ax.plot(
        fpr_lr,
        tpr_lr,
        label=f'Logistic AUC = {auc_lr:.2f}'
    )

    # KNN
    fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
    auc_knn = auc(fpr_knn, tpr_knn)

    ax.plot(
        fpr_knn,
        tpr_knn,
        label=f'KNN AUC = {auc_knn:.2f}'
    )

    # RF
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    auc_rf = auc(fpr_rf, tpr_rf)

    ax.plot(
        fpr_rf,
        tpr_rf,
        label=f'RF AUC = {auc_rf:.2f}'
    )

    # XGB
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
    auc_xgb = auc(fpr_xgb, tpr_xgb)

    ax.plot(
        fpr_xgb,
        tpr_xgb,
        label=f'XGB AUC = {auc_xgb:.2f}'
    )

    ax.plot([0,1],[0,1],'--')

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    ax.legend()

    st.pyplot(fig)

    # ----------------------------
    # MODEL ACCURACY
    # ----------------------------

    st.header("Model Accuracy Comparison")

    models = [
        'Logistic',
        'KNN',
        'Random Forest',
        'XGBoost'
    ]

    accuracies = [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_knn),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_xgb)
    ]

    fig, ax = plt.subplots(figsize=(8,5))

    sns.barplot(
        x=models,
        y=accuracies,
        ax=ax
    )

    ax.set_ylim(0,1)

    st.pyplot(fig)

    # ----------------------------
    # FEATURE IMPORTANCE
    # ----------------------------

    st.header("Top Features Influencing Churn")

    importance = pd.Series(
        xgb_model.feature_importances_,
        index=X.columns
    )

    importance = importance.sort_values(
        ascending=False
    ).head(10)

    fig, ax = plt.subplots(figsize=(8,5))

    sns.barplot(
        x=importance.values,
        y=importance.index,
        ax=ax
    )

    st.pyplot(fig)

# ----------------------------
# PREDICTION
# ----------------------------

elif section == "Prediction":

    st.subheader("Predict Customer Churn")

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer",
            "Credit card"
        ]
    )

    tenure = st.slider(
        "Tenure (months)",
        0,
        72,
        12
    )

    monthly_charges = st.slider(
        "Monthly Charges",
        0,
        150,
        70
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    internet = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    if st.button("Predict"):

        input_data = pd.DataFrame({

            'tenure': [tenure],

            'MonthlyCharges': [monthly_charges],

            'gender_Male': [
                1 if gender == 'Male' else 0
            ],

            'Partner_Yes': [
                1 if partner == 'Yes' else 0
            ],

            'PaymentMethod_Credit card (automatic)': [
                1 if payment == 'Credit card' else 0
            ],

            'PaymentMethod_Electronic check': [
                1 if payment == 'Electronic check' else 0
            ],

            'PaymentMethod_Mailed check': [
                1 if payment == 'Mailed check' else 0
            ],

            'Contract_One year': [
                1 if contract == 'One year' else 0
            ],

            'Contract_Two year': [
                1 if contract == 'Two year' else 0
            ],

            'InternetService_Fiber optic': [
                1 if internet == 'Fiber optic' else 0
            ],

            'InternetService_No': [
                1 if internet == 'No' else 0
            ],
        })

        # Add missing columns
        for col in scaler.feature_names_in_:

            if col not in input_data.columns:
                input_data[col] = 0

        # Arrange columns properly
        input_data = input_data[
            scaler.feature_names_in_
        ]

        # Scale
        input_scaled = scaler.transform(input_data)

        # Predict
        pred = model.predict(input_scaled)[0]

        prob = model.predict_proba(
            input_scaled
        )[0][1]

        # Output
        if pred == 1:

            st.error(
                f"High Churn Risk ({prob:.2%})"
            )

        else:

            st.success(
                f"Low Churn Risk ({prob:.2%})"
            )
            
            