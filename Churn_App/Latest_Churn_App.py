import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from alibi_detect.cd import KSDrift

# Load saved model
model = joblib.load("data_model.joblib")

# Define feature columns
numeric_features = ['Age', 'AmountSpent', 'LoginFrequency', 'DateSinceLastLogin']
categorical_features = ['Gender', 'MaritalStatus', 'IncomeLevel', 'ResolutionStatus']

# Load dataset (for drift detection only)
@st.cache_data
def load_data():
    return pd.read_excel("new_data.xlsx")

data = load_data()

# Sidebar navigation
page = st.sidebar.selectbox("Select Page", ["Homepage", "Prediction & Explanation", "ML Monitoring (Data Drift)"])

# ----------------------------------------------
# PAGE 1: HOMEPAGE
# ----------------------------------------------
if page == "Homepage":
    st.title("🏦 Lloyds Bank - Customer Churn Model")
    st.markdown("Welcome to the churn model app. This tool helps predict customer churn and monitor model drift.")

    st.subheader("📌 Model Information")
    st.markdown("""
    - **Algorithm**: Random Forest
    - **Features**: Preprocessed numeric and categorical data
    - **Use Case**: Predict if a customer is likely to churn
    """)

    st.subheader("📊 Data Sample")
    st.dataframe(data.head())

# ----------------------------------------------
# PAGE 2: PREDICTION & EXPLANATION
# ----------------------------------------------
elif page == "Prediction & Explanation":
    st.title("🔮 Predict Customer Churn")

    with st.form("prediction_form"):
        Age = st.slider("Age", 18, 100, 35)
        AmountSpent = st.number_input("Amount Spent", 0, 10000, 500)
        LoginFrequency = st.slider("Login Frequency", 0, 100, 10)
        DateSinceLastLogin = st.slider("Days Since Last Login", 0, 365, 10)
        Gender = st.selectbox("Gender", ["Male", "Female"])
        MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        IncomeLevel = st.selectbox("Income Level", ["Low", "Medium", "High"])
        ResolutionStatus = st.selectbox("Resolution Status", ["Resolved", "Pending", "Escalated"])
        submit = st.form_submit_button("Predict")

    if submit:
        user_input = pd.DataFrame([{
            'Age': Age,
            'AmountSpent': AmountSpent,
            'LoginFrequency': LoginFrequency,
            'DateSinceLastLogin': DateSinceLastLogin,
            'Gender': Gender,
            'MaritalStatus': MaritalStatus,
            'IncomeLevel': IncomeLevel,
            'ResolutionStatus': ResolutionStatus
        }])
        prediction = model.predict(user_input)[0]
        probability = model.predict_proba(user_input)[0][1]

        st.success(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
        st.info(f"Churn Probability: {probability:.2f}")

        # Feature Importance
        rf = model.named_steps["classifier"]
        ohe_features = list(model.named_steps["preprocessing_pipeline"]
                            .named_transformers_["categorical"]
                            .named_steps["ohe"]
                            .get_feature_names_out(categorical_features))
        all_features = numeric_features + ohe_features
        importances = rf.feature_importances_

        fig, ax = plt.subplots()
        sns.barplot(x=importances, y=all_features, ax=ax)
        ax.set_title("Feature Importance")
        st.pyplot(fig)

# ----------------------------------------------
# PAGE 3: MONITORING - DRIFT DETECTION
# ----------------------------------------------
elif page == "ML Monitoring (Data Drift)":
    st.title("📉 Drift Detection (Monitoring)")

    st.markdown("Detect data distribution changes using KS Drift Test (Alibi Detect).")

    # Prepare data
    X = data.drop(columns=["ChurnStatus"])
    y = data["ChurnStatus"]
    X_train, X_test = train_test_split(X, test_size=0.2, stratify=y, random_state=42)

    #numeric_data = X.select_dtypes(include=np.number).columns.drop("CustomerID")
    
        # Select numeric columns and exclude 'CustomerID' if it exists
    numeric_cols = X.select_dtypes(include=np.number).columns
    numeric_data = [col for col in numeric_cols if col != "CustomerID"]

    cd = KSDrift(X_train[numeric_data].values, p_val=0.05)
    drift_result = cd.predict(X_test[numeric_data].values)

    drift_df = pd.DataFrame({
        "Feature": numeric_data,
        "p_value": drift_result['data']['p_val'],
        "Drifted": drift_result['data']['is_drift']
    }).sort_values("p_value")

    st.dataframe(drift_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=drift_df, x='p_value', y='Feature', hue='Drifted', palette={True: 'red', False: 'blue'}, dodge=False)
    ax.axvline(x=0.05, color='gray', linestyle='--', label='Significance Threshold (0.05)')
    ax.set_title("KS Drift Detection by Feature")
    ax.set_xlabel("p-value")
    ax.set_ylabel("Feature")
    ax.legend()
    st.pyplot(fig)
