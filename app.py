import streamlit as st
import pandas as pd
import joblib

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)

# -----------------------------------------------------
# LOAD MODEL & PREPROCESSOR
# -----------------------------------------------------

model = joblib.load("xgb_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# -----------------------------------------------------
# TITLE
# -----------------------------------------------------

st.title(" AI Loan Approval Predictor")

st.write(
    """
    Predict whether a loan application is likely to be approved
    based on the applicant's financial and credit information.
    """
)

st.divider()
# -----------------------------------------------------
# USER INPUT
# -----------------------------------------------------

col1, col2 = st.columns(2)

# =====================================================
# LEFT COLUMN
# =====================================================

with col1:

    st.subheader(" Applicant Information")

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    occupation_status = st.selectbox(
        "Occupation Status",
        ["Employed", "Self-Employed", "Student"]
    )

    years_employed = st.number_input(
        "Years Employed",
        min_value=0,
        max_value=50,
        value=5
    )

    annual_income = st.number_input(
        "Annual Income",
        min_value=0.0,
        value=50000.0
    )

    savings_assets = st.number_input(
        "Savings / Assets",
        min_value=0.0,
        value=10000.0
    )

    current_debt = st.number_input(
        "Current Debt",
        min_value=0.0,
        value=5000.0
    )


# =====================================================
# RIGHT COLUMN
# =====================================================

with col2:

    st.subheader(" Credit & Loan Information")

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=700
    )

    credit_history_years = st.number_input(
        "Credit History (Years)",
        min_value=0,
        max_value=50,
        value=8
    )

    defaults_on_file = st.number_input(
        "Defaults on File",
        min_value=0,
        value=0
    )

    delinquencies_last_2yrs = st.number_input(
        "Delinquencies (Last 2 Years)",
        min_value=0,
        value=0
    )

    derogatory_marks = st.number_input(
        "Derogatory Marks",
        min_value=0,
        value=0
    )

    product_type = st.selectbox(
        "Loan Product",
        [
            "Personal Loan",
            "Credit Card",
            "Line of Credit"
        ]
    )

    loan_intent = st.selectbox(
        "Loan Purpose",
        [
            "Business",
            "Debt Consolidation",
            "Education",
            "Home Improvement",
            "Medical",
            "Personal"
        ]
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=10000.0
    )

    interest_rate = st.number_input(
        "Interest Rate (%)",
        min_value=0.0,
        value=10.5
    )

    debt_to_income_ratio = st.number_input(
        "Debt-to-Income Ratio",
        min_value=0.0,
        value=0.30,
        step=0.01
    )

    loan_to_income_ratio = st.number_input(
        "Loan-to-Income Ratio",
        min_value=0.0,
        value=0.20,
        step=0.01
    )

    payment_to_income_ratio = st.number_input(
        "Payment-to-Income Ratio",
        min_value=0.0,
        value=0.10,
        step=0.01
    )
# -----------------------------------------------------
# PREDICT BUTTON
# -----------------------------------------------------

st.divider()

if st.button("🔍 Predict Loan Eligibility", use_container_width=True):

    # Create input dataframe
    input_data = pd.DataFrame({
        'age': [age],
        'occupation_status': [occupation_status],
        'years_employed': [years_employed],
        'annual_income': [annual_income],
        'credit_score': [credit_score],
        'credit_history_years': [credit_history_years],
        'savings_assets': [savings_assets],
        'current_debt': [current_debt],
        'defaults_on_file': [defaults_on_file],
        'delinquencies_last_2yrs': [delinquencies_last_2yrs],
        'derogatory_marks': [derogatory_marks],
        'product_type': [product_type],
        'loan_intent': [loan_intent],
        'loan_amount': [loan_amount],
        'interest_rate': [interest_rate],
        'debt_to_income_ratio': [debt_to_income_ratio],
        'loan_to_income_ratio': [loan_to_income_ratio],
        'payment_to_income_ratio': [payment_to_income_ratio]
    })

    # Apply preprocessing
    processed_data = preprocessor.transform(input_data)

    # Prediction
    prediction = model.predict(processed_data)[0]

    # Prediction Probability
    probability = model.predict_proba(processed_data)[0][1]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:

        st.success(" Based on the provided information, the applicant is likely to be eligible for loan approval.")

    else:

        st.error(" Based on the provided information, the applicant is unlikely to be eligible for loan approval.")

    st.metric(
        label="Approval Probability",
        value=f"{probability:.2%}"
    )

    # Risk Level
    if probability >= 0.80:
        st.success("🟢 Risk Level: Low")

    elif probability >= 0.60:
        st.warning("🟡 Risk Level: Moderate")

    else:
        st.error("🔴 Risk Level: High")