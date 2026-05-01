import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Configure page
st.set_page_config(page_title="Bank Loan Classifier", layout="wide")

# Load model and preprocessing tools
@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, scaler, le

model, scaler, le = load_model()

# App title and description
st.title("🏦 Bank Loan Classification Predictor")
st.markdown("---")
st.write("""
This application uses machine learning to predict whether a loan will be approved or not.
Enter the applicant's information below to get a prediction.
""")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    credit_score = st.slider(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650,
        step=10
    )

with col2:
    loan_amount = st.slider(
        "Loan Amount ($)",
        min_value=5000,
        max_value=50000,
        value=25000,
        step=1000
    )

col3, col4 = st.columns(2)

with col3:
    years_employed = st.slider(
        "Years Employed",
        min_value=0,
        max_value=30,
        value=5,
        step=1
    )

with col4:
    employment_type = st.selectbox(
        "Employment Type",
        options=["Salaried", "Self-Employed"]
    )

# Make prediction
st.markdown("---")

if st.button("🔮 Predict Loan Approval", use_container_width=True):
    # Encode employment type
    employment_type_encoded = le.transform([employment_type])[0]
    
    # Create input array
    input_data = np.array([[
        credit_score,
        loan_amount,
        years_employed,
        employment_type_encoded
    ]])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    # Display results
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if prediction == 1:
            st.success("✅ LOAN APPROVED", icon="✓")
        else:
            st.error("❌ LOAN REJECTED", icon="✗")
    
    with col2:
        approval_prob = prediction_proba[1] * 100
        st.metric("Approval Probability", f"{approval_prob:.2f}%")
    
    with col3:
        rejection_prob = prediction_proba[0] * 100
        st.metric("Rejection Probability", f"{rejection_prob:.2f}%")
    
    # Show input summary
    st.markdown("---")
    st.subheader("📋 Input Summary")
    summary_df = pd.DataFrame({
        'Parameter': ['Credit Score', 'Loan Amount', 'Years Employed', 'Employment Type'],
        'Value': [credit_score, f"${loan_amount:,}", years_employed, employment_type]
    })
    st.table(summary_df)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p style='color: gray;'>Bank Loan Classification System | Powered by ML Pipeline</p>
</div>
""", unsafe_allow_html=True)
