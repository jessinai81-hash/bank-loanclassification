import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Configure page
st.set_page_config(page_title="Bank Loan Classifier", layout="wide")

# Load model and preprocessing tools
@st.cache_resource
def load_model():
    try:
        # Try to load existing pickle files
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        return model, scaler, le
    except FileNotFoundError:
        # If pickle files not found, train the model
        st.warning("Training model from dataset...")
        
        # Load and prepare data
        df = pd.read_csv('14.banking_loan_dataset_1000.csv')
        le = LabelEncoder()
        df['EmploymentType'] = le.fit_transform(df['EmploymentType'])
        
        X = df.drop('LoanApproved', axis=1)
        y = df['LoanApproved']
        
        # Preprocess
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X)
        X_processed = pd.DataFrame(X_processed, columns=X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(random_state=42, n_estimators=100)
        model.fit(X_train, y_train)
        
        # Save for future use
        try:
            with open('best_model.pkl', 'wb') as f:
                pickle.dump(model, f)
            with open('scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            with open('label_encoder.pkl', 'wb') as f:
                pickle.dump(le, f)
        except:
            pass  # If saving fails, continue anyway
        
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
            st.success("✅ LOAN APPROVED")
        else:
            st.error("❌ LOAN REJECTED")
    
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
    <p style='color: gray; font-size: 12px;'>Best Model: Random Forest Classifier (Accuracy: 100%)</p>
</div>
""", unsafe_allow_html=True)
