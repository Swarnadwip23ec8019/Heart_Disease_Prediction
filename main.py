# main.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

# Load saved model and scaler
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About", "Disease Prediction"])

# HOME PAGE
if page == "Home":
    st.title("❤️ Heart Disease Prediction App")
    st.image("heart_health.jpg", use_container_width=True)
    st.write("""
    This web app predicts the **likelihood of heart disease** based on user input.
    
    **Steps to use:**
    1. Go to the *Disease Prediction* section.
    2. Enter patient health parameters.
    3. Get an instant prediction result.
    """)

# ABOUT PAGE
elif page == "About":
    st.title("ℹ️ About the Project")
    st.write("""
    This machine learning model uses **Logistic Regression** trained on a heart disease dataset.
    It classifies whether a person **has heart disease (1)** or **does not (0)** based on medical indicators.
    
    **Dataset Features:**
    - Age, Sex, Chest Pain Type, Resting BP, Cholesterol  
    - Fasting Blood Sugar, Rest ECG, Max Heart Rate  
    - Exercise-induced Angina, Oldpeak, Slope, Major Vessels, Thal
    
    **Model:** Logistic Regression  
    **Frameworks:** Scikit-learn, Streamlit  
    """)

# PREDICTION PAGE
elif page == "Disease Prediction":
    st.title("🩺 Heart Disease Prediction")
    st.subheader("Enter the following details:")

    # Input fields
    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.number_input("Serum Cholestoral (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [1, 0])
    restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
    oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
    ca = st.selectbox("Major Vessels Colored by Fluoroscopy (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (0 = Normal, 1 = Fixed defect, 2 = Reversible defect)", [0, 1, 2])

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    if st.button("Predict"):
        prediction = model.predict(input_data_scaled)
        if prediction[0] == 1:
            st.error("⚠️ The person has a Heart Disease")
        else:
            st.success("✅ The person does NOT have a Heart Disease")
