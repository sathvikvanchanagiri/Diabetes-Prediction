

import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("diabetes_logistic_model.pkl")

st.title("Diabetes Prediction")

st.write("Enter the patient details below to predict the diabetes outcome.")

# User inputs
Pregnancies = st.number_input("Pregnancies", min_value=0, value=1)
Glucose = st.number_input("Glucose", min_value=0.0, value=120.0)
BloodPressure = st.number_input("Blood Pressure", min_value=0.0, value=70.0)
SkinThickness = st.number_input("Skin Thickness", min_value=0.0, value=20.0)
Insulin = st.number_input("Insulin", min_value=0.0, value=80.0)
BMI = st.number_input("BMI", min_value=0.0, value=25.0)
DiabetesPedigreeFunction = st.number_input(
    "Diabetes Pedigree Function", min_value=0.0, value=0.5
)
Age = st.number_input("Age", min_value=1, value=30)

# Create input dataframe
input_data = pd.DataFrame({
    "Pregnancies": [Pregnancies],
    "Glucose": [Glucose],
    "BloodPressure": [BloodPressure],
    "SkinThickness": [SkinThickness],
    "Insulin": [Insulin],
    "BMI": [BMI],
    "DiabetesPedigreeFunction": [DiabetesPedigreeFunction],
    "Age": [Age]
})

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("The model predicts: Diabetes")
    else:
        st.success("The model predicts: No Diabetes")