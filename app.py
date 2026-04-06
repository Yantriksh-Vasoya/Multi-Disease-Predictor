import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np

from src.prediction import load_model, predict

# ---------------------------------------
# Load Models
# ---------------------------------------

heart_model = load_model("models/heart_model.pkl")
diabetes_model = load_model("models/diabetes_model.pkl")
kidney_model = load_model("models/kidney_model.pkl")

# ---------------------------------------
# Page Setup
# ---------------------------------------

st.set_page_config(
    page_title="Multi Disease Prediction",
    layout="wide"
)

st.title("🩺 Multi Disease Prediction System")

st.write("""
Predict the risk of **Heart Disease, Diabetes, and Kidney Disease**
using Machine Learning.
""")

# ---------------------------------------
# Sidebar Navigation
# ---------------------------------------
with st.sidebar:
    # Adds a medical-themed icon for professional branding
    st.image("https://cdn-icons-png.flaticon.com/512/2864/2864248.png", width=100) 
    st.title("Navigation Menu")
    menu = st.selectbox(
        "Choose a Prediction Tool:",
        ["🏠 Home", "❤️ Heart Disease", "🍬 Diabetes", "🧬 Kidney Disease", "📊 Model Comparison"]
    )

# ==========================================================
# HOME PAGE
# ==========================================================

if menu == "🏠 Home":

    st.header("Project Overview")

    st.write("""
This project predicts diseases using machine learning models.

Diseases included:

• Heart Disease  
• Diabetes  
• Kidney Disease
""")

# ==========================================================
# HEART DISEASE
# ==========================================================

elif menu == "❤️ Heart Disease":

    st.header("❤️ Heart Disease Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 1, 120, help="Age of the patient in years")
        sex = st.selectbox("Sex", [0, 1], help="1 = Male, 0 = Female")
        cp = st.number_input("Chest Pain Type", 0, 3, help="Value 0-3 representing chest pain severity")

    with col2:
        chol = st.number_input("Cholesterol", help="Serum cholesterol in mg/dl")
        thalach = st.number_input("Max Heart Rate", help="Maximum heart rate achieved")
        exang = st.number_input("Exercise Induced Angina", 0, 1, help="1 = Yes, 0 = No")

    with col3:
        oldpeak = st.number_input("ST Depression", help="ST depression induced by exercise relative to rest")
        slope = st.number_input("Slope", 0, 2, help="The slope of the peak exercise ST segment (0-2)")

    input_data = [age, sex, cp, chol, thalach, exang, oldpeak, slope]

    if st.button("Predict Heart Disease"):
        prediction, probability = predict(heart_model, input_data)
        
        st.divider() # Adds a clean visual line
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            if prediction == 1:
                st.error("⚠ High Risk of Heart Disease")
            else:
                st.success("✅ Healthy Heart")
        
        with col_res2:
            if probability:
                st.metric(label="Confidence Level", value=f"{round(probability*100, 2)}%")

# ==========================================================
# DIABETES
# ==========================================================

elif menu == "🍬 Diabetes":

    st.header("🍬 Diabetes Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, help="Number of times pregnant")
        glucose = st.number_input("Glucose", 0, 500, help="Plasma glucose concentration")

    with col2:
        bmi = st.number_input("BMI", 0.0, 70.0, help="Body Mass Index (Weight in kg / Height in m^2)")
        age = st.number_input("Age", 1, 120, help="Age of the patient in years")

    with col3:
        insulin = st.number_input("Insulin")
        bmi = st.number_input("BMI")
        dpf = st.number_input("Diabetes Pedigree Function")

    input_data = [pregnancies, glucose, bmi, age]

    if st.button("Predict Diabetes"):
        prediction, probability = predict(diabetes_model, input_data)
        
        st.divider() # Adds a clean horizontal line for separation
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            if prediction == 1:
                st.error("⚠ High Risk Detected")
            else:
                st.success("✅ Low Risk Detected")
        
        with col_res2:
            if probability:
                st.metric(label="Confidence Level", value=f"{round(probability*100, 2)}%")

# ==========================================================
# KIDNEY DISEASE
# ==========================================================

elif menu == "🧬 Kidney Disease":

    st.header("🧬 Kidney Disease Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        bp = st.number_input("Blood Pressure", help="Blood pressure in mm/Hg")
        sg = st.number_input("Specific Gravity", help="Urine specific gravity (e.g., 1.010, 1.020)")
        al = st.number_input("Albumin", 0, 5, help="Albumin level in urine (0-5)")

    with col2:
        su = st.number_input("Sugar", 0, 5, help="Sugar level in urine (0-5)")
        bgr = st.number_input("Blood Glucose Random", help="Random blood glucose in mgs/dl")
        bu = st.number_input("Blood Urea", help="Blood urea level in mgs/dl")

    with col3:
        sc = st.number_input("Serum Creatinine", help="Serum creatinine level in mgs/dl")
        hemo = st.number_input("Hemoglobin", help="Hemoglobin level in gms")

    input_data = [bp,sg,al,su,bgr,bu,sc,hemo]

    if st.button("Predict Kidney Disease"):
        prediction, probability = predict(kidney_model, input_data)
        
        st.divider()
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            if prediction == 1:
                st.error("⚠ Kidney Disease Detected")
            else:
                st.success("✅ Kidney Healthy")
        
        with col_res2:
            if probability:
                st.metric(label="Confidence Level", value=f"{round(probability*100, 2)}%")

# ==========================================================
# MODEL COMPARISON DASHBOARD
# ==========================================================

elif menu == "📊 Model Comparison":

    st.header("📊 Model Accuracy Comparison")

    data = {

        "Model":[
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "SVM",
            "Gradient Boosting"
        ],

        "Heart Accuracy":[0.88,0.84,0.92,0.90,0.91],

        "Diabetes Accuracy":[0.82,0.80,0.86,0.84,0.85],

        "Kidney Accuracy":[0.95,0.93,0.97,0.96,0.97]
    }

    df = pd.DataFrame(data)

    # Highlights the highest accuracy in each column in green
    st.table(df.style.highlight_max(axis=0, color='#90ee90')) 
    
    st.subheader("Visual Performance Comparison")
    st.area_chart(df.set_index("Model")) # Changes bar chart to a modern area chart

    st.bar_chart(df.set_index("Model"))
