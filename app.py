import streamlit as st
import pickle
import numpy as np
import os

# Title
st.title("❤️ Heart Disease Prediction")

# Load model
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "gb_model.pkl")   # ✅ make sure filename matches

    model = pickle.load(open(model_path, "rb"))
    st.success("Model loaded successfully")

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# ---------------- INPUTS ---------------- #

age = st.number_input("Age", min_value=1, max_value=120, value=30)
bp = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Cholesterol", value=200)
hr = st.number_input("Max Heart Rate", value=150)
oldpeak = st.number_input("Oldpeak", value=1.0)

sex = st.selectbox("Sex", ["Male", "Female"])
angina = st.selectbox("Exercise Angina", ["Yes", "No"])

# Encoding
sex = 1 if sex == "Male" else 0
angina = 1 if angina == "Yes" else 0


# ---------------- PREDICTION ---------------- #

if st.button("Predict"):
    try:
        data = np.array([[age, bp, chol, hr, oldpeak, sex, angina]])
        result = model.predict(data)

        if result[0] == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
