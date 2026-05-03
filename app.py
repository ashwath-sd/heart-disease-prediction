import streamlit as st
import pickle
import numpy as np

# Title
st.title("❤️ Heart Disease Prediction")
st.write("Enter patient details below:")

# Load model & scaler
try:
    model = pickle.load(open("gb_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    st.success("Model & Scaler loaded successfully")

except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()


# ---------------- INPUTS ---------------- #

age = st.number_input("Age", min_value=1, max_value=120, value=30)
resting_bp = st.number_input("Resting Blood Pressure", value=120)
cholesterol = st.number_input("Cholesterol", value=200)
max_hr = st.number_input("Max Heart Rate", value=150)
oldpeak = st.number_input("Oldpeak", value=1.0)

sex = st.selectbox("Sex", ["Male", "Female"])
exercise_angina = st.selectbox("Exercise Angina", ["Yes", "No"])

# Encoding
sex = 1 if sex == "Male" else 0
exercise_angina = 1 if exercise_angina == "Yes" else 0


# ---------------- PREDICTION ---------------- #

if st.button("Predict"):

    try:
        # Numerical data (same order as training)
        input_data = np.array([[age, resting_bp, cholesterol, max_hr, oldpeak]])

        # Apply scaling
        input_scaled = scaler.transform(input_data)

        # Add encoded features
        final_input = np.concatenate((input_scaled, [[sex, exercise_angina]]), axis=1)

        prediction = model.predict(final_input)

        if prediction[0] == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
