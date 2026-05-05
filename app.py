import streamlit as st
import pickle
import pandas as pd

st.title("Heart Disease Prediction")

# Load model + columns
model = pickle.load(open("gb_model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# Inputs (same as your dataset BEFORE encoding)
age = st.number_input("Age", 1, 120, 30)
bp = st.number_input("Resting BP", 120)
chol = st.number_input("Cholesterol", 200)
hr = st.number_input("Max HR", 150)
oldpeak = st.number_input("Oldpeak", 1.0)

sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["ATA","NAP","ASY","TA"])
restecg = st.selectbox("Resting ECG", ["Normal","ST","LVH"])
angina = st.selectbox("Exercise Angina", ["Yes", "No"])
slope = st.selectbox("ST Slope", ["Up","Flat","Down"])

if st.button("Predict"):

    # Create dataframe
    input_dict = {
        "Age": age,
        "RestingBP": bp,
        "Cholesterol": chol,
        "MaxHR": hr,
        "Oldpeak": oldpeak,
        "Sex": sex,
        "ChestPainType": cp,
        "RestingECG": restecg,
        "ExerciseAngina": angina,
        "ST_Slope": slope
    }

    df = pd.DataFrame([input_dict])

    # Apply same encoding
    df = pd.get_dummies(df)

    # Match training columns
    df = df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(df)

    if prediction[0] == 1:
        st.error(" High Risk")
    else:
        st.success("Low Risk")