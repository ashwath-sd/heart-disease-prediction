import streamlit as st
import pickle
import pandas as pd

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# ======================
# CUSTOM CSS
# ======================
st.markdown("""
<style>

/* Full Page Background */
.stApp {
    background-color: #f3f4f6;
}

/* Header Banner */
.header-container {
    background: linear-gradient(135deg, #0f172a, #2563eb);
    padding: 45px;
    border-radius: 25px;
    text-align: center;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.25);
}

/* Header Title */
.header-container h1 {
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 10px;
    color: white;
}

/* Header Subtitle */
.header-container p {
    font-size: 18px;
    color: #dbeafe;
}

/* Main Input Card */
.main-box {
    background-color: white;
    padding: 35px;
    border-radius: 20px;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.12);
}

/* Input Labels */
label {
    font-weight: 600 !important;
    color: #111827 !important;
}

/* Button Styling */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border-radius: 12px;
    height: 52px;
    font-size: 18px;
    font-weight: bold;
    border: none;
    margin-top: 15px;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af);
    color: white;
}

/* High Risk Output */
.result-high {
    background-color: #fee2e2;
    padding: 18px;
    border-radius: 12px;
    color: #991b1b;
    font-size: 24px;
    text-align: center;
    font-weight: bold;
    margin-top: 20px;
}

/* Low Risk Output */
.result-low {
    background-color: #dcfce7;
    padding: 18px;
    border-radius: 12px;
    color: #166534;
    font-size: 24px;
    text-align: center;
    font-weight: bold;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ======================
# HEADER SECTION
# ======================
st.markdown("""
<div class="header-container">
    <h1> Heart Disease Prediction</h1>
    <p>
        Predict whether a patient is at risk of heart disease
        using machine learning and medical attributes.
    </p>
</div>
""", unsafe_allow_html=True)

# ======================
# LOAD MODEL
# ======================
model = pickle.load(open("gb_model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# ======================
# MAIN CONTAINER
# ======================
st.markdown('<div class="main-box">', unsafe_allow_html=True)

# ======================
# INPUT FIELDS
# ======================

age = st.number_input("Age", 1, 120, 30)

bp = st.number_input("Resting Blood Pressure", 80, 250, 120)

chol = st.number_input("Cholesterol", 100, 600, 200)

hr = st.number_input("Maximum Heart Rate", 60, 250, 150)

oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)

sex = st.selectbox("Sex", ["Male", "Female"])

cp = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "ASY", "TA"]
)

restecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

angina = st.selectbox(
    "Exercise Angina",
    ["Yes", "No"]
)

slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

# ======================
# PREDICTION BUTTON
# ======================
if st.button("Predict Heart Disease Risk"):

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

    # Create DataFrame
    df = pd.DataFrame([input_dict])

    # Encoding
    df = pd.get_dummies(df)

    # Match Training Columns
    df = df.reindex(columns=columns, fill_value=0)

    # Prediction
    prediction = model.predict(df)

    # Output Result
    if prediction[0] == 1:
        st.markdown(
            '<div class="result-high"> High Risk of Heart Disease</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-low">Low Risk of Heart Disease</div>',
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)
