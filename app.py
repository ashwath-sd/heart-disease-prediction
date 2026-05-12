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

/* ===== FULL PAGE BACKGROUND ===== */
.stApp {
    background: linear-gradient(135deg, #dbeafe, #bfdbfe, #93c5fd);
    background-attachment: fixed;
}

/* ===== HEADER SECTION ===== */
.header-container {
    background: linear-gradient(135deg, #0f172a, #2563eb);
    padding: 45px;
    border-radius: 25px;
    text-align: center;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.25);
}

.header-container h1 {
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 10px;
    color: white;
}

.header-container p {
    font-size: 18px;
    color: #dbeafe;
}

/* ===== MAIN CARD ===== */
.main-box {
    background-color: rgba(255,255,255,0.95);
    padding: 35px;
    border-radius: 22px;
    box-shadow: 0px 0px 20px rgba(0,0,0,0.15);
}

/* ===== LABELS ===== */
label {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #1e293b !important;
}

/* ===== NUMBER INPUT FIELD ===== */
.stNumberInput input {
    background-color: #e0f2fe !important;
    color: #111827 !important;
    border-radius: 12px !important;
    border: 2px solid #7dd3fc !important;
    padding: 12px !important;
    font-size: 16px !important;
}

/* ===== SELECT BOX ===== */
.stSelectbox div[data-baseweb="select"] {
    background-color: #dbeafe !important;
    border-radius: 12px !important;
    border: 2px solid #60a5fa !important;
}

/* ===== SELECTBOX TEXT ===== */
.stSelectbox div {
    color: #111827 !important;
    font-size: 16px !important;
}

/* ===== INPUT HOVER EFFECT ===== */
.stNumberInput input:hover,
.stSelectbox div[data-baseweb="select"]:hover {
    border: 2px solid #2563eb !important;
    box-shadow: 0px 0px 10px rgba(37,99,235,0.35);
}

/* ===== BUTTON ===== */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border-radius: 14px;
    height: 55px;
    font-size: 18px;
    font-weight: bold;
    border: none;
    margin-top: 20px;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af);
    color: white;
}

/* ===== HIGH RISK RESULT ===== */
.result-high {
    background-color: #fee2e2;
    padding: 18px;
    border-radius: 14px;
    color: #991b1b;
    font-size: 24px;
    text-align: center;
    font-weight: bold;
    margin-top: 25px;
}

/* ===== LOW RISK RESULT ===== */
.result-low {
    background-color: #dcfce7;
    padding: 18px;
    border-radius: 14px;
    color: #166534;
    font-size: 24px;
    text-align: center;
    font-weight: bold;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)

# ======================
# HEADER SECTION
# ======================
st.markdown("""
<div class="header-container">
    <h1>❤️ Heart Disease Prediction</h1>
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
            '<div class="result-high">⚠️ High Risk of Heart Disease</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-low">✅ Low Risk of Heart Disease</div>',
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)
