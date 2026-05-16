import streamlit as st
import pickle
import pandas as pd

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

/* ===== APP BACKGROUND ===== */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e3a8a, #2563eb);
    background-attachment: fixed;
}

/* ===== MAIN CONTAINER ===== */
.main-container {
    background-color: rgba(255,255,255,0.97);
    padding: 40px;
    border-radius: 25px;
    margin-top: 20px;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.25);
}

/* ===== HEADER ===== */
.header-box {
    background: linear-gradient(135deg, #1e293b, #2563eb);
    padding: 40px;
    border-radius: 25px;
    text-align: center;
    margin-bottom: 35px;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.3);
}

.header-title {
    color: white;
    font-size: 48px;
    font-weight: bold;
    margin-bottom: 10px;
}

.header-subtitle {
    color: #dbeafe;
    font-size: 20px;
}

/* ===== SECTION TITLE ===== */
.section-title {
    font-size: 24px;
    font-weight: bold;
    color: #1e3a8a;
    margin-top: 15px;
    margin-bottom: 15px;
}

/* ===== INPUT BOX ===== */
.stNumberInput input {
    background-color: #eff6ff !important;
    border: 2px solid #93c5fd !important;
    border-radius: 14px !important;
    padding: 12px !important;
    color: #111827 !important;
    font-size: 16px !important;
}

/* ===== SELECT BOX ===== */
.stSelectbox div[data-baseweb="select"] {
    background-color: #eff6ff !important;
    border: 2px solid #93c5fd !important;
    border-radius: 14px !important;
}

/* ===== LABELS ===== */
label {
    color: #0f172a !important;
    font-weight: 600 !important;
    font-size: 16px !important;
}

/* ===== BUTTON ===== */
.stButton > button {
    width: 100%;
    height: 60px;
    border: none;
    border-radius: 16px;
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    font-size: 22px;
    font-weight: bold;
    margin-top: 25px;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.02);
    background: linear-gradient(135deg, #1d4ed8, #1e40af);
    color: white;
}

/* ===== RESULT SUCCESS ===== */
.result-low {
    background: linear-gradient(135deg, #dcfce7, #bbf7d0);
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: #166534;
    margin-top: 30px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
}

/* ===== RESULT DANGER ===== */
.result-high {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: #991b1b;
    margin-top: 30px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="header-box">
    <div class="header-title">❤️ Heart Disease Prediction</div>
    <div class="header-subtitle">
        AI-Powered Healthcare Risk Prediction System
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("gb_model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# =========================
# MAIN UI
# =========================
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown('<div class="section-title">📋 Patient Medical Details</div>', unsafe_allow_html=True)

# ===== COLUMNS =====
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 30)
    bp = st.number_input("Resting Blood Pressure", 80, 250, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    hr = st.number_input("Maximum Heart Rate", 60, 250, 150)
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)

with col2:
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

# =========================
# PREDICTION
# =========================
if st.button("🔍 Predict Heart Disease Risk"):

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

    # DataFrame
    df = pd.DataFrame([input_dict])

    # Encoding
    df = pd.get_dummies(df)

    # Match Columns
    df = df.reindex(columns=columns, fill_value=0)

    # Prediction
    prediction = model.predict(df)

    # Output
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
