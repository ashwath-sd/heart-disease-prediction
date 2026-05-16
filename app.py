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

/* ===== FULL APP BACKGROUND ===== */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e3a8a, #2563eb);
    background-attachment: fixed;
}

/* ===== HEADER ===== */
.header-box {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    padding: 40px;
    border-radius: 28px;
    text-align: center;
    margin-bottom: 35px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0px 8px 25px rgba(0,0,0,0.25);
}

.header-title {
    color: white;
    font-size: 46px;
    font-weight: bold;
    margin-bottom: 12px;
}

.header-subtitle {
    color: #dbeafe;
    font-size: 18px;
}

/* ===== MAIN CARD ===== */
.main-container {
    background: rgba(255,255,255,0.96);
    padding: 40px;
    border-radius: 28px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.18);
}

/* ===== SECTION TITLE ===== */
.section-title {
    font-size: 26px;
    font-weight: bold;
    color: #1e3a8a;
    margin-bottom: 20px;
}

/* ===== LABELS ===== */
label {
    color: #0f172a !important;
    font-size: 17px !important;
    font-weight: 700 !important;
}

/* ===== NUMBER INPUT ===== */
.stNumberInput input {
    background: #eff6ff !important;
    border: 2px solid #3b82f6 !important;
    border-radius: 14px !important;
    color: #111827 !important;
    padding: 12px !important;
    font-size: 17px !important;
    font-weight: 600 !important;
}

/* ===== SELECT BOX ===== */
.stSelectbox div[data-baseweb="select"] {
    background: #eff6ff !important;
    border: 2px solid #3b82f6 !important;
    border-radius: 14px !important;
    min-height: 50px;
}

/* ===== SELECT TEXT ===== */
.stSelectbox div {
    color: #111827 !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}

/* ===== INPUT HOVER EFFECT ===== */
.stNumberInput input:hover,
.stSelectbox div[data-baseweb="select"]:hover {
    border: 2px solid #2563eb !important;
    box-shadow: 0px 0px 12px rgba(37,99,235,0.35);
}

/* ===== BUTTON ===== */
.stButton > button {
    width: 100%;
    height: 58px;
    border: none;
    border-radius: 16px;
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    font-size: 20px;
    font-weight: bold;
    margin-top: 25px;
    transition: 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    background: linear-gradient(135deg, #1d4ed8, #1e40af);
    box-shadow: 0px 6px 15px rgba(37,99,235,0.4);
    color: white;
}

/* ===== RESULT SUCCESS ===== */
.result-low {
    background: linear-gradient(135deg, #dcfce7, #bbf7d0);
    padding: 18px;
    border-radius: 18px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    color: #166534;
    margin-top: 28px;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.15);
}

/* ===== RESULT DANGER ===== */
.result-high {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    padding: 18px;
    border-radius: 18px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    color: #991b1b;
    margin-top: 28px;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.15);
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER SECTION
# =========================
st.markdown("""
<div class="header-box">
    <div class="header-title">❤️ Heart Disease Prediction</div>
    <div class="header-subtitle">
        AI Powered Heart Disease Risk Analysis System
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("gb_model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# =========================
# MAIN CONTAINER
# =========================
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">📋 Patient Medical Details</div>',
    unsafe_allow_html=True
)

# =========================
# INPUT FIELDS
# =========================
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 30)

    bp = st.number_input(
        "Resting Blood Pressure",
        80, 250, 120
    )

    chol = st.number_input(
        "Cholesterol",
        100, 600, 200
    )

    hr = st.number_input(
        "Maximum Heart Rate",
        60, 250, 150
    )

    oldpeak = st.number_input(
        "Oldpeak",
        0.0, 10.0, 1.0
    )

with col2:
    sex = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )

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

    # Create DataFrame
    df = pd.DataFrame([input_dict])

    # Encoding
    df = pd.get_dummies(df)

    # Match Model Columns
    df = df.reindex(columns=columns, fill_value=0)

    # Prediction
    prediction = model.predict(df)

    # Output Result
    if prediction[0] == 1:
        st.markdown(
            '''
            <div class="result-high">
                ⚠️ High Risk of Heart Disease
            </div>
            ''',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '''
            <div class="result-low">
                ✅ Low Risk of Heart Disease
            </div>
            ''',
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)
