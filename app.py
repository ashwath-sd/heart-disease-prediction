import streamlit as st
import pickle
import pandas as pd

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>

/* ===== BACKGROUND ===== */
.stApp {
    background: linear-gradient(135deg, #f8fafc, #e0f2fe);
    font-family: 'Segoe UI', sans-serif;
}

/* ===== REMOVE STREAMLIT DEFAULT SPACE ===== */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ===== HEADER ===== */
.header-container {
    background: linear-gradient(135deg, #0f172a, #2563eb);
    padding: 40px;
    border-radius: 24px;
    text-align: center;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.15);
}

.header-title {
    font-size: 44px;
    font-weight: 700;
    margin-bottom: 10px;
}

.header-subtitle {
    font-size: 17px;
    color: #dbeafe;
}

/* ===== MAIN CARD ===== */
.main-card {
    background: rgba(255,255,255,0.95);
    padding: 35px;
    border-radius: 24px;
    box-shadow: 0px 5px 18px rgba(0,0,0,0.08);
}

/* ===== SECTION TITLE ===== */
.section-title {
    font-size: 28px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 25px;
}

/* ===== LABELS ===== */
label {
    color: #1e293b !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}

/* ===== NUMBER INPUT ===== */
.stNumberInput input {
    background-color: #f8fafc !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 12px !important;
    color: #111827 !important;
    font-size: 15px !important;
    padding: 10px !important;
    height: 42px !important;
}

/* ===== SELECT BOX ===== */
.stSelectbox div[data-baseweb="select"] {
    background-color: #f8fafc !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 12px !important;
    min-height: 42px !important;
}

/* ===== SELECT TEXT ===== */
.stSelectbox div {
    color: #111827 !important;
    font-size: 15px !important;
}

/* ===== INPUT HOVER ===== */
.stNumberInput input:hover,
.stSelectbox div[data-baseweb="select"]:hover {
    border: 1.5px solid #2563eb !important;
}

/* ===== BUTTON ===== */
.stButton > button {
    width: 100%;
    height: 52px;
    border: none;
    border-radius: 14px;
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    font-size: 18px;
    font-weight: 600;
    transition: 0.3s ease;
    margin-top: 15px;
}

.stButton > button:hover {
    transform: translateY(-2px);
    background: linear-gradient(135deg, #1d4ed8, #1e40af);
    box-shadow: 0px 5px 15px rgba(37,99,235,0.25);
    color: white;
}

/* ===== RESULT BOX ===== */
.result-success {
    background: #ecfdf5;
    border-left: 6px solid #22c55e;
    padding: 14px;
    border-radius: 12px;
    font-size: 15px;
    font-weight: 600;
    color: #166534;
    margin-top: 20px;
}

.result-danger {
    background: #fef2f2;
    border-left: 6px solid #ef4444;
    padding: 14px;
    border-radius: 12px;
    font-size: 15px;
    font-weight: 600;
    color: #991b1b;
    margin-top: 20px;
}

/* ===== FOOTER ===== */
.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================
st.markdown("""
<div class="header-container">
    <div class="header-title">
        ❤️ Heart Disease Prediction
    </div>

    <div class="header-subtitle">
        Predict heart disease risk using AI-powered healthcare analytics
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================
model = pickle.load(open("gb_model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# ==========================================
# MAIN CARD
# ==========================================
st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.markdown("""
<div class="section-title">
📋 Patient Medical Details
</div>
""", unsafe_allow_html=True)

# ==========================================
# INPUT FIELDS
# ==========================================
col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        1, 120, 30
    )

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

# ==========================================
# PREDICTION
# ==========================================
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

    # Match Columns
    df = df.reindex(columns=columns, fill_value=0)

    # Prediction
    prediction = model.predict(df)

    # RESULT
    if prediction[0] == 1:

        st.markdown("""
        <div class="result-danger">
            ⚠️ Prediction Result: High Risk of Heart Disease
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="result-success">
            ✅ Prediction Result: Low Risk of Heart Disease
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    AI Powered Healthcare Prediction System
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
