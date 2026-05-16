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

/* ===== BACKGROUND ===== */
.stApp {
    background-color: #f4f7fb;
}

/* ===== HEADER ===== */
.header-box {
    background: linear-gradient(135deg, #0f172a, #2563eb);
    padding: 35px;
    border-radius: 25px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
}

.header-title {
    font-size: 40px;
    font-weight: bold;
}

.header-subtitle {
    font-size: 16px;
    color: #dbeafe;
    margin-top: 10px;
}

/* ===== CARD ===== */
.card {
    background: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* ===== SECTION TITLE ===== */
.section-title {
    font-size: 30px;
    font-weight: bold;
    color: #111827;
    margin-bottom: 20px;
}

/* ===== LABELS ===== */
label {
    color: #111827 !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}

/* ===== INPUT BOX ===== */
.stNumberInput input {
    background-color: #f9fafb !important;
    border: 1px solid #d1d5db !important;
    border-radius: 10px !important;
    height: 40px !important;
    font-size: 14px !important;
}

.stSelectbox div[data-baseweb="select"] {
    background-color: #f9fafb !important;
    border: 1px solid #d1d5db !important;
    border-radius: 10px !important;
    min-height: 40px !important;
}

/* ===== BUTTON ===== */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border: none;
    border-radius: 12px;
    height: 45px;
    width: 220px;
    font-size: 16px;
    font-weight: bold;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af);
    color: white;
}

/* ===== RESULT BOX ===== */
.result-box {
    background: #ecfdf5;
    border: 1px solid #bbf7d0;
    padding: 12px;
    border-radius: 10px;
    font-size: 14px;
    font-weight: 600;
    color: #166534;
    margin-top: 15px;
}

.result-box-danger {
    background: #fef2f2;
    border: 1px solid #fecaca;
    padding: 12px;
    border-radius: 10px;
    font-size: 14px;
    font-weight: 600;
    color: #991b1b;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="header-box">
    <div class="header-title">
        ❤️ Heart Disease Prediction
    </div>

    <div class="header-subtitle">
        Predict whether a patient is at risk of heart disease
        using machine learning and medical attributes.
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("gb_model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# =========================
# TITLE
# =========================
st.markdown("""
<div class="section-title">
📋 Patient Information
</div>
""", unsafe_allow_html=True)

# =========================
# THREE CARDS
# =========================
col1, col2, col3 = st.columns(3)

# ===== CARD 1 =====
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🧑 Personal Info")

    age = st.number_input("Age", 1, 120, 30)

    sex = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    cp = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "ASY", "TA"]
    )

    st.markdown('</div>', unsafe_allow_html=True)

# ===== CARD 2 =====
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("💼 Medical Details")

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

    st.markdown('</div>', unsafe_allow_html=True)

# ===== CARD 3 =====
with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📊 Health Analysis")

    oldpeak = st.number_input(
        "Oldpeak",
        0.0, 10.0, 1.0
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

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# BUTTON
# =========================
if st.button("🔍 Generate Prediction"):

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

    # Match columns
    df = df.reindex(columns=columns, fill_value=0)

    # Prediction
    prediction = model.predict(df)

    # =========================
    # RESULT SECTION
    # =========================
    st.markdown("""
    <br>
    <div class="section-title">
    📌 Prediction Result
    </div>
    """, unsafe_allow_html=True)

    if prediction[0] == 1:
        st.markdown("""
        <div class="result-box-danger">
        ⚠️ Patient has High Risk of Heart Disease
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="result-box">
        ✅ Patient has Low Risk of Heart Disease
        </div>
        """, unsafe_allow_html=True)
