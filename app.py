import streamlit as st
import pickle
import pandas as pd

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Heart Disease Prediction",
    
    layout="centered"
)

# =====================================
# CUSTOM CSS
# =====================================
st.markdown("""
<style>

/* ===== MAIN BACKGROUND ===== */
/* ===== MAIN BACKGROUND ===== */
/* ===== MAIN BACKGROUND ===== */
.stApp {
    background: linear-gradient(
        135deg,
        #f8fafc,
        #e0f2fe,
        #dbeafe,
        #bfdbfe
    );
    background-attachment: fixed;
}

/* ===== REMOVE EXTRA SPACE ===== */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ===== HEADER ===== */
.header-box {
    background: linear-gradient(135deg, #111827, #2563eb);
    padding: 32px;
    border-radius: 24px;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.25);
}

.header-title {
    color: white;
    font-size: 40px;
    font-weight: bold;
    margin-bottom: 8px;
}

.header-subtitle {
    color: #dbeafe;
    font-size: 15px;
}

/* ===== MAIN CARD ===== */
.main-container {
    background: rgba(255,255,255,0.96);
    padding: 30px;
    border-radius: 24px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.18);
    max-width: 850px;
    margin: auto;
}

/* ===== SECTION TITLE ===== */
.section-title {
    font-size: 24px;
    font-weight: bold;
    color: #1e3a8a;
    margin-bottom: 20px;
    text-align: center;
}

/* ===== LABELS ===== */
label {
    color: #0f172a !important;
    font-size: 14px !important;
    font-weight: 700 !important;
}

/* ===== INPUT FIELDS ===== */
.stNumberInput input {
    background: #f8fafc !important;
    border: 1.8px solid #60a5fa !important;
    border-radius: 12px !important;
    color: #111827 !important;
    font-size: 14px !important;
    height: 42px !important;
}

/* ===== SELECT BOX ===== */
.stSelectbox div[data-baseweb="select"] {
    background: #f8fafc !important;
    border: 1.8px solid #60a5fa !important;
    border-radius: 12px !important;
    min-height: 42px;
}

/* ===== SELECT TEXT ===== */
.stSelectbox div {
    color: #111827 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}

/* ===== HOVER EFFECT ===== */
.stNumberInput input:hover,
.stSelectbox div[data-baseweb="select"]:hover {
    border: 1.8px solid #2563eb !important;
    box-shadow: 0px 0px 8px rgba(37,99,235,0.25);
}

/* ===== BUTTON ===== */
.stButton > button {
    width: 100%;
    height: 48px;
    border: none;
    border-radius: 14px;
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    font-size: 16px;
    font-weight: bold;
    margin-top: 18px;
    transition: 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    background: linear-gradient(135deg, #1d4ed8, #1e40af);
    color: white;
}

/* ===== RESULT SUCCESS ===== */
.result-low {
    background: #dcfce7;
    border-left: 6px solid #16a34a;
    padding: 14px;
    border-radius: 12px;
    text-align: center;
    font-size: 15px;
    font-weight: bold;
    color: #166534;
    margin-top: 20px;
}

/* ===== RESULT DANGER ===== */
.result-high {
    background: #fee2e2;
    border-left: 6px solid #dc2626;
    padding: 14px;
    border-radius: 12px;
    text-align: center;
    font-size: 15px;
    font-weight: bold;
    color: #991b1b;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================
st.markdown("""
<div class="header-box">
    <div class="header-title">
         Heart Disease Prediction
    </div>

   

""", unsafe_allow_html=True)

# =====================================
# LOAD MODEL
# =====================================
model = pickle.load(open("gb_model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# =====================================
# MAIN CONTAINER
# =====================================
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown("""
<div class="section-title">
📋 Patient Medical Details
</div>
""", unsafe_allow_html=True)

# =====================================
# INPUT FIELDS
# =====================================
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

# =====================================
# PREDICTION
# =====================================
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
        <div class="result-high">
             High Risk of Heart Disease
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="result-low">
             Low Risk of Heart Disease
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
