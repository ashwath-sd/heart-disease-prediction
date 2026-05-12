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

.stApp {
    background: linear-gradient(to right, #dbeafe, #f0f9ff);
    background-attachment: fixed;
}

.main-box {
    background-color: rgba(255,255,255,0.9);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.2);
}

h1 {
    text-align: center;
    color: #b91c1c;
}

.stButton>button {
    width: 100%;
    background-color: #dc2626;
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    background-color: #991b1b;
    color: white;
}

.result-high {
    background-color: #fee2e2;
    padding: 15px;
    border-radius: 10px;
    color: #991b1b;
    font-size: 22px;
    text-align: center;
    font-weight: bold;
}

.result-low {
    background-color: #dcfce7;
    padding: 15px;
    border-radius: 10px;
    color: #166534;
    font-size: 22px;
    text-align: center;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ======================
# TITLE
# ======================
st.markdown("<h1>❤️ Heart Disease Prediction System</h1>", unsafe_allow_html=True)

# ======================
# LOAD MODEL
# ======================
model = pickle.load(open("gb_model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# ======================
# MAIN CONTAINER
# ======================
st.markdown('<div class="main-box">', unsafe_allow_html=True)

# Inputs
age = st.number_input("Age", 1, 120, 30)
bp = st.number_input("Resting BP", 120)
chol = st.number_input("Cholesterol", 200)
hr = st.number_input("Max HR", 150)
oldpeak = st.number_input("Oldpeak", 1.0)

sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
restecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
angina = st.selectbox("Exercise Angina", ["Yes", "No"])
slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ======================
# PREDICTION
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

    df = pd.DataFrame([input_dict])

    # Encoding
    df = pd.get_dummies(df)

    # Match columns
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
