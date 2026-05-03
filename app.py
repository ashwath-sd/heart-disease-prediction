import streamlit as st
import pickle
import os

st.title("Heart Disease Prediction")

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "gb_model.pkl")

    model = pickle.load(open(model_path, "rb"))
    st.success("Model loaded successfully")

except Exception as e:
    st.error(f"Error: {e}")
