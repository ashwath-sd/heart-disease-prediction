import streamlit as st
import pickle

st.title("Heart Disease Prediction")

try:
    model = pickle.load(open("gb_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))  # make sure this exists

    st.success("Model loaded successfully")

except Exception as e:
    st.error(f"Error loading model: {e}")
