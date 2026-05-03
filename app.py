import streamlit as st
import pickle
import os

st.title("Heart Disease Prediction")

st.write("App is running...")

try:
    model_path = os.path.join(os.getcwd(), "gboost_model.pkl")
    model = pickle.load(open(model_path, "rb"))

    st.success("Model loaded successfully")

except Exception as e:
    st.error(f"Error: {e}")
