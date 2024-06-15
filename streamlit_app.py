import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load(r"C:\Users\User\decision_tree_model.pkl")

# Define the app
st.title("Water Potability Prediction")

# Define input fields
hardness = st.number_input("Hardness", min_value=0.0, max_value=400.0, value=100.0)
solids = st.number_input("Solids", min_value=0.0, max_value=60000.0, value=15000.0)
chloramines = st.number_input("Chloramines", min_value=0.0, max_value=15.0, value=5.0)
sulfate = st.number_input("Sulfate", min_value=0.0, max_value=500.0, value=150.0)
organic_carbon = st.number_input("Organic Carbon", min_value=0.0, max_value=30.0, value=10.0)

# Predict button
if st.button("Predict"):
    features = np.array([[hardness, solids, chloramines, sulfate, organic_carbon]])
    prediction = model.predict(features)
    result = "Potable" if prediction[0] == 1 else "Not Potable"
    st.write(f"The water is: {result}")
