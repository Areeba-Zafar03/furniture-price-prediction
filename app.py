import streamlit as st
import numpy as np
import pickle

# Load trained model
with open("linear_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Furniture Price Prediction")

# User Inputs
size = st.number_input("Enter furniture size (sq ft):", min_value=5.0, max_value=50.0, step=0.1)
material_quality = st.slider("Material Quality (1-5):", 1, 5, 3)
brand_reputation = st.slider("Brand Reputation (1-5):", 1, 5, 3)
age = st.number_input("Enter furniture age (years):", min_value=0.0, max_value=10.0, step=0.1)

# Predict button
if st.button("Predict Price"):
    prediction = model.predict(np.array([[size, material_quality, brand_reputation, age]]))
    st.success(f"Predicted Price: ${prediction[0]:,.2f}")
