import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load(open("diabetes_model.pkl", "rb"))
scaler = joblib.load(open("scaler (2).pkl", "rb"))

st.title("🩺 Diabetes Prediction App")

st.markdown("### Enter Patient Details:")

# Input sliders
pregnancies = st.slider("Number of Pregnancies", 0, 20, 1)
glucose = st.slider("Glucose Level", 0, 300, 120)
blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 200, 70)
skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)
insulin = st.slider("Insulin Level", 0, 900, 80)
bmi = st.slider("BMI", 0.0, 70.0, 25.0)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.slider("Age", 10, 100, 30)

# Collect features into numpy array
features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

# Apply scaling
features_scaled = scaler.transform(features)

# Predict button
if st.button("🔍 Predict"):
    prediction = model.predict(features_scaled)[0]

    if prediction == 1:
        st.error("⚠️ High risk of Diabetes!")
    else:
        st.success("✅ Low risk of Diabetes")
