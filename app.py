import streamlit as st, pandas as pd, joblib, json
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler (2).pkl")
schema = json.load(open("schema.json"))["feature_order"]

st.title("🩺 Diabetes Risk Predictor")
vals = {}
for f in schema:
    vals[f] = st.number_input(f, value=0.0, step=0.1)
if st.button("Predict"):
    X = pd.DataFrame([vals])[schema]
    Xs = scaler.transform(X)
    proba = model.predict_proba(Xs)[0,1]
    st.metric("Risk Probability", f"{proba:.2%}")
    st.caption("⚠️ Educational demo only; not medical advice.")
