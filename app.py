import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Iris Flower Classification")
st.write("Predict the Iris flower species using Machine Learning.")

model_path = os.path.join("models", "iris_model.pkl")

if not os.path.exists(model_path):
    st.error("Model not found. Please train the model first.")
    st.stop()

model = joblib.load(model_path)

st.subheader("Enter Flower Measurements")

sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=3.5
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=1.4
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=0.2
)

if st.button("Predict"):

    sample = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]],
        columns=[
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm"
        ]
    )

    prediction = model.predict(sample)[0]

    st.success(f"Predicted Species: {prediction}")

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(sample).max() * 100
        st.info(f"Confidence: {probability:.2f}%")