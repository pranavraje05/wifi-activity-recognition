import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.model import train_and_evaluate

st.title("WiFi-Based Human Activity Recognition")

st.write("Train model and view results")

model, acc, cm, preds = train_and_evaluate()

st.subheader("Model Performance")
st.write(f"Accuracy: {acc:.2f}")

st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
ax.imshow(cm)
ax.set_title("Confusion Matrix")
st.pyplot(fig)

st.subheader("Upload CSV for Prediction")
file = st.file_uploader("Upload CSV file")

if file:
    data = pd.read_csv(file)
    prediction = model.predict(data)
    st.success("Prediction complete!")
    st.write(prediction)
