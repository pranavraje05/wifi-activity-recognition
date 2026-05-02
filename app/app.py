import streamlit as st
import pandas as pd
from src.model import train_model

st.title("WiFi-Based Human Activity Recognition")

st.write("Upload data to predict activity")

model = train_model()

file = st.file_uploader("Upload CSV file")

if file:
    data = pd.read_csv(file)

    prediction = model.predict(data)

    st.success("Prediction complete!")
    st.write(prediction)
