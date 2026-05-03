import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from src.model import train_model

st.title("WiFi-Based Human Activity Recognition")

# Run training
model, acc, y_test, preds = train_model()

# Confusion matrix
cm = confusion_matrix(y_test, preds)

st.subheader("📊 Model Performance")
st.write(f"Accuracy: {acc:.2f}")

st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
ax.imshow(cm, cmap='viridis')
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# Prediction upload
st.subheader("📂 Upload CSV for Prediction")
file = st.file_uploader("Upload CSV file")

if file:
    data = pd.read_csv(file)
    prediction = model.predict(data)
    st.write("Predictions:")
    st.write(prediction)
