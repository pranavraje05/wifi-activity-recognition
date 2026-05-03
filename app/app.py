import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from src.model import train_and_evaluate


# Title
st.title("WiFi-Based Human Activity Recognition")

# Run model
model, acc, y_test, preds = train_and_evaluate()

# Create confusion matrix
cm = confusion_matrix(y_test, preds)

# Show accuracy
st.subheader("📊 Model Performance")
st.write(f"Accuracy: {acc:.2f}")

# Show confusion matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
ax.imshow(cm, cmap='viridis')
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# Upload CSV for prediction
st.subheader("📂 Upload CSV for Prediction")
file = st.file_uploader("Upload CSV file")

if file:
    data = pd.read_csv(file)
    prediction = model.predict(data)

    st.write("Predictions:")
    st.write(prediction)
