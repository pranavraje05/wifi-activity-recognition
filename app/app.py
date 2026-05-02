import streamlit as st

st.title("WiFi-Based Human Activity Recognition")

st.write("Upload CSI data to predict human activity")

file = st.file_uploader("Upload CSI file")

if file:
    st.success("File uploaded successfully!")
    st.write("Predicted Activity: Walking (Demo)")
