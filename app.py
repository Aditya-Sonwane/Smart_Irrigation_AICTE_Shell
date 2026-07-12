import streamlit as st
import numpy as np
import pandas as pd
import joblib  

model = joblib.load("Farm_Irrigation_System.pkl")  
scaler = joblib.load("MinMaxScaler.pkl")

feature_names = [f"sensor_{i}" for i in range(20)]

st.set_page_config(page_title="Smart Irrigation Dashboard", layout="wide")

st.title("🌱 Smart Sprinkler Control System")
st.subheader("Real-Time Automated Field Irrigation Telemetry")
st.write("Adjust the environmental sensor metrics below to simulate active field conditions.")
st.markdown("---")

st.markdown("### Environmental Sensor Telemetry Inputs")
sensor_values = []

cols = st.columns(4) 
for i in range(20):
    with cols[i % 4]:
        val = st.slider(f"Sensor {i}", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
        sensor_values.append(val)

st.markdown("---")

if st.button("Analyze Telemetry & Predict Sprinklers", use_container_width=True):
    input_df = pd.DataFrame([sensor_values], columns=feature_names)
    
    scaled_input = scaler.transform(input_df)
    
    scaled_df = pd.DataFrame(scaled_input, columns=feature_names)
    
    prediction = model.predict(scaled_df)[0]

    st.markdown("### System Activation Status:")
    
    result_cols = st.columns(3)
    for i, status in enumerate(prediction):
        with result_cols[i]:
            if status == 1:
                st.success(f"🟢 Parcel {i}: **ACTIVE (ON)**")
            else:
                st.error(f"🔴 Parcel {i}: **IDLE (OFF)**")         