import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import time

st.title("🛠️ Diagnostics Mode")

# --- CHECKPOINT 1: LOCATE FILES ---
st.header("1. File System Check")
script_dir = os.path.dirname(os.path.abspath(__file__))
st.write(f"📂 Script Location: `{script_dir}`")

model_path = os.path.join(script_dir, 'biosensor_model.pkl')
scaler_path = os.path.join(script_dir, 'scaler.pkl')

if os.path.exists(model_path):
    st.success(f"✅ Found model file ({os.path.getsize(model_path)} bytes)")
else:
    st.error("❌ MODEL FILE MISSING. Please run train_ai.py again.")
    st.stop()

# --- CHECKPOINT 2: LOAD MODEL ---
st.header("2. Model Loading")
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    st.success("✅ Model & Scaler loaded into memory.")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# --- CHECKPOINT 3: GENERATE DATA ---
st.header("3. Data Generation")
glucose_val = st.slider("Glucose Level", 0.0, 100.0, 50.0)
freq_range = np.linspace(2, 5, 20) # Low res for debugging

input_data = pd.DataFrame({
    'glucose_concentration': [glucose_val] * 20,
    'frequency_thz': freq_range
})
st.write("Input Data Preview (First 3 rows):")
st.dataframe(input_data.head(3))

# --- CHECKPOINT 4: PREDICTION ---
st.header("4. AI Prediction")
try:
    input_scaled = scaler.transform(input_data)
    preds = model.predict(input_scaled)
    st.success("✅ Prediction successful.")
except Exception as e:
    st.error(f"❌ Prediction crashed: {e}")
    st.stop()

# Check for NaNs (Broken Math)
if np.isnan(preds).any():
    st.error("❌ OUTPUT CONTAINS NaNs (Not a Number)! The model is broken.")
    st.stop()
else:
    st.write("Output Data Preview (First 3 rows):")
    results = pd.DataFrame({'Freq': freq_range, 'S11': preds[:, 0], 'S21': preds[:, 1]})
    st.dataframe(results.head(3))

# --- CHECKPOINT 5: STATIC PLOT ---
st.header("5. Visual Check (Matplotlib)")
st.write("Attempting to draw static image...")

try:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freq_range, preds[:, 0], 'r-', label='S11 (Reflection)', linewidth=2)
    ax.plot(freq_range, preds[:, 1], 'b--', label='S21 (Transmission)')
    ax.set_title(f"Sensor Response at {glucose_val} mg/dL")
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig) # <--- This sends a static IMAGE, not code.
    st.success("✅ Graph sent to browser.")
except Exception as e:
    st.error(f"❌ Plotting failed: {e}")
