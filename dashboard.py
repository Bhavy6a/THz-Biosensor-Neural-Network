import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# --- Page Config ---
st.set_page_config(page_title="THz Data Viewer", layout="wide")
st.header("📡 THz S-Parameters Viewer (From CSV)")

# --- 1. Load Data ---
# We check two locations: The hardcoded path AND the local folder (failsafe)
hardcoded_path = r"D:\User\Desktop\biosensor_AI\training_data.csv"
local_path = "training_data.csv"

file_to_load = hardcoded_path if os.path.exists(hardcoded_path) else local_path

try:
    df = pd.read_csv(file_to_load)
    
    # --- FIX: Normalize Column Names ---
    # This ensures "Concentration", "concentration", and " Glucose_Concentration" all work
    df.columns = df.columns.str.strip().str.lower()
    
    # Rename columns to standard keys for easy access
    # We look for keywords in the columns to identify them
    col_map = {}
    for col in df.columns:
        if "conc" in col: col_map[col] = "concentration"
        elif "freq" in col: col_map[col] = "frequency"
        elif "s11" in col: col_map[col] = "S11"
        elif "s21" in col: col_map[col] = "S21"
    
    df = df.rename(columns=col_map)
    
    # Verify we have the required data
    required_cols = ["concentration", "frequency", "S11", "S21"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV is missing columns! Found: {df.columns.tolist()}")
        st.stop()
        
    st.success(f"Loaded {len(df)} rows successfully from `{file_to_load}`")

except Exception as e:
    st.error(f"Could not load file. Checked path: {hardcoded_path}")
    st.error(f"Error details: {e}")
    st.stop()

# --- 2. User Selection ---
st.subheader("🔬 Select Glucose Concentration")

concentration_values = sorted(df["concentration"].unique())

selected_conc = st.selectbox(
    "Choose concentration (mg/dL):",
    concentration_values
)

# Filter data
filtered_df = df[df["concentration"] == selected_conc]

# --- 3. Professional Dual Plotting ---
st.write(f"### S-Parameters at {selected_conc} mg/dL")

# We use make_subplots to handle the massive scale difference between S11 and S21
fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.1,
    subplot_titles=("Reflection Coefficient (S11)", "Transmission Coefficient (S21)")
)

# Top Plot: S11
fig.add_trace(go.Scatter(
    x=filtered_df["frequency"], 
    y=filtered_df["S11"],
    mode='lines', 
    name='S11 (dB)',
    line=dict(color='#FF4B4B', width=3)
), row=1, col=1)

# Bottom Plot: S21
fig.add_trace(go.Scatter(
    x=filtered_df["frequency"], 
    y=filtered_df["S21"],
    mode='lines', 
    name='S21 (dB)',
    line=dict(color='#1F77B4', width=2)
), row=2, col=1)

# Layout Formatting
fig.update_layout(
    height=700,
    template="plotly_white",
    hovermode="x unified",
    showlegend=False
)

# Axis Labels
fig.update_yaxes(title_text="Magnitude (dB)", row=1, col=1)
fig.update_yaxes(title_text="Magnitude (dB)", row=2, col=1)
fig.update_xaxes(title_text="Frequency (THz)", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# Show raw data at the bottom
with st.expander("View Raw Data for this Concentration"):
    st.dataframe(filtered_df)
