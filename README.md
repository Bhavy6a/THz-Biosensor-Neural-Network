# THz Biosensor for Glucose Detection: AI-Driven Analysis

This project focuses on the development and analysis of a Terahertz (THz) biosensor designed for glucose detection using a silicon ring resonator. By leveraging machine learning and interactive visualization, the system simulates and analyzes the resonance shifts in S-parameters (S11 and S21) that occur as glucose concentrations vary.

## 🚀 Project Overview

The project integrates electromagnetic simulation data (originally derived from CST Studio Suite) with a Python-based machine learning pipeline to predict biosensor responses. It includes a physics-informed synthetic data generator, a neural network training script, and a professional-grade dashboard for data exploration.

### Key Features

* **Physics-Informed Data Generation:** Simulates THz resonance notches using Lorentzian distributions, incorporating red shift and amplitude decay as a function of glucose levels.
* **Neural Network Integration:** Utilizes a Multi-layer Perceptron (MLP) regressor to learn the complex mapping between glucose concentration, frequency, and S-parameters.
* **Interactive Analytics Dashboard:** A Streamlit-based interface for visualizing reflection (S11) and transmission (S21) coefficients with interactive Plotly graphs.
* **System Diagnostics:** A dedicated diagnostic mode to verify model health, data integrity, and prediction consistency.

---

## 🏗️ Project Architecture

### 1. Data Generation & Training (`train_ai.py`)

This script serves as the backbone of the project. It generates a synthetic dataset of 150 samples with a frequency resolution of 400 points between 2 THz and 5 THz.

* **Mathematical Modeling:** Resonance notches are modeled using the Lorentzian function:

* **Physics Simulation:** Implements a red shift factor (0.003 per unit) and amplitude decay (0.02 per unit) to mimic real-world sensor behavior.
* **Model Training:** Trains an `MLPRegressor` and saves the resulting model and scaler as `.pkl` files for deployment.

### 2. Interactive Dashboard (`dashboard.py`)

A comprehensive visualization tool that allows users to:

* Load simulated S-parameter data from `training_data.csv`.
* Filter results by specific glucose concentrations (mg/dL) via a dropdown menu.
* View dual-plot visualizations of the Reflection Coefficient (S11) and Transmission Coefficient (S21) with shared x-axes for easy comparison.

### 3. Diagnostic Mode (`debug.py`)

A utility script to ensure the system is functioning correctly. It performs:

* **File System Checks:** Verifies the presence and size of `biosensor_model.pkl` and `scaler.pkl`.
* **Prediction Validation:** Tests the model against real-time inputs to ensure no `NaN` (Not a Number) values are produced.
* **Static Plotting:** Provides a quick Matplotlib visual check of the model's current predictions.

---

## 📊 Dataset Structure

The generated dataset (`training_data.csv`) contains the following features:

* `glucose_concentration`: The simulated glucose level.
* `frequency_thz`: The frequency range (2–5 THz).
* `s11_db`: The Reflection Coefficient in decibels.
* `s21_db`: The Transmission Coefficient in decibels.

---

## 🛠️ Installation & Usage

### Prerequisites

Ensure you have Python installed, then install the required dependencies:

```bash
pip install pandas numpy scikit-learn streamlit plotly matplotlib joblib

```

### Running the Project

1. **Generate Data and Train Model:**
```bash
python train_ai.py

```


This will produce `training_data.csv`, `biosensor_model.pkl`, and `scaler.pkl`.
2. **Launch the Dashboard:**
```bash
streamlit run dashboard.py

```


3. **Run Diagnostics (Optional):**
```bash
streamlit run debug.py

```



---

## 🔬 Scientific Context

This work leverages the high sensitivity of THz waves to the dielectric properties of biological molecules. Using a silicon ring resonator allows for enhanced light-matter interaction, making it possible to detect minute changes in glucose concentration through resonance frequency shifts and peak attenuation.
