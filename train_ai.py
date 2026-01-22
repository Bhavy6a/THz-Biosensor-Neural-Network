# Filename: train_ai.py
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import time
import os

def lorentzian(x, x0, amp, gamma):
    return amp * (gamma**2 / ((x - x0)**2 + gamma**2))

def generate_and_train():
    print("--- STEP 1: GENERATING SYNTHETIC DATA ---")
    n_samples = 150
    freq_res = 400
    dataset = []
    
    np.random.seed(42)
    glucose_levels = np.linspace(0, 100, n_samples)
    print(f"Simulating {n_samples} glucose concentrations...")

    for glucose in glucose_levels:
        freq_range = np.linspace(2, 5, freq_res)
        
        # Physics: Red Shift & Amplitude Decay
        f_shift = glucose * 0.003
        amp_decay = glucose * 0.02

        # S11 (Reflection)
        s11 = np.zeros_like(freq_range)
        s11 += lorentzian(freq_range, 4.21 - f_shift, -5.5 + amp_decay, 0.015)
        s11 += lorentzian(freq_range, 4.55 - f_shift, -25.0 + (amp_decay*2), 0.01)
        s11 += np.random.normal(0, 0.015, size=len(freq_range))

        # S21 (Transmission)
        s21 = np.full_like(freq_range, -1300.0)
        notches = [(2.70, -180, 0.005), (4.10, -180, 0.005), (4.25, -180, 0.005), (4.55, -180, 0.008), (4.75, -180, 0.005)]
        for f0, amp, gam in notches:
            s21 += lorentzian(freq_range, f0 - f_shift, amp + (amp_decay*5), gam)
        s21 += np.random.normal(0, 0.5, size=len(freq_range))

        # Pack
        temp_df = pd.DataFrame({
            'glucose_concentration': glucose,
            'frequency_thz': freq_range,
            's11_db': s11,
            's21_db': s21
        })
        dataset.append(temp_df)

    df = pd.concat(dataset, ignore_index=True)
    df.to_csv('training_data.csv', index=False)
    print("Data generation complete. File saved.")

    print("\n--- STEP 2: TRAINING NEURAL NETWORK ---")
    X = df[['glucose_concentration', 'frequency_thz']]
    y = df[['s11_db', 's21_db']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train
    model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
    start_time = time.time()
    model.fit(X_scaled, y)
    print(f"Training finished in {time.time() - start_time:.2f} seconds.")
    print(f"Model Accuracy (R2): {model.score(X_scaled, y):.5f}")

    # Save
    joblib.dump(model, 'biosensor_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("SUCCESS: Model saved as 'biosensor_model.pkl'")

if __name__ == "__main__":
    generate_and_train()
