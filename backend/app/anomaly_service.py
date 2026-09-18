"""
FleetSense - Live Anomaly Scoring Service
==============================================
Wraps the trained autoencoder to score a specific test engine's most
recent sensor reading against the healthy-data reconstruction threshold.
This is intentionally INDEPENDENT of the LSTM's RUL prediction - two
different models, two different paradigms, agreeing or disagreeing on
the same engine is itself meaningful signal.
"""
import pandas as pd
import numpy as np
import torch
import joblib

from train_lstm import FEATURE_COLS, DATA_DIR
from train_autoencoder import Autoencoder


def load_autoencoder():
    model = Autoencoder(n_features=len(FEATURE_COLS))
    model.load_state_dict(torch.load(f"{DATA_DIR}/autoencoder.pt", weights_only=True))
    model.eval()
    threshold = np.load(f"{DATA_DIR}/anomaly_threshold.npy")
    scaler = joblib.load(f"{DATA_DIR}/scaler.pkl")
    return model, threshold, scaler


def get_anomaly_status(engine_id: int) -> dict:
    test = pd.read_csv(f"{DATA_DIR}/test_FD001.txt", sep=r'\s+', header=None,
                        names=['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']
                        + [f's_{i}' for i in range(1, 22)])

    engine_data = test[test['unit_nr'] == engine_id].sort_values('time_cycles')
    if len(engine_data) == 0:
        return {"error": f"Engine {engine_id} not found in test data"}

    model, threshold, scaler = load_autoencoder()
    latest_row = engine_data.iloc[[-1]]
    X_scaled = scaler.transform(latest_row[FEATURE_COLS])
    x_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        recon = model(x_tensor)
        error = float(torch.mean((recon - x_tensor) ** 2).item())

    return {
        "engine_id": engine_id,
        "reconstruction_error": round(error, 5),
        "anomaly_threshold": round(float(threshold), 5),
        "is_anomalous": bool(error > threshold),
        "severity_ratio": round(error / float(threshold), 2),
    }