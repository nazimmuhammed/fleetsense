"""
FleetSense - Live Fault Injection Simulator
================================================
Lets an operator artificially stress an engine's sensor readings and
watch the REAL trained LSTM react - not a fabricated number. We take the
engine's actual last window of sensor data and apply a controlled
synthetic drift (scaled by a severity slider), then run a genuine
forward pass through the model with MC-Dropout uncertainty.
"""
import numpy as np
import pandas as pd
import torch

from predict_with_uncertainty import load_model_and_scaler, predict_with_uncertainty
from train_lstm import FEATURE_COLS, WINDOW_SIZE, DATA_DIR


def simulate_degradation(engine_id: int, severity: float):
    """
    severity: 0.0 (no injected fault) to 1.0 (extreme synthetic stress).

    Extrapolates each sensor's OWN observed trend across the current
    window further forward - pushing the engine further along its real,
    already-occurring degradation trajectory, scaled by severity. This
    avoids random noise (which can move in the wrong direction relative
    to genuine degradation) in favor of amplifying a real pattern.
    """
    test = pd.read_csv(f"{DATA_DIR}/test_FD001.txt", sep=r'\s+', header=None,
                        names=['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']
                        + [f's_{i}' for i in range(1, 22)])

    model, scaler = load_model_and_scaler()
    engine_data = test[test['unit_nr'] == engine_id].sort_values('time_cycles')
    if len(engine_data) < WINDOW_SIZE:
        return {"error": f"Engine {engine_id} has insufficient cycle history"}

    window = engine_data[FEATURE_COLS].values[-WINDOW_SIZE:].copy()

    # Per-feature trend already present across this window (end - start).
    trend_per_cycle = (window[-1] - window[0]) / WINDOW_SIZE

    # Push forward by extra "simulated" cycles proportional to severity -
    # e.g. severity=1.0 simulates roughly one full extra window's worth
    # of continued decline along the engine's own real trajectory.
    extra_cycles = severity * WINDOW_SIZE
    window_stressed = window + trend_per_cycle * extra_cycles

    scaled = scaler.transform(window_stressed)
    x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
    result = predict_with_uncertainty(model, x)

    return {
        "engine_id": engine_id,
        "severity_applied": severity,
        "predicted_rul": round(result["mean_rul"], 1),
        "uncertainty_std": round(result["std_rul"], 1),
        "confidence_interval_95": [round(result["lower_95"], 1), round(result["upper_95"], 1)],
        "note": "Real forward pass through the trained LSTM, extrapolating this engine's own observed sensor trend further forward - not random noise.",
    }