"""
FleetSense - Autoencoder Anomaly Detection Validation
=========================================================
Tests whether the autoencoder (trained ONLY on healthy/early-life data)
correctly flags degraded/late-life engine data as anomalous - this is
the real proof the unsupervised approach works as an independent
early-warning signal.
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


def compute_reconstruction_error(model, x_scaled):
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    with torch.no_grad():
        recon = model(x_tensor)
        errors = torch.mean((recon - x_tensor) ** 2, dim=1).numpy()
    return errors


def validate_on_full_lifecycle(engine_id=1):
    """
    For one engine's full life, compute reconstruction error at every cycle.
    We EXPECT error to be low early on (healthy) and rise as the engine
    degrades (late cycles) - this is the actual behavior we're validating.
    """
    df = pd.read_csv(f"{DATA_DIR}/train_FD001_with_RUL.csv")
    engine_df = df[df['unit_nr'] == engine_id].sort_values('time_cycles')

    model, threshold, scaler = load_autoencoder()
    X_scaled = scaler.transform(engine_df[FEATURE_COLS])
    errors = compute_reconstruction_error(model, X_scaled)

    engine_df = engine_df.copy()
    engine_df['reconstruction_error'] = errors
    engine_df['flagged_anomaly'] = errors > threshold

    n_cycles = len(engine_df)
    first_third = engine_df.iloc[:n_cycles // 3]
    last_third = engine_df.iloc[-n_cycles // 3:]

    print(f"\n{'='*60}")
    print(f"ENGINE {engine_id} - FULL LIFECYCLE VALIDATION ({n_cycles} cycles)")
    print(f"{'='*60}")
    print(f"First third (early/healthy) - mean error: {first_third['reconstruction_error'].mean():.5f}, "
          f"anomalies flagged: {first_third['flagged_anomaly'].sum()}/{len(first_third)}")
    print(f"Last third (late/degraded) - mean error: {last_third['reconstruction_error'].mean():.5f}, "
          f"anomalies flagged: {last_third['flagged_anomaly'].sum()}/{len(last_third)}")

    first_anomaly_idx = engine_df[engine_df['flagged_anomaly']].index
    if len(first_anomaly_idx) > 0:
        first_flag_cycle = engine_df.loc[first_anomaly_idx[0], 'time_cycles']
        first_flag_rul = engine_df.loc[first_anomaly_idx[0], 'RUL']
        print(f"First anomaly flagged at cycle {first_flag_cycle} (RUL={first_flag_rul} remaining at that point)")
    else:
        print("No anomalies flagged in this engine's lifecycle")

    return engine_df


def validate_across_all_engines():
    df = pd.read_csv(f"{DATA_DIR}/train_FD001_with_RUL.csv")
    model, threshold, scaler = load_autoencoder()

    early_flag_rates, late_flag_rates = [], []
    for unit in df['unit_nr'].unique():
        engine_df = df[df['unit_nr'] == unit].sort_values('time_cycles')
        X_scaled = scaler.transform(engine_df[FEATURE_COLS])
        errors = compute_reconstruction_error(model, X_scaled)
        flagged = errors > threshold

        n = len(engine_df)
        early_flag_rates.append(flagged[:n // 3].mean())
        late_flag_rates.append(flagged[-n // 3:].mean())

    print(f"\n{'='*60}")
    print(f"AGGREGATE VALIDATION - ALL 100 ENGINES")
    print(f"{'='*60}")
    print(f"Avg anomaly flag rate, EARLY life (healthy): {np.mean(early_flag_rates)*100:.1f}%")
    print(f"Avg anomaly flag rate, LATE life (degraded): {np.mean(late_flag_rates)*100:.1f}%")
    print(f"(A working detector should show LATE >> EARLY)")


if __name__ == "__main__":
    validate_on_full_lifecycle(engine_id=1)
    validate_on_full_lifecycle(engine_id=25)
    validate_across_all_engines()