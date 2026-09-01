"""
FleetSense - Prediction with Monte Carlo Dropout Uncertainty
================================================================
Loads the trained LSTM and generates RUL predictions WITH confidence
intervals using Monte Carlo Dropout (Gal & Ghahramani, 2016) - instead of
one number, we get a distribution, which is what a real maintenance
decision needs: "how confident are we in this prediction?"
"""

import pandas as pd
import numpy as np
import torch
import joblib

from train_lstm import RULLSTM, FEATURE_COLS, WINDOW_SIZE, RUL_CAP, DATA_DIR


def load_model_and_scaler():
    model = RULLSTM(n_features=len(FEATURE_COLS))
    model.load_state_dict(torch.load(f"{DATA_DIR}/rul_lstm_model.pt"))
    scaler = joblib.load(f"{DATA_DIR}/scaler.pkl")
    return model, scaler


def predict_with_uncertainty(model, x, n_samples=50):
    """
    Keep dropout ACTIVE (model.train()) during inference and run multiple
    forward passes. The spread of these predictions IS the model's
    uncertainty estimate - this is the actual Monte Carlo Dropout technique.
    """
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(x).item())
    preds = np.array(preds)
    return {
        "mean_rul": float(preds.mean()),
        "std_rul": float(preds.std()),
        "lower_95": float(np.percentile(preds, 2.5)),
        "upper_95": float(np.percentile(preds, 97.5)),
    }


def sanity_check_on_test_engine(engine_id=1):
    """
    Runs a real prediction on a specific test engine and compares against
    the ACTUAL RUL ground truth (from RUL_FD001.txt) - your real accuracy
    check on genuinely held-out data the model never trained on.
    """
    test = pd.read_csv(f"{DATA_DIR}/test_FD001.txt", sep=r'\s+', header=None,
                        names=['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']
                        + [f's_{i}' for i in range(1, 22)])
    rul_test = pd.read_csv(f"{DATA_DIR}/RUL_FD001.txt", sep=r'\s+', header=None, names=['RUL'])

    model, scaler = load_model_and_scaler()

    engine_data = test[test['unit_nr'] == engine_id].sort_values('time_cycles')
    if len(engine_data) < WINDOW_SIZE:
        print(f"Engine {engine_id} has fewer than {WINDOW_SIZE} cycles, skipping")
        return

    engine_data_scaled = engine_data.copy()
    engine_data_scaled[FEATURE_COLS] = scaler.transform(engine_data[FEATURE_COLS])
    last_window = engine_data_scaled[FEATURE_COLS].values[-WINDOW_SIZE:]
    x = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0)

    result = predict_with_uncertainty(model, x)
    actual_rul = rul_test.iloc[engine_id - 1]['RUL']

    print(f"\n{'='*60}")
    print(f"ENGINE {engine_id} - REAL TEST SET PREDICTION")
    print(f"{'='*60}")
    print(f"Predicted RUL: {result['mean_rul']:.1f} cycles")
    print(f"Uncertainty (std): +/- {result['std_rul']:.1f} cycles")
    print(f"95% confidence interval: [{result['lower_95']:.1f}, {result['upper_95']:.1f}]")
    print(f"ACTUAL RUL (ground truth): {actual_rul}")
    print(f"Absolute error: {abs(result['mean_rul'] - actual_rul):.1f} cycles")
    print(f"Ground truth within 95% CI? {'YES' if result['lower_95'] <= actual_rul <= result['upper_95'] else 'NO'}")


if __name__ == "__main__":
    for engine_id in [1, 5, 10, 25, 50]:
        sanity_check_on_test_engine(engine_id)