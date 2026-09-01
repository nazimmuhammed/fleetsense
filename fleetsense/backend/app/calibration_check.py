"""
FleetSense - Full Calibration Check
======================================
Runs uncertainty-aware predictions across ALL 100 test engines and checks:
if our 95% confidence intervals are well-calibrated, roughly 95% of them
should actually contain the true RUL value. This is a real, standard way
to validate that an uncertainty estimate isn't just a number, but a
statistically meaningful one.
"""

import pandas as pd
import numpy as np
import torch

from train_lstm import FEATURE_COLS, WINDOW_SIZE, DATA_DIR
from predict_with_uncertainty import load_model_and_scaler, predict_with_uncertainty


def run_full_calibration():
    test = pd.read_csv(f"{DATA_DIR}/test_FD001.txt", sep=r'\s+', header=None,
                        names=['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']
                        + [f's_{i}' for i in range(1, 22)])
    rul_test = pd.read_csv(f"{DATA_DIR}/RUL_FD001.txt", sep=r'\s+', header=None, names=['RUL'])

    model, scaler = load_model_and_scaler()

    results = []
    for engine_id in sorted(test['unit_nr'].unique()):
        engine_data = test[test['unit_nr'] == engine_id].sort_values('time_cycles')
        if len(engine_data) < WINDOW_SIZE:
            continue

        engine_data_scaled = engine_data.copy()
        engine_data_scaled[FEATURE_COLS] = scaler.transform(engine_data[FEATURE_COLS])
        last_window = engine_data_scaled[FEATURE_COLS].values[-WINDOW_SIZE:]
        x = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0)

        pred = predict_with_uncertainty(model, x)
        actual = rul_test.iloc[engine_id - 1]['RUL']

        within_ci = pred['lower_95'] <= actual <= pred['upper_95']
        results.append({
            "engine_id": engine_id,
            "predicted": pred['mean_rul'],
            "actual": actual,
            "abs_error": abs(pred['mean_rul'] - actual),
            "std": pred['std_rul'],
            "within_95_ci": within_ci,
        })

    df = pd.DataFrame(results)

    print("=" * 60)
    print("FULL CALIBRATION CHECK - ALL TEST ENGINES")
    print("=" * 60)
    print(f"Engines evaluated: {len(df)}")
    print(f"Overall RMSE: {np.sqrt((df['abs_error']**2).mean()):.2f} cycles")
    print(f"Overall MAE: {df['abs_error'].mean():.2f} cycles")
    print(f"\n% of true RUL values within 95% confidence interval: "
          f"{df['within_95_ci'].mean()*100:.1f}%")
    print(f"(Well-calibrated uncertainty should be close to 95%)")

    print(f"\nWorst 5 predictions (largest error):")
    print(df.nlargest(5, 'abs_error')[['engine_id', 'predicted', 'actual', 'abs_error']].to_string(index=False))

    df.to_csv(f"{DATA_DIR}/calibration_results.csv", index=False)
    print(f"\nSaved full results to {DATA_DIR}/calibration_results.csv")

    return df


if __name__ == "__main__":
    run_full_calibration()