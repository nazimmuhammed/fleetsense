"""
FleetSense - LSTM RUL Prediction with Monte Carlo Dropout Uncertainty
=========================================================================
Trains an LSTM to predict Remaining Useful Life (RUL) from sensor sequences,
using RUL capping (standard technique in this research area - engines are
treated as "healthy" until close to failure, since early-life sensor data
carries little degradation signal) and Monte Carlo Dropout for uncertainty
quantification at inference time.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

WINDOW_SIZE = 30
RUL_CAP = 125  # standard published technique for this dataset

setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = [f's_{i}' for i in range(1, 22)]
CONSTANT_SENSORS = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']  # confirmed via your EDA
FEATURE_COLS = setting_names + [s for s in sensor_names if s not in CONSTANT_SENSORS]

DATA_DIR = "data"


def load_and_prepare():
    df = pd.read_csv(f"{DATA_DIR}/train_FD001_with_RUL.csv")
    df['RUL'] = df['RUL'].clip(upper=RUL_CAP)
    return df


def build_sequences(df, feature_cols, window_size):
    """
    Sliding window per engine - each sequence is `window_size` consecutive
    cycles, target is the RUL at the END of that window. This is how you
    turn a time-series into supervised learning examples for an LSTM.
    """
    sequences, targets = [], []
    for unit in df['unit_nr'].unique():
        unit_df = df[df['unit_nr'] == unit].sort_values('time_cycles')
        data = unit_df[feature_cols].values
        rul = unit_df['RUL'].values
        if len(data) < window_size:
            continue
        for i in range(len(data) - window_size + 1):
            sequences.append(data[i:i + window_size])
            targets.append(rul[i + window_size - 1])
    return np.array(sequences), np.array(targets)


class RULDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RULLSTM(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        x = self.relu(self.fc1(last))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)


def predict_with_uncertainty(model, x, n_samples=30):
    """
    Monte Carlo Dropout: keep dropout ACTIVE at inference time (model.train(),
    not model.eval()) and run multiple forward passes. The spread of these
    predictions approximates the model's uncertainty - published technique
    (Gal & Ghahramani, 2016) for uncertainty in deep learning without needing
    a separate probabilistic model architecture.
    """
    model.train()  # keep dropout active - this is the key trick
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(x).numpy())
    preds = np.array(preds)
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)
    return mean_pred, std_pred


def train():
    print("Loading and preparing data...")
    df = load_and_prepare()

    # Split by ENGINE, not by row - critical to avoid data leakage
    # (rows from the same engine are highly correlated in time)
    unit_ids = df['unit_nr'].unique()
    np.random.seed(42)
    np.random.shuffle(unit_ids)
    split_idx = int(len(unit_ids) * 0.8)
    train_units, val_units = unit_ids[:split_idx], unit_ids[split_idx:]

    train_df = df[df['unit_nr'].isin(train_units)].copy()
    val_df = df[df['unit_nr'].isin(val_units)].copy()

    scaler = MinMaxScaler()
    train_df[FEATURE_COLS] = scaler.fit_transform(train_df[FEATURE_COLS])
    val_df[FEATURE_COLS] = scaler.transform(val_df[FEATURE_COLS])
    joblib.dump(scaler, f"{DATA_DIR}/scaler.pkl")

    X_train, y_train = build_sequences(train_df, FEATURE_COLS, WINDOW_SIZE)
    X_val, y_val = build_sequences(val_df, FEATURE_COLS, WINDOW_SIZE)

    print(f"Train sequences: {X_train.shape}, Val sequences: {X_val.shape}")

    train_loader = DataLoader(RULDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(RULDataset(X_val, y_val), batch_size=32, shuffle=False)

    model = RULLSTM(n_features=len(FEATURE_COLS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    print("\nTraining...")
    for epoch in range(30):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    val_preds.extend(model(xb).numpy())
                    val_targets.extend(yb.numpy())
            val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
            print(f"Epoch {epoch+1}: train_loss={total_loss/len(train_loader):.2f}, val_RMSE={val_rmse:.2f}")

    # Final evaluation with uncertainty
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            val_preds.extend(model(xb).numpy())
            val_targets.extend(yb.numpy())

    final_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
    final_mae = mean_absolute_error(val_targets, val_preds)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Validation RMSE: {final_rmse:.2f} cycles")
    print(f"Validation MAE: {final_mae:.2f} cycles")
    print(f"Published benchmark reference (FD001, attention-GRU): ~12.83 RMSE")

    torch.save(model.state_dict(), f"{DATA_DIR}/rul_lstm_model.pt")
    print(f"\nModel saved to {DATA_DIR}/rul_lstm_model.pt")

    return model, X_val, y_val


if __name__ == "__main__":
    train()