"""
FleetSense - Autoencoder Anomaly Detection
==============================================
Independent, unsupervised early-warning layer. Trains ONLY on healthy
(early-life) engine data, learning to reconstruct normal sensor patterns.
When fed degraded/anomalous data, reconstruction error spikes - this
requires no RUL labels at all, a genuinely different ML paradigm from
the supervised LSTM.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import joblib

from train_lstm import FEATURE_COLS, DATA_DIR

HEALTHY_CYCLE_THRESHOLD = 0.3  # use only the first 30% of each engine's life as "healthy" training data


class Autoencoder(nn.Module):
    def __init__(self, n_features, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, n_features),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def prepare_healthy_data():
    df = pd.read_csv(f"{DATA_DIR}/train_FD001_with_RUL.csv")

    # Keep only early-life cycles per engine (assumed healthy)
    healthy_rows = []
    for unit in df['unit_nr'].unique():
        unit_df = df[df['unit_nr'] == unit].sort_values('time_cycles')
        cutoff = int(len(unit_df) * HEALTHY_CYCLE_THRESHOLD)
        healthy_rows.append(unit_df.iloc[:cutoff])
    healthy_df = pd.concat(healthy_rows)

    scaler = joblib.load(f"{DATA_DIR}/scaler.pkl")  # reuse the SAME scaler as the LSTM for consistency
    healthy_scaled = scaler.transform(healthy_df[FEATURE_COLS])
    return healthy_scaled, scaler


def train_autoencoder():
    print("Preparing healthy (early-life) training data...")
    X_healthy, scaler = prepare_healthy_data()
    print(f"Healthy training samples: {X_healthy.shape}")

    X_tensor = torch.tensor(X_healthy, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tensor, X_tensor), batch_size=32, shuffle=True)

    model = Autoencoder(n_features=len(FEATURE_COLS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    print("\nTraining autoencoder on healthy data only...")
    for epoch in range(50):
        total_loss = 0
        for xb, _ in loader:
            optimizer.zero_grad()
            recon = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: reconstruction loss = {total_loss/len(loader):.5f}")

    torch.save(model.state_dict(), f"{DATA_DIR}/autoencoder.pt")
    print(f"\nModel saved to {DATA_DIR}/autoencoder.pt")

    # Establish a healthy-baseline reconstruction error threshold
    model.eval()
    with torch.no_grad():
        recon = model(X_tensor)
        errors = torch.mean((recon - X_tensor) ** 2, dim=1).numpy()

    threshold = np.percentile(errors, 95)  # 95th percentile of healthy errors = anomaly threshold
    print(f"\nHealthy reconstruction error stats: mean={errors.mean():.5f}, 95th pct={threshold:.5f}")
    np.save(f"{DATA_DIR}/anomaly_threshold.npy", threshold)

    return model, threshold


if __name__ == "__main__":
    train_autoencoder()