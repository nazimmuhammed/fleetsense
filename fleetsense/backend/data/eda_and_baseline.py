"""
FleetSense - Week 1: Data Loading + EDA
==========================================
NASA C-MAPSS Turbofan Engine Degradation Dataset (Saxena & Goebel, 2008)
FD001 subset: 100 training engines, 100 test engines, 1 operating condition, 1 fault mode.
"""

import pandas as pd
import numpy as np

index_names = ['unit_nr', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = [f's_{i}' for i in range(1, 22)]
col_names = index_names + setting_names + sensor_names

DATA_DIR = "data"


def load_data():
    train = pd.read_csv(f"{DATA_DIR}/train_FD001.txt", sep=r'\s+', header=None, names=col_names)
    test = pd.read_csv(f"{DATA_DIR}/test_FD001.txt", sep=r'\s+', header=None, names=col_names)
    rul_test = pd.read_csv(f"{DATA_DIR}/RUL_FD001.txt", sep=r'\s+', header=None, names=['RUL'])
    return train, test, rul_test


def compute_rul_train(train: pd.DataFrame) -> pd.DataFrame:
    max_cycle = train.groupby('unit_nr')['time_cycles'].max().reset_index()
    max_cycle.columns = ['unit_nr', 'max_cycle']
    train = train.merge(max_cycle, on='unit_nr', how='left')
    train['RUL'] = train['max_cycle'] - train['time_cycles']
    train.drop('max_cycle', axis=1, inplace=True)
    return train


def basic_eda(train: pd.DataFrame, test: pd.DataFrame, rul_test: pd.DataFrame):
    print("=" * 60)
    print("BASIC EDA - FleetSense / NASA C-MAPSS FD001")
    print("=" * 60)
    print(f"Training engines: {train['unit_nr'].nunique()}")
    print(f"Training records: {len(train):,}")
    print(f"Test engines: {test['unit_nr'].nunique()}")
    print(f"Test records: {len(test):,}")

    print(f"\nCycle length per engine (training) - min/max/mean:")
    cycle_counts = train.groupby('unit_nr')['time_cycles'].max()
    print(f"  Min: {cycle_counts.min()}, Max: {cycle_counts.max()}, Mean: {cycle_counts.mean():.1f}")

    print(f"\nRUL distribution (training, derived):")
    print(train['RUL'].describe())

    print(f"\nSensor columns: {len(sensor_names)}")
    print(f"\nChecking for constant/near-constant sensors (uninformative):")
    for col in sensor_names:
        std = train[col].std()
        if std < 0.001:
            print(f"  {col}: std={std:.6f} -> CONSTANT, likely uninformative")

    print(f"\nMissing values total: {train.isnull().sum().sum()}")


if __name__ == "__main__":
    train, test, rul_test = load_data()
    train = compute_rul_train(train)
    basic_eda(train, test, rul_test)

    train.to_csv(f"{DATA_DIR}/train_FD001_with_RUL.csv", index=False)
    print(f"\nSaved processed training data with RUL labels to {DATA_DIR}/train_FD001_with_RUL.csv")