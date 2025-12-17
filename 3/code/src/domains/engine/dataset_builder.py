import os
import numpy as np
import pandas as pd
from .augment import add_gaussian_noise, neighbor_replace


def _repeat_periods(df_period: pd.DataFrame, n_periods: int) -> pd.DataFrame:
    return pd.concat([df_period.copy() for _ in range(n_periods)], ignore_index=True)


def build_engine_train_test(
    normal_period: pd.DataFrame,
    anomaly_periods: list[pd.DataFrame],
    normal_periods: int,
    train_frac: float,
    anomaly_insert_periods: int,
    seed: int,
    augment_anomalies: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)

    base = _repeat_periods(normal_period, normal_periods)
    n = len(base)
    split = int(n * train_frac)

    train_df = base.iloc[:split].reset_index(drop=True)
    test_df = base.iloc[split:].reset_index(drop=True)
    labels = np.zeros(len(test_df), dtype=int)

    for _ in range(anomaly_insert_periods):
        a = anomaly_periods[int(rng.integers(0, len(anomaly_periods)))].copy()

        if augment_anomalies:
            a = neighbor_replace(a, p=0.05, seed=int(rng.integers(0, 1_000_000)))
            a = add_gaussian_noise(a, sigma=0.005, seed=int(rng.integers(0, 1_000_000)))

        insert_at = int(rng.integers(0, max(1, len(test_df) - len(a))))
        test_df = pd.concat([test_df.iloc[:insert_at], a, test_df.iloc[insert_at:]], ignore_index=True)

        new_labels = np.zeros(len(test_df), dtype=int)
        new_labels[:insert_at] = labels[:insert_at]
        new_labels[insert_at:insert_at+len(a)] = 1
        new_labels[insert_at+len(a):] = labels[insert_at:]
        labels = new_labels

    return train_df, test_df, labels


def save_engine_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame, labels: np.ndarray, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.csv")
    test_path = os.path.join(out_dir, "test.csv")
    lab_path = os.path.join(out_dir, "test_labels.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    pd.DataFrame({"label": labels}).to_csv(lab_path, index=False)
    return {"train_csv": train_path, "test_csv": test_path, "labels_csv": lab_path}
