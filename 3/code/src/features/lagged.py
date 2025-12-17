import numpy as np


def make_lagged_xy(data: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    T, F = data.shape
    if T <= lags:
        raise ValueError(f"Not enough data: T={T} <= lags={lags}")

    X = np.zeros((T - lags, lags * F), dtype=float)
    y = np.zeros((T - lags, F), dtype=float)

    for i in range(lags, T):
        window = data[i - lags:i, :]
        X[i - lags, :] = window.reshape(-1)
        y[i - lags, :] = data[i, :]
    return X, y


def align_labels(labels: np.ndarray, lags: int) -> np.ndarray:
    return labels[lags:]
