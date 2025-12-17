import numpy as np
import pandas as pd


def add_gaussian_noise(df: pd.DataFrame, sigma: float = 0.005, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols] + rng.normal(0.0, sigma, size=out[num_cols].shape)
    return out


def neighbor_replace(df: pd.DataFrame, p: float = 0.05, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    mask = rng.random(size=out[num_cols].shape) < p
    shifted = out[num_cols].shift(1).bfill()
    out[num_cols] = np.where(mask, shifted.to_numpy(), out[num_cols].to_numpy())
    return out
