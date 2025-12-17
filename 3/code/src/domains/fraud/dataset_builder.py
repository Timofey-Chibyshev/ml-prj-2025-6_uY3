import numpy as np
import pandas as pd


def aggregate_to_timeseries(df: pd.DataFrame, time_col: str, label_col: str, agg_seconds: int) -> tuple[pd.DataFrame, np.ndarray]:
    d = df.copy()

    t = d[time_col].astype(float).to_numpy()
    bins = (t // agg_seconds).astype(int)
    d["_bin"] = bins

    grp = d.groupby("_bin", sort=True)

    amount = grp["Amount"]
    fraud = grp[label_col]

    ts_df = pd.DataFrame({
        "tx_count": grp.size().astype(float),
        "amount_sum": amount.sum().astype(float),
        "amount_mean": amount.mean().astype(float),
        "amount_std": amount.std(ddof=0).fillna(0.0).astype(float),
        "amount_max": amount.max().astype(float),
        "amount_min": amount.min().astype(float),
        "fraud_count": fraud.sum().astype(float),
        "fraud_rate": (fraud.sum() / grp.size()).fillna(0.0).astype(float),
    }).reset_index(drop=True)

    labels = (ts_df["fraud_count"].to_numpy() > 0).astype(int)

    ts_df = ts_df.drop(columns=["fraud_count", "fraud_rate"])

    return ts_df, labels


def _longest_zero_run(labels: np.ndarray) -> tuple[int, int]:
    best_s, best_e = 0, 0
    s = None
    for i, v in enumerate(labels):
        if v == 0 and s is None:
            s = i
        if v == 1 and s is not None:
            if i - s > best_e - best_s:
                best_s, best_e = s, i
            s = None
    if s is not None:
        if len(labels) - s > best_e - best_s:
            best_s, best_e = s, len(labels)
    return best_s, best_e


def split_train_test_no_fraud_prefix(ts_df: pd.DataFrame, labels: np.ndarray, min_train_bins: int = 300) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    first_fraud = np.argmax(labels == 1) if np.any(labels == 1) else len(labels)

    if first_fraud >= min_train_bins:
        train_df = ts_df.iloc[:first_fraud].reset_index(drop=True)
    else:
        s, e = _longest_zero_run(labels)
        train_df = ts_df.iloc[s:e].reset_index(drop=True)

    test_df = ts_df.reset_index(drop=True)
    return train_df, test_df, labels
