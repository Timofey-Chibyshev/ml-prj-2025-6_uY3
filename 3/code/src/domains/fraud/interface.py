import os
import pandas as pd
from .raw_loader import load_fraud_csv
from .dataset_builder import aggregate_to_timeseries, split_train_test_no_fraud_prefix


def get_fraud_dataset(cfg) -> dict:
    cache = cfg.cache_dir
    train_csv = os.path.join(cache, "train.csv") if cache else None
    test_csv = os.path.join(cache, "test.csv") if cache else None
    labels_csv = os.path.join(cache, "labels.csv") if cache else None

    if cache and os.path.exists(train_csv) and os.path.exists(test_csv) and os.path.exists(labels_csv):
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        labels = pd.read_csv(labels_csv)["label"].to_numpy(dtype=int)
        # "period" для совместимости с остальным кодом
        return {"train_df": train_df, "test_df": test_df, "labels": labels, "period": 1} 

    raw = load_fraud_csv(cfg.csv_path)
    ts_df, labels = aggregate_to_timeseries(
        raw,
        time_col=cfg.time_col,
        label_col=cfg.label_col,
        agg_seconds=cfg.agg_seconds,
    )

    train_df, test_df, labels = split_train_test_no_fraud_prefix(ts_df, labels, min_train_bins=cfg.min_train_bins)

    if cache:
        os.makedirs(cache, exist_ok=True)
        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)
        pd.DataFrame({"label": labels}).to_csv(labels_csv, index=False)

    return {"train_df": train_df, "test_df": test_df, "labels": labels, "period": 1}
