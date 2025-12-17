import os
import pandas as pd
from .raw_loader import load_engine_scenarios
from .dataset_builder import build_engine_train_test, save_engine_dataset


def get_engine_dataset(cfg) -> dict:
    cache = cfg.cache_dir
    train_csv = os.path.join(cache, "train.csv") if cache else None
    test_csv = os.path.join(cache, "test.csv") if cache else None
    labels_csv = os.path.join(cache, "test_labels.csv") if cache else None

    if cache and os.path.exists(train_csv) and os.path.exists(test_csv) and os.path.exists(labels_csv):
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        labels = pd.read_csv(labels_csv)["label"].to_numpy()
        return {"train_df": train_df, "test_df": test_df, "labels": labels, "period": cfg.period}

    scens = load_engine_scenarios(cfg.normal_csv, cfg.anomaly_csvs)
    train_df, test_df, labels = build_engine_train_test(
        normal_period=scens["normal"],
        anomaly_periods=scens["anomalies"],
        normal_periods=cfg.normal_periods,
        train_frac=cfg.train_frac,
        anomaly_insert_periods=cfg.anomaly_insert_periods,
        seed=cfg.seed,
    )

    if cache:
        save_engine_dataset(train_df, test_df, labels, cache)

    return {"train_df": train_df, "test_df": test_df, "labels": labels, "period": cfg.period}
