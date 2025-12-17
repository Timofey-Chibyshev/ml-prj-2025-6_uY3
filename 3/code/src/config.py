from dataclasses import dataclass
from typing import Optional


@dataclass
class EngineConfig:
    normal_csv: str
    anomaly_csvs: list[str]
    period: int
    normal_periods: int = 200
    train_frac: float = 0.75
    anomaly_insert_periods: int = 5
    seed: int = 42
    cache_dir: Optional[str] = "data/engine/processed"


@dataclass
class FraudConfig:
    csv_path: str
    time_col: str = "Time"
    label_col: str = "Class"
    agg_seconds: int = 60
    min_train_bins: int = 300
    cache_dir: Optional[str] = "data/fraud/processed"


@dataclass
class ModelConfig:
    forecasting: str = "rf" # "linear" или "rf"
    lags: int = 25
    rf_n_estimators: int = 200
    rf_max_depth: Optional[int] = None


@dataclass
class ScoringConfig:
    norm_ord: int = 1
    kmeans_k: int = 10
    kmeans_window: int = 50
    kmeans_stride: int = 5
    kmeans_train_samples: int = 2000


@dataclass
class RunConfig:
    domain: str = "engine"
    out_dir: str = "artifacts"
