import os
import numpy as np
from joblib import dump
from .features.lagged import make_lagged_xy, align_labels
from .models.forecasting import build_forecaster
from .models.anomaly_scoring import norm_score, KMeansWindowScorer
from .eval import roc_auc
from .plotting import plot_scores


def run_timeseries_df(train_df, test_df, labels, model_cfg, scoring_cfg, out_dir: str, tag: str):
    train_num = train_df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    test_num = test_df.select_dtypes(include=[np.number]).to_numpy(dtype=float)

    forecaster = build_forecaster(model_cfg)
    lags = forecaster.lags

    x_train, y_train = make_lagged_xy(train_num, lags)
    x_test, y_test = make_lagged_xy(test_num, lags)
    y_labels = align_labels(labels, lags)

    forecaster.model.fit(x_train, y_train)

    y_hat = forecaster.model.predict(x_test)
    residuals = (y_test - y_hat)

    s_norm = norm_score(residuals, ord=scoring_cfg.norm_ord)

    y_hat_train = forecaster.model.predict(x_train)
    residuals_train = (y_train - y_hat_train)
    s_norm_train = norm_score(residuals_train, ord=scoring_cfg.norm_ord)

    kscorer = KMeansWindowScorer(
        k=scoring_cfg.kmeans_k,
        window=scoring_cfg.kmeans_window,
        stride=scoring_cfg.kmeans_stride,
        train_samples=scoring_cfg.kmeans_train_samples,
        seed=42,
    )
    kscorer.fit(s_norm_train)
    s_kmeans = kscorer.score(s_norm)

    auc_norm = roc_auc(s_norm, y_labels)
    auc_km = roc_auc(s_kmeans, y_labels)

    os.makedirs(out_dir, exist_ok=True)
    dump(forecaster, os.path.join(out_dir, f"{tag}_forecast_model.joblib"))
    dump(kscorer, os.path.join(out_dir, f"{tag}_kmeans_scorer.joblib"))

    plot_scores(s_norm, y_labels, os.path.join(out_dir, f"{tag}_scores_norm.png"), title=f"{tag}: Norm AUC={auc_norm:.3f}")
    plot_scores(s_kmeans, y_labels, os.path.join(out_dir, f"{tag}_scores_kmeans.png"), title=f"{tag}: KMeans AUC={auc_km:.3f}")

    return {"auc_norm": auc_norm, "auc_kmeans": auc_km, "n_test_points": len(s_norm)}


def run_engine(engine_cfg, model_cfg, scoring_cfg, out_dir: str):
    from .domains.engine.interface import get_engine_dataset
    ds = get_engine_dataset(engine_cfg)
    return run_timeseries_df(ds["train_df"], ds["test_df"], ds["labels"], model_cfg, scoring_cfg, out_dir, tag="engine")


def run_fraud(fraud_cfg, model_cfg, scoring_cfg, out_dir: str):
    from .domains.fraud.interface import get_fraud_dataset
    ds = get_fraud_dataset(fraud_cfg)
    return run_timeseries_df(ds["train_df"], ds["test_df"], ds["labels"], model_cfg, scoring_cfg, out_dir, tag="fraud")


def run(domain: str, domain_cfg, model_cfg, scoring_cfg, out_dir: str):
    if domain == "engine":
        return run_engine(domain_cfg, model_cfg, scoring_cfg, out_dir)
    if domain == "fraud":
        return run_fraud(domain_cfg, model_cfg, scoring_cfg, out_dir)
    raise ValueError(f"Unknown domain: {domain}")
