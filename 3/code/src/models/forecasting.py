from dataclasses import dataclass


@dataclass
class ForecastModelBundle:
    model: object
    lags: int


def build_forecaster(cfg) -> ForecastModelBundle:
    name = cfg.forecasting.lower()

    if name == "linear":
        from sklearn.linear_model import LinearRegression
        return ForecastModelBundle(model=LinearRegression(), lags=cfg.lags)

    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(
            n_estimators=cfg.rf_n_estimators,
            max_depth=cfg.rf_max_depth,
            n_jobs=-1,
            random_state=42,
        )
        return ForecastModelBundle(model=m, lags=cfg.lags)

    raise ValueError(f"Unknown forecasting model: {cfg.forecasting}")
