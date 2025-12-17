import argparse
from src.config import EngineConfig, FraudConfig, ModelConfig, ScoringConfig
from src.pipeline import run


def main():
    p = argparse.ArgumentParser()

    # какая предметная область
    p.add_argument("--domain", choices=["engine", "fraud"], required=True)

    # настройка модели предсказания (linear - линейная регрессия, rf - случайный лес)
    p.add_argument("--forecasting", choices=["linear", "rf"], default="rf")
    p.add_argument("--lags", type=int, default=25)
    p.add_argument("--rf-n", type=int, default=200)

    # настройка модели обнаружения аномалий k-means
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--stride", type=int, default=5)

    # параметры для сборки датасета для --domain engine
    p.add_argument("--period", type=int, default=100)
    p.add_argument("--normal-periods", type=int, default=200)
    p.add_argument("--anom-inserts", type=int, default=5)

    # куда выводить результаты
    p.add_argument("--out", default="artifacts")

    # где брать данные
    p.add_argument("--engine-normal", default="data/engine/raw/Scenario_Normal.csv")
    p.add_argument("--engine-anom", nargs="+", default=[
        "data/engine/raw/Scenario_One.csv",
        "data/engine/raw/Scenario_Two.csv",
        "data/engine/raw/Scenario_Three.csv",
        "data/engine/raw/Scenario_Four.csv",
    ])

    p.add_argument("--fraud-csv", default="data/fraud/raw/creditcard.csv")
    p.add_argument("--fraud-agg-sec", type=int, default=60)
    
    args = p.parse_args()

    model_cfg = ModelConfig(
        forecasting=args.forecasting,
        lags=args.lags,
        rf_n_estimators=args.rf_n,
    )
    scoring_cfg = ScoringConfig(
        kmeans_k=args.k,
        kmeans_window=args.window,
        kmeans_stride=args.stride,
    )

    if args.domain == "engine":
        domain_cfg = EngineConfig(
            normal_csv=args.engine_normal,
            anomaly_csvs=args.engine_anom,
            period=args.period,
            normal_periods=args.normal_periods,
            anomaly_insert_periods=args.anom_inserts,
        )
    elif args.domain == "fraud":
        domain_cfg = FraudConfig(
            csv_path=args.fraud_csv,
            agg_seconds=args.fraud_agg_sec,
        )
    else:
        raise ValueError("unknown domain")

    res = run(args.domain, domain_cfg, model_cfg, scoring_cfg, args.out)
    print(res)

if __name__ == "__main__":
    main()
