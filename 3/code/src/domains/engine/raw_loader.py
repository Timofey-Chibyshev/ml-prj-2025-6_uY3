import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_engine_scenarios(normal_csv: str, anomaly_csvs: list[str]) -> dict:
    normal = load_csv(normal_csv)
    anomalies = [load_csv(p) for p in anomaly_csvs]
    return {"normal": normal, "anomalies": anomalies}
