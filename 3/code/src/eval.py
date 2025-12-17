import numpy as np
from sklearn.metrics import roc_auc_score


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    s = np.asarray(scores).reshape(-1)
    y = np.asarray(labels).reshape(-1)[: len(s)]
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))
