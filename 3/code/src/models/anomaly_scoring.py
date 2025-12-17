import numpy as np
from sklearn.cluster import KMeans


def norm_score(residuals: np.ndarray, ord: int = 1) -> np.ndarray:
    if ord == 1:
        return np.sum(np.abs(residuals), axis=1)
    if ord == 2:
        return np.sqrt(np.sum(residuals ** 2, axis=1))
    raise ValueError("ord must be 1 or 2")


def make_windows(vec: np.ndarray, window: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    N = len(vec)
    if N < window:
        raise ValueError(f"N={N} < window={window}")
    starts = np.arange(0, N - window + 1, stride)
    W = np.stack([vec[s:s+window] for s in starts], axis=0)
    return W, starts


def window_scores_to_point_scores(w_scores: np.ndarray, starts: np.ndarray, N: int, window: int) -> np.ndarray:
    acc = np.zeros(N, dtype=float)
    cnt = np.zeros(N, dtype=float)
    for score, s in zip(w_scores, starts):
        acc[s:s+window] += score
        cnt[s:s+window] += 1.0
    cnt[cnt == 0] = 1.0
    return acc / cnt


class KMeansWindowScorer:
    def __init__(self, k: int, window: int, stride: int, train_samples: int = 2000, seed: int = 42):
        self.k = k
        self.window = window
        self.stride = stride
        self.train_samples = train_samples
        self.seed = seed
        self.km = None


    def fit(self, base_score: np.ndarray):
        W, _ = make_windows(base_score, self.window, self.stride)

        if len(W) == 0:
            raise ValueError("No windows available for KMeans fitting")

        k = min(self.k, len(W)) 

        if len(W) > self.train_samples:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(W), size=self.train_samples, replace=False)
            W_fit = W[idx]
        else:
            W_fit = W

        self.km = KMeans(n_clusters=k, n_init="auto", random_state=self.seed)
        self.km.fit(W_fit)
        return self


    def score(self, base_score: np.ndarray) -> np.ndarray:
        if self.km is None:
            raise RuntimeError("KMeansWindowScorer is not fitted")
        W, starts = make_windows(base_score, self.window, self.stride)
        # расстояния до ближайших центров
        dists = self.km.transform(W)
        w_scores = np.min(dists, axis=1)
        point_scores = window_scores_to_point_scores(w_scores, starts, len(base_score), self.window)
        return point_scores
