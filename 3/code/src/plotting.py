import numpy as np
import matplotlib.pyplot as plt


def _anomaly_runs(labels: np.ndarray):
    runs = []
    in_run = False
    s = 0
    for i, v in enumerate(labels):
        if v == 1 and not in_run:
            in_run = True
            s = i
        elif v == 0 and in_run:
            runs.append((s, i))
            in_run = False
    if in_run:
        runs.append((s, len(labels)))
    return runs


def plot_scores(scores, labels, path, title=None):
    s = np.asarray(scores).reshape(-1)
    y = np.asarray(labels).reshape(-1)

    n = min(len(s), len(y))
    s = s[:n]
    y = y[:n]

    x = np.arange(n)

    fig = plt.figure()
    ax = plt.gca()

    ax.plot(x, s, linewidth=1.0)

    runs = _anomaly_runs(y)
    for (a, b) in runs:
        ax.axvspan(a, b - 1, alpha=0.2)

    ax.set_xlabel("time index")
    ax.set_ylabel("anomaly score")
    if title:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
