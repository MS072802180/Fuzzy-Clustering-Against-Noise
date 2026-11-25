"""Label-free validity criteria for choosing PFCM parameters."""
import numpy as np
from sklearn.metrics import silhouette_score


def xie_beni(X, U, V, m):
    D2 = ((X[None, :, :] - V[:, None, :]) ** 2).sum(-1)
    sep = min(((V[i] - V[j]) ** 2).sum() for i in range(len(V)) for j in range(len(V)) if i != j)
    return float(((U ** m) * D2).sum() / (X.shape[0] * max(sep, 1e-12)))


def _sil(X, labels, rng, max_n=1500):
    if len(np.unique(labels)) < 2:
        return -1.0
    idx = rng.choice(len(X), size=min(max_n, len(X)), replace=False)
    if len(np.unique(labels[idx])) < 2:
        return -1.0
    return float(silhouette_score(X[idx], labels[idx]))


def criteria(X, fit, m, trim=0.10, seed=0):
    """Returns criteria where HIGHER is better (Xie-Beni is negated)."""
    rng = np.random.default_rng(seed)
    U, V, T = fit.info["U"], fit.info["centers"], fit.info["T"]
    labels = fit.labels
    keep = T.max(0) >= np.quantile(T.max(0), trim)   # drop the least typical 10%
    return {
        "neg_xie_beni": -xie_beni(X, U, V, m),
        "silhouette": _sil(X, labels, rng),
        "trimmed_silhouette": _sil(X[keep], labels[keep], rng),
    }
