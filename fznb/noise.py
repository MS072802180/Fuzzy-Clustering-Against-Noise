"""The shared noise-injection protocol.

Three noise types, each controlled by one contamination level r in [0, 1):

* ``uniform``  : background noise (Banfield & Raftery 1993). Outlying points drawn uniformly
                 from the bounding box of the clean data, enlarged by 50% on every side.
                 r is the fraction of the *final* data set that is noise.
* ``heavy``    : heavy-tailed scatter. Points drawn from a multivariate Student-t (df = 1)
                 around a randomly chosen cluster, using that cluster's mean and covariance,
                 and kept only if they lie outside the cluster's 99.9% Mahalanobis ellipsoid
                 (radius capped at 10x that ellipsoid). r as above.
* ``rowwise``  : sample-level corruption (the original project proposal's protocol). A fraction r
                 of the samples is selected and Student-t noise (df = 3, scale 3) is added to every
                 feature of those samples. No samples are added.
* ``cellwise`` : feature-level corruption (Zaccaria et al. 2026). A fraction r of the samples
                 is selected and, in each, ceil(10% of d) cells are replaced by +-U(5, 10)
                 (in standardized units). No samples are added.

Every function returns (X_noisy, is_injected, is_contaminated):
  is_injected     -> True for points that were added (excluded from clustering metrics)
  is_contaminated -> True for every point that should be flagged as an outlier
"""
import numpy as np
from scipy.stats import chi2

NOISE_TYPES = ("uniform", "heavy", "rowwise", "cellwise")


def _n_added(n, r):
    return int(round(r / (1.0 - r) * n)) if r > 0 else 0


def uniform(X, y, r, rng):
    n, d = X.shape
    k = _n_added(n, r)
    lo, hi = X.min(0), X.max(0)
    span = hi - lo
    Z = rng.uniform(lo - 0.5 * span, hi + 0.5 * span, size=(k, d))
    return _stack(X, Z)


def heavy(X, y, r, rng, df=1.0, max_ratio=10.0):
    n, d = X.shape
    k = _n_added(n, r)
    labels, counts = np.unique(y, return_counts=True)
    thr = chi2.ppf(0.999, d)
    out = []
    while len(out) < k:
        c = rng.choice(labels, p=counts / counts.sum())
        Xc = X[y == c]
        mu = Xc.mean(0)
        S = np.cov(Xc, rowvar=False) + 1e-3 * np.eye(d)
        L = np.linalg.cholesky(S)
        g = rng.standard_normal(d)
        w = rng.chisquare(df) / df
        z = g / np.sqrt(w)                      # standard t in whitened space
        m2 = float(z @ z)                       # squared Mahalanobis distance
        if m2 <= thr:
            continue
        cap = max_ratio ** 2 * thr
        if m2 > cap:
            z *= np.sqrt(cap / m2)
        out.append(mu + L @ z)
    Z = np.array(out).reshape(k, d)
    return _stack(X, Z)


def cellwise(X, y, r, rng, frac_cells=0.10, low=5.0, high=10.0):
    n, d = X.shape
    Xn = X.copy()
    rows = rng.choice(n, size=int(round(r * n)), replace=False) if r > 0 else np.array([], int)
    k = max(1, int(np.ceil(frac_cells * d)))
    for i in rows:
        cols = rng.choice(d, size=k, replace=False)
        Xn[i, cols] = rng.choice([-1.0, 1.0], size=k) * rng.uniform(low, high, size=k)
    contaminated = np.zeros(n, bool)
    contaminated[rows] = True
    return Xn, np.zeros(n, bool), contaminated


def rowwise(X, y, r, rng, df=3.0, scale=3.0):
    n, d = X.shape
    rows = rng.choice(n, size=int(round(r * n)), replace=False) if r > 0 else np.array([], int)
    Xn = X.copy()
    Xn[rows] += scale * rng.standard_t(df, size=(len(rows), d))
    contaminated = np.zeros(n, bool)
    contaminated[rows] = True
    return Xn, np.zeros(n, bool), contaminated


def _stack(X, Z):
    Xn = np.vstack([X, Z])
    injected = np.r_[np.zeros(len(X), bool), np.ones(len(Z), bool)]
    return Xn, injected, injected.copy()


def inject(X, y, noise_type, r, seed):
    """Apply one noise type at level r. Deterministic given seed."""
    rng = np.random.default_rng(seed)
    if r <= 0:
        n = len(X)
        return X.copy(), np.zeros(n, bool), np.zeros(n, bool)
    fn = {"uniform": uniform, "heavy": heavy, "rowwise": rowwise, "cellwise": cellwise}[noise_type]
    return fn(X, y, r, rng)
