"""ST-RNC: Self-Tuning Robust Noise Clustering (the model proposed in this project).

ST-RNC enhances Dave's noise clustering (NC, Dave 1991) in two ways:

1. A parameter-free, chi-square-calibrated noise distance (RNC).
   NC sets the noise distance as delta^2 = lambda * mean_ik d_ik^2. That mean runs over *all*
   clusters (so it grows with the number of clusters and their separation), it is inflated by the
   very outliers it should reject, and lambda must be tuned. RNC instead uses each sample's
   squared distance to its *nearest* centre, takes the median (robust to up to 50% outliers),
   and rescales it from the median to the q-quantile of a chi-square distribution with d degrees
   of freedom:
        delta^2 = median_k( min_i d_ik^2 ) * chi2_q(d) / chi2_0.5(d),     q = 0.99.
   If clusters are roughly Gaussian, min_i d_ik^2 / sigma^2 is roughly chi2(d) for inliers, so a
   point is claimed by the noise cluster when it lies beyond the 99% contour of a typical cluster.
   The rule adapts to dimension and needs no tuning.

2. Label-free selection of the fuzzifier m (ST).
   The canonical m = 2 makes memberships nearly uniform in high dimension or with many clusters,
   so centres merge (Winkler et al. 2011). ST fits the model for each m in a small grid and keeps
   the m with the lowest Xie-Beni index (Xie & Beni 1991) computed on the samples that are *not*
   claimed by the noise cluster, rejecting partitions in which a cluster has collapsed (some
   cluster receives under 1% of the inliers).

The ablation variants are registered as well: RNC (component 1 only, m = 2) and ST-NC
(component 2 only, with NC's original lambda rule). q and the m grid are fixed a priori.
"""
import numpy as np
from scipy.stats import chi2

from .base import FitResult, sqdist
from .fcm import fcm_best
from .noise_clustering import nc_memberships, NoiseClustering


def chi2_delta2(D2, d, q=0.99):
    """Noise distance: median nearest-centre squared distance, rescaled to the chi2_d q-quantile."""
    return float(np.median(D2.min(0)) * chi2.ppf(q, d) / chi2.ppf(0.5, d))


class RobustNoiseClustering:
    """RNC: noise clustering with the chi-square-calibrated noise distance (component 1)."""
    name = "RNC"
    paradigm, mechanism = "fuzzy", "M3 explicit outlier handling"

    def __init__(self, m=2.0, q=0.99, n_init=5, tol=1e-5, max_iter=300, seed=0):
        self.m, self.q, self.n_init, self.tol, self.max_iter, self.seed = m, q, n_init, tol, max_iter, seed

    def fit(self, X, c):
        rng = np.random.default_rng(self.seed)
        d = X.shape[1]
        V, _, D2, _, _, _ = fcm_best(X, c, self.m, self.n_init, rng)
        hist = []
        for it in range(self.max_iter):
            delta2 = chi2_delta2(D2, d, self.q)
            U, u0 = nc_memberships(D2, delta2, self.m)
            Um = U ** self.m
            hist.append(float((Um * D2).sum() + (u0 ** self.m).sum() * delta2))
            V_new = Um @ X / np.maximum(Um.sum(1, keepdims=True), 1e-300)
            shift = np.abs(V_new - V).max()
            V = V_new
            D2 = sqdist(X, V)
            if shift < self.tol:
                break
        delta2 = chi2_delta2(D2, d, self.q)
        U, u0 = nc_memberships(D2, delta2, self.m)
        hist.append(float(((U ** self.m) * D2).sum() + (u0 ** self.m).sum() * delta2))
        return FitResult(U.argmax(0), u0, u0 > U.max(0),
                         {"iterations": it + 1, "history": hist, "centers": V, "U": U, "u_noise": u0,
                          "delta2": delta2, "m": self.m})


def inlier_xie_beni(X, fit, m):
    """Xie-Beni index on the samples not claimed by the noise cluster.
    Returns (collapsed, value): collapsed = True if some cluster has no inlier assigned to it."""
    U, V = fit.info["U"], fit.info["centers"]
    c = len(V)
    keep = ~fit.outlier_flag if fit.outlier_flag is not None else np.ones(len(X), bool)
    collapsed = keep.sum() < 2 * c or len(np.unique(fit.labels[keep])) < c
    if keep.sum() == 0:
        return True, np.inf
    D2 = sqdist(X[keep], V)
    sep = min(((V[i] - V[j]) ** 2).sum() for i in range(c) for j in range(c) if i != j)
    return collapsed, float(((U[:, keep] ** m) * D2).sum() / (keep.sum() * max(sep, 1e-12)))


def collapsed(fit, min_share=0.01):
    """A partition has collapsed if, among the samples not claimed by the noise cluster, some
    cluster receives no samples or less than min_share of them (merged / coincident centres)."""
    c = len(fit.info["centers"])
    keep = ~fit.outlier_flag if fit.outlier_flag is not None else np.ones(len(fit.labels), bool)
    if keep.sum() < 2 * c:
        return True
    counts = np.bincount(fit.labels[keep], minlength=c)
    return bool(counts.min() < min_share * keep.sum())


class _SelfTuning:
    """Chooses the fuzzifier m without labels.

    rule = "xb" (default): the non-collapsed fit with the lowest inlier Xie-Beni index.
    rule = "guard": the largest m in m_grid (the canonical m = 2 first) whose partition has not
        collapsed; if all collapse, the smallest m.
    The default was chosen on the development suite (configs/small.yaml datasets: Iris, Wine, WDBC,
    Digits, small synthetic), which is disjoint from the evaluation suite (configs/full.yaml):
    mean noisy ARI 0.662 (xb) vs 0.653 (guard) vs 0.632 (NC), 3 seeds, r in {0.1, 0.2}.
    """
    paradigm = "fuzzy"
    base = None

    def __init__(self, m_grid=(1.2, 1.5, 2.0), rule="xb", seed=0, **base_params):
        self.m_grid, self.rule, self.seed, self.base_params = tuple(sorted(m_grid)), rule, seed, base_params

    def fit(self, X, c):
        scores = {}
        if self.rule == "guard":
            for m in sorted(self.m_grid, reverse=True):          # 2.0, 1.5, 1.2
                fit = self.base(m=m, seed=self.seed, **self.base_params).fit(X, c)
                scores[m] = collapsed(fit)
                if not scores[m]:
                    break
            chosen = m
        else:
            best = None
            for m in self.m_grid:
                fit_m = self.base(m=m, seed=self.seed, **self.base_params).fit(X, c)
                col, xb = inlier_xie_beni(X, fit_m, m)
                scores[m] = (col or collapsed(fit_m), xb)
                if best is None or scores[m] < best[0]:
                    best = (scores[m], m, fit_m)
            _, chosen, fit = best
        fit.info.update({"m": chosen, "m_scores": scores, "rule": self.rule})
        return fit


class SelfTuningRNC(_SelfTuning):
    """ST-RNC: the proposed model (both components)."""
    name = "ST-RNC"
    mechanism = "M3 explicit outlier handling (proposed)"
    base = RobustNoiseClustering


class SelfTuningNC(_SelfTuning):
    """ST-NC: fuzzifier selection on top of the original NC (component 2 only)."""
    name = "ST-NC"
    mechanism = "M3 explicit outlier handling"
    base = NoiseClustering
