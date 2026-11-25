"""Noise Clustering (Dave 1991): FCM plus one noise cluster at a constant distance delta.

delta^2 = lam * mean_ik d_ik^2, recomputed every iteration. A sample is flagged as noise when its
noise-cluster membership exceeds its membership in every real cluster.
"""
import numpy as np
from .base import FitResult, sqdist
from .fcm import fcm_best


def nc_memberships(D2, delta2, m):
    p = 1.0 / (m - 1.0)
    logw = np.vstack([-p * np.log(D2), np.full((1, D2.shape[1]), -p * np.log(delta2))])
    logw -= logw.max(0, keepdims=True)
    W = np.exp(logw)
    W /= W.sum(0, keepdims=True)
    return W[:-1], W[-1]


class NoiseClustering:
    name = "NC"
    paradigm, mechanism = "fuzzy", "M3 explicit outlier handling"

    def __init__(self, m=2.0, lam=0.5, n_init=5, tol=1e-5, max_iter=300, seed=0):
        self.m, self.lam, self.n_init, self.tol, self.max_iter, self.seed = m, lam, n_init, tol, max_iter, seed

    def fit(self, X, c):
        rng = np.random.default_rng(self.seed)
        V, _, D2, _, _, _ = fcm_best(X, c, self.m, self.n_init, rng)
        hist = []
        for it in range(self.max_iter):
            delta2 = self.lam * D2.mean()
            U, u0 = nc_memberships(D2, delta2, self.m)
            Um = U ** self.m
            hist.append(float((Um * D2).sum() + (u0 ** self.m).sum() * delta2))
            V_new = Um @ X / np.maximum(Um.sum(1, keepdims=True), 1e-300)
            shift = np.abs(V_new - V).max()
            V = V_new
            D2 = sqdist(X, V)
            if shift < self.tol:
                break
        delta2 = self.lam * D2.mean()
        U, u0 = nc_memberships(D2, delta2, self.m)
        hist.append(float(((U ** self.m) * D2).sum() + (u0 ** self.m).sum() * delta2))
        return FitResult(U.argmax(0), u0, u0 > U.max(0),
                         {"iterations": it + 1, "history": hist, "centers": V, "U": U, "u_noise": u0,
                          "delta2": delta2})
