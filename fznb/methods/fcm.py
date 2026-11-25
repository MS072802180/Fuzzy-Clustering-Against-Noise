"""Fuzzy C-Means (Bezdek 1981)."""
import numpy as np
from .base import FitResult, sqdist, init_centers, fcm_memberships


def fcm_core(X, c, m=2.0, tol=1e-5, max_iter=300, rng=None, V0=None):
    """One FCM run from k-means++ centres. Returns (V, U, D2, J, iterations, objective history)."""
    V = init_centers(X, c, rng) if V0 is None else V0.copy()
    hist = []
    for it in range(max_iter):
        D2 = sqdist(X, V)
        U = fcm_memberships(D2, m)
        Um = U ** m
        hist.append(float((Um * D2).sum()))
        V_new = Um @ X / Um.sum(1, keepdims=True)
        shift = np.abs(V_new - V).max()
        V = V_new
        if shift < tol:
            break
    D2 = sqdist(X, V)
    U = fcm_memberships(D2, m)
    J = float(((U ** m) * D2).sum())
    hist.append(J)
    return V, U, D2, J, it + 1, hist


def fcm_best(X, c, m=2.0, n_init=5, rng=None, **kw):
    """Best of n_init restarts (lowest objective)."""
    best = None
    for _ in range(n_init):
        out = fcm_core(X, c, m=m, rng=rng, **kw)
        if best is None or out[3] < best[3]:
            best = out
    return best


class FCM:
    name = "FCM"
    paradigm, mechanism = "fuzzy", "baseline"

    def __init__(self, m=2.0, n_init=5, tol=1e-5, max_iter=300, seed=0):
        self.m, self.n_init, self.tol, self.max_iter, self.seed = m, n_init, tol, max_iter, seed

    def fit(self, X, c):
        rng = np.random.default_rng(self.seed)
        V, U, D2, J, it, hist = fcm_best(X, c, self.m, self.n_init, rng, tol=self.tol, max_iter=self.max_iter)
        return FitResult(U.argmax(0), np.sqrt(D2.min(0)), None,
                         {"objective": J, "iterations": it, "history": hist, "centers": V, "U": U})
