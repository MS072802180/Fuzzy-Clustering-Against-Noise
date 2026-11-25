"""Possibilistic Fuzzy C-Means (Pal, Pal, Keller & Bezdek 2005).

J = sum_ik (a u_ik^m + b t_ik^eta) d_ik^2 + sum_i gamma_i sum_k (1 - t_ik)^eta
Initialized from FCM; gamma_i = K * sum_k u_ik^m d_ik^2 / sum_k u_ik^m is computed once from that
FCM run and then held fixed (Krishnapuram & Keller 1996; Pal et al. 2005). Re-estimating gamma at
every iteration lets the scales drift and encourages coincident clusters.
"""
import numpy as np
from .base import FitResult, sqdist, fcm_memberships
from .fcm import fcm_best


def typicality(D2, gamma, b, eta):
    z = np.log(b * D2 / gamma[:, None]) / (eta - 1.0)
    return 1.0 / (1.0 + np.exp(np.clip(z, -700, 700)))


def pfcm_objective(U, T, D2, gamma, m, eta, a, b):
    return float(((a * U ** m + b * T ** eta) * D2).sum() + (gamma[:, None] * (1 - T) ** eta).sum())


class PFCM:
    name = "PFCM"
    paradigm, mechanism = "fuzzy", "M1 relaxed memberships"

    def __init__(self, m=2.0, eta=2.0, a=1.0, b=1.0, K=1.0, n_init=5, tol=1e-5, max_iter=300, seed=0):
        self.m, self.eta, self.a, self.b, self.K = m, eta, a, b, K
        self.n_init, self.tol, self.max_iter, self.seed = n_init, tol, max_iter, seed

    def fit(self, X, c):
        rng = np.random.default_rng(self.seed)
        V, U, D2, _, _, _ = fcm_best(X, c, self.m, self.n_init, rng)
        Um = U ** self.m
        gamma = self.K * (Um * D2).sum(1) / Um.sum(1)
        hist = []
        for it in range(self.max_iter):
            U = fcm_memberships(D2, self.m)
            T = typicality(D2, gamma, self.b, self.eta)
            hist.append(pfcm_objective(U, T, D2, gamma, self.m, self.eta, self.a, self.b))
            W = self.a * U ** self.m + self.b * T ** self.eta
            V_new = W @ X / W.sum(1, keepdims=True)
            shift = np.abs(V_new - V).max()
            V = V_new
            D2 = sqdist(X, V)
            if shift < self.tol:
                break
        U = fcm_memberships(D2, self.m)
        T = typicality(D2, gamma, self.b, self.eta)
        hist.append(pfcm_objective(U, T, D2, gamma, self.m, self.eta, self.a, self.b))
        return FitResult(U.argmax(0), 1.0 - T.max(0), None,
                         {"iterations": it + 1, "history": hist, "centers": V, "U": U, "T": T, "gamma": gamma})
