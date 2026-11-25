"""Probabilistic paradigm: plain GMM, GMM + uniform noise component, and a mixture of t distributions."""
import numpy as np
from scipy.special import logsumexp, gammaln, digamma
from scipy.optimize import brentq
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from .base import FitResult

REG = 1e-3  # added to covariance diagonals (data are standardized)


def _gauss_logpdf(X, mu, S):
    d = X.shape[1]
    L = np.linalg.cholesky(S)
    Z = np.linalg.solve(L, (X - mu).T)
    maha = (Z ** 2).sum(0)
    logdet = 2.0 * np.log(np.diag(L)).sum()
    return -0.5 * (d * np.log(2 * np.pi) + logdet + maha), maha


class GMM:
    name = "GMM"
    paradigm, mechanism = "probabilistic", "baseline"

    def __init__(self, n_init=5, max_iter=500, seed=0):
        self.n_init, self.max_iter, self.seed = n_init, max_iter, seed

    def fit(self, X, c):
        g = GaussianMixture(c, covariance_type="full", reg_covar=REG, n_init=self.n_init,
                            max_iter=self.max_iter, random_state=self.seed).fit(X)
        return FitResult(g.predict(X), -g.score_samples(X), None,
                         {"converged": bool(g.converged_), "iterations": int(g.n_iter_),
                          "U": g.predict_proba(X).T, "centers": g.means_, "bic": float(g.bic(X))})


class GMMNoise:
    """Gaussian mixture plus a uniform noise component over the data's bounding box
    (Banfield & Raftery 1993; Fraley & Raftery 2002). Fitted by EM, initialized from a GMM."""
    name = "GMM+noise"
    paradigm, mechanism = "probabilistic", "M3 explicit outlier handling"

    def __init__(self, init_noise=0.05, n_init=5, tol=1e-6, max_iter=500, seed=0):
        self.init_noise, self.n_init, self.tol, self.max_iter, self.seed = init_noise, n_init, tol, max_iter, seed

    def fit(self, X, c):
        n, d = X.shape
        g = GaussianMixture(c, covariance_type="full", reg_covar=REG, n_init=self.n_init,
                            random_state=self.seed).fit(X)
        mu, S = g.means_.copy(), g.covariances_.copy()
        pi = np.r_[g.weights_ * (1 - self.init_noise), self.init_noise]
        logV = np.log(np.maximum(X.max(0) - X.min(0), 1e-6)).sum()
        prev = -np.inf
        hist = []
        for it in range(self.max_iter):
            L = np.empty((c + 1, n))
            for k in range(c):
                L[k] = np.log(max(pi[k], 1e-300)) + _gauss_logpdf(X, mu[k], S[k])[0]
            L[c] = np.log(max(pi[c], 1e-300)) - logV
            ll = logsumexp(L, axis=0)
            R = np.exp(L - ll)
            total = ll.sum()
            hist.append(float(total))
            pi = R.mean(1)
            for k in range(c):
                w = R[k]
                sw = max(w.sum(), 1e-10)
                mu[k] = w @ X / sw
                Xc = X - mu[k]
                S[k] = (w[:, None] * Xc).T @ Xc / sw + REG * np.eye(d)
            if abs(total - prev) < self.tol * abs(total):
                break
            prev = total
        post_noise = R[c]
        labels = R[:c].argmax(0)
        return FitResult(labels, post_noise, post_noise > 0.5,
                         {"iterations": it + 1, "noise_proportion": float(pi[c]), "history": hist,
                          "U": R[:c], "u_noise": post_noise, "centers": mu})


class TMixture:
    """Mixture of multivariate t distributions with per-component degrees of freedom,
    fitted by EM (Peel & McLachlan 2000)."""
    name = "t-mixture"
    paradigm, mechanism = "probabilistic", "M2 heavy-tailed likelihood"

    def __init__(self, nu0=10.0, estimate_nu=True, n_init=3, tol=1e-6, max_iter=300, seed=0):
        self.nu0, self.estimate_nu, self.n_init = nu0, estimate_nu, n_init
        self.tol, self.max_iter, self.seed = tol, max_iter, seed

    @staticmethod
    def _t_logpdf(X, mu, S, nu):
        d = X.shape[1]
        L = np.linalg.cholesky(S)
        Z = np.linalg.solve(L, (X - mu).T)
        maha = (Z ** 2).sum(0)
        logdet = 2.0 * np.log(np.diag(L)).sum()
        lp = (gammaln((nu + d) / 2) - gammaln(nu / 2) - 0.5 * d * np.log(nu * np.pi)
              - 0.5 * logdet - 0.5 * (nu + d) * np.log1p(maha / nu))
        return lp, maha

    def _fit_once(self, X, c, seed):
        n, d = X.shape
        lab = KMeans(c, n_init=1, random_state=seed).fit_predict(X)
        R = np.eye(c)[lab].T.astype(float)
        mu = np.array([X[lab == k].mean(0) if np.any(lab == k) else X[seed % n] for k in range(c)])
        S = np.array([np.cov(X[lab == k], rowvar=False) + REG * np.eye(d) if np.sum(lab == k) > d
                      else np.eye(d) for k in range(c)])
        pi = R.mean(1)
        nu = np.full(c, self.nu0)
        prev = -np.inf
        hist = []
        for it in range(self.max_iter):
            L = np.empty((c, n)); maha = np.empty((c, n))
            for k in range(c):
                lp, maha[k] = self._t_logpdf(X, mu[k], S[k], nu[k])
                L[k] = np.log(max(pi[k], 1e-300)) + lp
            ll = logsumexp(L, axis=0)
            R = np.exp(L - ll)
            total = ll.sum()
            hist.append(float(total))
            U = (nu[:, None] + d) / (nu[:, None] + maha)
            pi = R.mean(1)
            for k in range(c):
                w = R[k] * U[k]
                mu[k] = w @ X / max(w.sum(), 1e-10)
                Xc = X - mu[k]
                S[k] = (w[:, None] * Xc).T @ Xc / max(R[k].sum(), 1e-10) + REG * np.eye(d)
                if self.estimate_nu:
                    nk = max(R[k].sum(), 1e-10)
                    const = 1 + (R[k] * (np.log(U[k]) - U[k])).sum() / nk \
                        + digamma((nu[k] + d) / 2) - np.log((nu[k] + d) / 2)
                    f = lambda v: -digamma(v / 2) + np.log(v / 2) + const
                    try:
                        nu[k] = brentq(f, 0.5, 200.0)
                    except ValueError:
                        nu[k] = 200.0 if f(200.0) > 0 else 0.5
            if abs(total - prev) < self.tol * abs(total):
                break
            prev = total
        return total, R, ll, nu, it + 1, hist, mu

    def fit(self, X, c):
        best = None
        for i in range(self.n_init):
            out = self._fit_once(X, c, self.seed * 1000 + i)
            if best is None or out[0] > best[0]:
                best = out
        total, R, ll, nu, it, hist, mu = best
        return FitResult(R.argmax(0), -ll, None, {"loglik": float(total), "nu": nu.tolist(), "iterations": it,
                                                  "history": hist, "U": R, "centers": mu})
