"""Evaluation metrics: chance-corrected agreement, outlier detection, cost."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score,
                             normalized_mutual_info_score, roc_auc_score, f1_score)


def accuracy(y_true, y_pred):
    """Best-match accuracy (Hungarian assignment of predicted to true clusters)."""
    t, p = np.unique(y_true), np.unique(y_pred)
    M = np.zeros((len(p), len(t)))
    for i, a in enumerate(p):
        for j, b in enumerate(t):
            M[i, j] = np.sum((y_pred == a) & (y_true == b))
    r, c = linear_sum_assignment(-M)
    return M[r, c].sum() / len(y_true)


def partition_coefficient(U):
    """Bezdek's partition coefficient: mean over samples of sum_i u_ik^2 (U has shape c x n).
    1 = crisp partition, 1/c = maximally fuzzy. For NC and GMM+noise only the real clusters count,
    so points absorbed by the noise cluster lower the value."""
    return float((np.asarray(U) ** 2).sum(0).mean())


def evaluate(y_true, labels, outlier_score, outlier_flag, injected, contaminated):
    """All metrics for one fit.

    Clustering metrics use only the original samples (injected noise points are excluded),
    so they measure how much the noise distorted the clustering of the real data.
    """
    keep = ~injected
    yt, yp = y_true, labels[keep]
    res = {
        "ari": adjusted_rand_score(yt, yp),
        "ami": adjusted_mutual_info_score(yt, yp),
        "nmi": normalized_mutual_info_score(yt, yp),
        "acc": accuracy(yt, yp),
        "n_pred_clusters": int(len(np.unique(yp))),
    }
    if contaminated.any() and (~contaminated).any():
        s = np.nan_to_num(np.asarray(outlier_score, dtype=np.float64), nan=0.0, posinf=1e300, neginf=-1e300)
        res["outlier_auc"] = roc_auc_score(contaminated, s)
        res["outlier_f1"] = (f1_score(contaminated, outlier_flag) if outlier_flag is not None
                             else np.nan)
    else:
        res["outlier_auc"] = np.nan
        res["outlier_f1"] = np.nan
    return res
