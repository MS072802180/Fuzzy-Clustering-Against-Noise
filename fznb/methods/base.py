"""Common interface: every method returns a FitResult."""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from sklearn.cluster import kmeans_plusplus


@dataclass
class FitResult:
    labels: np.ndarray                    # crisp cluster label per sample (0..c-1)
    outlier_score: np.ndarray             # higher = more outlying (used for ROC-AUC)
    outlier_flag: Optional[np.ndarray]    # native outlier decision, or None if the method has none
    info: dict = field(default_factory=dict)


def sqdist(X, V):
    """Squared Euclidean distances, shape (c, n), floored to avoid division by zero."""
    D2 = (V ** 2).sum(1)[:, None] - 2.0 * V @ X.T + (X ** 2).sum(1)[None, :]
    return np.maximum(D2, 1e-12)


def init_centers(X, c, rng):
    V, _ = kmeans_plusplus(X, c, random_state=int(rng.integers(2**31 - 1)))
    return V.astype(float)


def fcm_memberships(D2, m):
    """FCM membership update, computed in log space for numerical stability."""
    logw = -np.log(D2) / (m - 1.0)
    logw -= logw.max(0, keepdims=True)
    W = np.exp(logw)
    return W / W.sum(0, keepdims=True)
