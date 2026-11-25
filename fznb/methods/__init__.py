"""Method registry. Each entry maps a short name to a class with .fit(X, c) -> FitResult."""
from .fcm import FCM
from .pfcm import PFCM
from .noise_clustering import NoiseClustering
from .gmm import GMM, GMMNoise, TMixture
from .ae_fcm import AEFCM
from .strnc import SelfTuningRNC, RobustNoiseClustering, SelfTuningNC

REGISTRY = {
    "FCM": FCM,
    "PFCM": PFCM,
    "NC": NoiseClustering,
    "GMM": GMM,
    "GMM+noise": GMMNoise,
    "t-mixture": TMixture,
    "AE+FCM": AEFCM,
    "ST-RNC": SelfTuningRNC,          # proposed model
    "RNC": RobustNoiseClustering,     # ablation: chi-square noise distance only
    "ST-NC": SelfTuningNC,            # ablation: fuzzifier selection only
}

# Mechanism pairs: methods in different paradigms that use the same robustness idea.
MECHANISM_PAIRS = [
    ("NC", "GMM+noise", "noise cluster vs noise component"),
    ("PFCM", "t-mixture", "down-weighting by typicality vs heavy-tailed likelihood"),
    ("FCM", "GMM", "unconstrained baselines"),
]


def make(name, seed=0, **params):
    return REGISTRY[name](seed=seed, **params)
