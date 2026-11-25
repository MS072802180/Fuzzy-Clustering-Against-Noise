import os
import numpy as np
import pytest
from fznb import data, noise, metrics
from fznb.methods import make, REGISTRY
from fznb.methods.base import fcm_memberships, sqdist


@pytest.fixture(scope="module")
def s1():
    return data.load("S1_2d_unequal")


def test_datasets_standardized():
    for name in data.REGISTRY:
        D = data.load(name)
        assert np.allclose(D.X.mean(0), 0, atol=1e-8) and np.allclose(D.X.std(0), 1, atol=1e-8)
        assert D.y.min() == 0 and D.y.max() == D.n_clusters - 1


@pytest.mark.parametrize("nt", noise.NOISE_TYPES)
def test_noise_levels(s1, nt):
    X, inj, con = noise.inject(s1.X, s1.y, nt, 0.2, seed=3)
    if nt in ("cellwise", "rowwise"):
        assert X.shape == s1.X.shape and inj.sum() == 0 and con.sum() == round(0.2 * len(s1.X))
    else:
        assert np.isclose(inj.mean(), 0.2, atol=0.01) and (inj == con).all()
    X2, *_ = noise.inject(s1.X, s1.y, nt, 0.2, seed=3)
    assert np.array_equal(X, X2)  # deterministic


def test_memberships_sum_to_one(s1):
    V = s1.X[:3]
    U = fcm_memberships(sqdist(s1.X, V), 2.0)
    assert np.allclose(U.sum(0), 1)


def test_accuracy_permutation_invariant():
    y = np.array([0, 0, 1, 1, 2, 2])
    assert metrics.accuracy(y, np.array([2, 2, 0, 0, 1, 1])) == 1.0


@pytest.mark.parametrize("name", [m for m in REGISTRY if m != "AE+FCM"])
def test_methods_recover_clean_clusters(s1, name):
    fit = make(name, seed=0).fit(s1.X, s1.n_clusters)
    res = metrics.evaluate(s1.y, fit.labels, fit.outlier_score, fit.outlier_flag,
                           np.zeros(len(s1.y), bool), np.zeros(len(s1.y), bool))
    assert res["ari"] > 0.9


@pytest.mark.parametrize("name", ["NC", "GMM+noise"])
def test_noise_models_flag_uniform_outliers(s1, name):
    X, inj, con = noise.inject(s1.X, s1.y, "uniform", 0.1, seed=1)
    fit = make(name, seed=0).fit(X, s1.n_clusters)
    res = metrics.evaluate(s1.y, fit.labels, fit.outlier_score, fit.outlier_flag, inj, con)
    assert res["outlier_auc"] > 0.9 and res["outlier_f1"] > 0.5


def test_ae_fcm_runs(s1):
    pytest.importorskip("torch")
    fit = make("AE+FCM", seed=0, epochs=30).fit(s1.X, s1.n_clusters)
    assert len(fit.labels) == len(s1.X)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_csv_loader():
    D = data.load("csv:" + os.path.join(ROOT, "data", "example_iris.csv"))
    assert D.X.shape == (150, 4) and D.n_clusters == 3


@pytest.mark.parametrize("name", ["waveform", "segmentation", "satimage", "optdigits", "pendigits", "texture",
                                  "mfeat_factors", "letter"])
def test_bundled_uci_datasets_load_offline(name):
    """The main-study datasets are bundled in data/pmlb and pass their SHA-256 check."""
    D = data.load(name)
    expected = {"waveform": (5000, 3), "segmentation": (2310, 7), "satimage": (6435, 6),
                "optdigits": (5620, 10), "pendigits": (10992, 10), "texture": (5500, 11),
                "mfeat_factors": (2000, 10), "letter": (20000, 26)}[name]
    assert (len(D.y), D.n_clusters) == expected
    assert np.isfinite(D.X).all()


def test_corrupted_file_is_rejected(tmp_path, monkeypatch):
    import shutil
    src = data.pmlb_path("segmentation")             # the bundled copy
    monkeypatch.setattr(data, "DATA_DIR", str(tmp_path))
    shutil.copy(src, tmp_path / "segmentation.tsv.gz")
    with open(tmp_path / "segmentation.tsv.gz", "ab") as f:
        f.write(b"tampered")
    with pytest.raises(RuntimeError, match="checksum"):
        data.load("segmentation")


def test_strnc_noise_distance_is_dimension_aware():
    from fznb.methods.strnc import chi2_delta2
    D2 = np.ones((3, 100))                         # every sample at squared distance 1 from its centre
    assert chi2_delta2(D2, 2) > chi2_delta2(D2, 50) > 1.0   # wider margin in low dimension


@pytest.mark.parametrize("name", ["ST-RNC", "RNC", "ST-NC"])
def test_proposed_model_and_ablations(s1, name):
    X, inj, con = noise.inject(s1.X, s1.y, "uniform", 0.1, seed=1)
    fit = make(name, seed=0).fit(X, s1.n_clusters)
    res = metrics.evaluate(s1.y, fit.labels, fit.outlier_score, fit.outlier_flag, inj, con)
    assert res["ari"] > 0.9 and res["outlier_auc"] > 0.9
    if name.startswith("ST"):
        assert fit.info["m"] in (1.2, 1.5, 2.0) and len(fit.info["m_scores"]) == 3
