"""Datasets used by the benchmark.

Every dataset is returned standardized (z-scores) with constant features removed, so that
noise levels and distances mean the same thing across datasets.

* Main study (configs/full.yaml): seven real datasets with 2,000-10,992 samples (six from UCI and
  Texture from the ELENA project), taken from PMLB (Olson et al., BioData Mining 2017). The PMLB files are bundled
  in data/pmlb/ and verified by SHA-256 on every load, so no download is needed.
* Extended (configs/extended.yaml): adds Letter (20,000 samples) and a 10,000-sample MNIST subset.
  MNIST (15 MB) is downloaded on first use into data/pmlb/ and checksum-verified.
* Small (configs/small.yaml): the earlier scikit-learn-bundled UCI sets (Iris, Wine, WDBC, Digits).

A failed download or checksum raises an error with instructions; nothing is silently replaced.
"""
from dataclasses import dataclass
import gzip
import hashlib
import os
import numpy as np
from sklearn import datasets as skd
from sklearn.preprocessing import StandardScaler


@dataclass
class Dataset:
    name: str
    X: np.ndarray        # (n, d) standardized features
    y: np.ndarray        # (n,) integer ground-truth labels, 0..c-1
    n_clusters: int
    source: str


def _finish(name, X, y, source):
    X = np.asarray(X, dtype=float)
    keep = X.std(axis=0) > 1e-12          # drop constant columns (e.g. blank pixels in digits)
    X = StandardScaler().fit_transform(X[:, keep])
    _, y = np.unique(np.asarray(y), return_inverse=True)
    return Dataset(name, X, y.astype(int), int(y.max() + 1), source)


def synthetic_2d():
    """S1: three 2-D Gaussian clusters of unequal size (200/100/50) and spread."""
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 3.5]])
    sizes, sds = [200, 100, 50], [0.7, 0.5, 0.4]
    X = np.vstack([rng.normal(c, s, size=(n, 2)) for c, n, s in zip(centers, sizes, sds)])
    y = np.repeat(np.arange(3), sizes)
    return _finish("S1_2d_unequal", X, y, "synthetic (seed 0)")


def synthetic_2d_large():
    """S1-L: the S1 design scaled up tenfold (2,000/1,000/500 points)."""
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 3.5]])
    sizes, sds = [2000, 1000, 500], [0.7, 0.5, 0.4]
    X = np.vstack([rng.normal(c, s, size=(n, 2)) for c, n, s in zip(centers, sizes, sds)])
    y = np.repeat(np.arange(3), sizes)
    return _finish("S1_large", X, y, "synthetic (seed 0)")


def synthetic_10d_large():
    """S2-L: five 10-D Gaussian clusters, 1,000 points each."""
    X, y = skd.make_blobs(n_samples=5000, n_features=10, centers=5, cluster_std=1.0,
                          center_box=(-5, 5), random_state=0)
    return _finish("S2_large", X, y, "synthetic, sklearn make_blobs (seed 0)")


def synthetic_10d():
    """S2: five 10-D Gaussian clusters, 100 points each."""
    X, y = skd.make_blobs(n_samples=500, n_features=10, centers=5, cluster_std=1.0,
                          center_box=(-5, 5), random_state=0)
    return _finish("S2_10d_blobs", X, y, "synthetic, sklearn make_blobs (seed 0)")


def iris():
    d = skd.load_iris()
    return _finish("iris", d.data, d.target, "UCI Iris (sklearn bundled)")


def wine():
    d = skd.load_wine()
    return _finish("wine", d.data, d.target, "UCI Wine (sklearn bundled)")


def breast_cancer():
    d = skd.load_breast_cancer()
    return _finish("breast_cancer", d.data, d.target, "UCI Breast Cancer Wisconsin Diagnostic (sklearn bundled)")


def digits():
    d = skd.load_digits()
    return _finish("digits", d.data, d.target, "UCI Optical Recognition of Handwritten Digits (sklearn bundled)")


# ------------------------------------------------------------------ PMLB / UCI datasets
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pmlb")
PMLB_URL = "https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/{0}/{0}.tsv.gz"
# name in PMLB, SHA-256 of the .tsv.gz (from PMLB's Git LFS pointers), original source
PMLB = {
    "waveform":     ("waveform_21", "7c2cc6b218c3c4868ccc2edcf6a637c796f4e6bf4ad9f0994e95a260bfe3aabc",
                     "UCI Waveform Database Generator (Version 1), 5,000 x 21, 3 classes"),
    "segmentation": ("segmentation", "980ba4342d2fe489793cc3431678318f8d1467fe6bf6a687eb18bfc2267a049f",
                     "UCI Image Segmentation, 2,310 x 19, 7 classes"),
    "satimage":     ("satimage", "1ee8cf9693351db7afe68f6fc32942845caae3e1030c688efa6c5d0b24229f46",
                     "UCI Statlog (Landsat Satellite), 6,435 x 36, 6 classes"),
    "optdigits":    ("optdigits", "b70b7ac68a54d1ccd41bd24f11324c3af96b67541791fb9d8684cbdf151533c3",
                     "UCI Optical Recognition of Handwritten Digits (full), 5,620 x 64, 10 classes"),
    "pendigits":    ("pendigits", "73294fc30c2c50bf737c954d04cca3a1be69508535d3d52e2512ec88fefaad3f",
                     "UCI Pen-Based Recognition of Handwritten Digits, 10,992 x 16, 10 classes"),
    "texture":      ("texture", "94abf897fcd6d93b0f78ea309b10ef5b83a8b95e46aa29974a43dbefb3741c0c",
                     "Texture (ELENA project, via KEEL), 5,500 x 40, 11 classes"),
    "mfeat_factors": ("mfeat_factors", "f5c6631a9716b8c50951dbf0eb66c2f00d299c115fee72a0c23aff5a7f59fd57",
                      "UCI Multiple Features, profile correlations view, 2,000 x 216, 10 classes"),
    "letter":       ("letter", "f8ffe4f9920dc63fb85b864cc1afffb66e45c6e302d175905fd7db04ba67b8c6",
                     "UCI Letter Recognition, 20,000 x 16, 26 classes"),
    "mnist":        ("mnist", "5b1d8788d9512819fd46d1acf1e86e70cb5f4418c8e8bfe299ebd3abf2188217",
                     "MNIST (LeCun et al. 1998), 70,000 x 784, 10 classes"),
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pmlb_path(key):
    """Local path of a PMLB file, downloading and verifying it if it is missing."""
    name, sha, _ = PMLB[key]
    path = os.path.join(DATA_DIR, f"{name}.tsv.gz")
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        url = PMLB_URL.format(name)
        print(f"downloading {key} from {url} ...", flush=True)
        import urllib.request
        tmp = path + ".part"
        try:
            urllib.request.urlretrieve(url, tmp)
        except Exception as e:
            raise RuntimeError(f"could not download {key} ({e}).\nDownload it manually from\n  {url}\n"
                               f"and save it as\n  {path}") from e
        os.replace(tmp, path)
    got = _sha256(path)
    if got != sha:
        raise RuntimeError(f"{path} failed its checksum (expected {sha[:12]}..., got {got[:12]}...). "
                           "Delete the file and run again to re-download it.")
    return path


def _read_pmlb(key):
    import pandas as pd
    with gzip.open(pmlb_path(key), "rt") as f:
        df = pd.read_csv(f, sep="\t")
    y = df.pop("target").to_numpy()
    return df.to_numpy(float), y


def _pmlb_loader(key):
    def load_fn():
        X, y = _read_pmlb(key)
        return _finish(key, X, y, PMLB[key][2] + " (via PMLB)")
    load_fn.__doc__ = PMLB[key][2]
    return load_fn


def mnist10k():
    """10,000 MNIST images (random subset, seed 0), reduced to 50 principal components.
    Full-covariance mixtures in 784 dimensions are not feasible for thousands of fits, so the
    PCA reduction (fitted on the clean subset, before any noise) is applied to every method."""
    from sklearn.decomposition import PCA
    X, y = _read_pmlb("mnist")
    idx = np.random.default_rng(0).choice(len(X), 10000, replace=False)
    X, y = X[idx], y[idx]
    X = PCA(50, random_state=0).fit_transform(X)
    return _finish("mnist10k", X, y, "MNIST, 10,000-image subset, PCA to 50 components (via PMLB)")


REGISTRY = {
    "S1_2d_unequal": synthetic_2d,
    "S2_10d_blobs": synthetic_10d,
    "iris": iris,
    "wine": wine,
    "breast_cancer": breast_cancer,
    "digits": digits,
    "S1_large": synthetic_2d_large,
    "S2_large": synthetic_10d_large,
    **{k: _pmlb_loader(k) for k in ["waveform", "segmentation", "satimage", "optdigits", "pendigits", "texture",
                                         "mfeat_factors", "letter"]},
    "mnist10k": mnist10k,
}


def from_csv(path, label_col=-1):
    """Any CSV with a header row: numeric feature columns plus one label column (default: last).
    Use it in a config as  datasets: [csv:data/my_file.csv]"""
    import pandas as pd
    df = pd.read_csv(path)
    lab = df.columns[label_col]
    X = df.drop(columns=[lab]).select_dtypes("number").to_numpy()
    import os
    return _finish(os.path.splitext(os.path.basename(path))[0], X, df[lab].to_numpy(), f"CSV file {path}")


def load(name):
    if name.startswith("csv:"):
        return from_csv(name[4:])
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[name]()
