"""Config loading shared by every entry point.

* Relative paths in a config (out_dir, csv: datasets) are resolved against the project root,
  so commands work from any working directory (VS Code, terminal, Colab).
* If PyTorch is not installed, AE+FCM is dropped with a warning instead of failing every fit.
"""
import os
import importlib.util
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIRED_KEYS = ["out_dir", "seeds", "datasets", "noise_types", "levels", "methods"]


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def torch_available():
    return importlib.util.find_spec("torch") is not None


def load_config(path, quiet=False):
    path = path if os.path.exists(path) else resolve(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    from . import data, noise
    from .methods import REGISTRY
    bad = [m for m in cfg["methods"] if m not in REGISTRY]
    if bad:
        raise ValueError(f"unknown method(s) {bad}; choose from {sorted(REGISTRY)}")
    bad = [n for n in cfg["noise_types"] if n not in noise.NOISE_TYPES]
    if bad:
        raise ValueError(f"unknown noise type(s) {bad}; choose from {list(noise.NOISE_TYPES)}")
    for ds in cfg["datasets"]:
        if not ds.startswith("csv:") and ds not in data.REGISTRY:
            raise ValueError(f"unknown dataset {ds!r}; choose from {sorted(data.REGISTRY)} or use csv:<path>")
    if any(not (0 <= float(l) < 1) for l in cfg["levels"]):
        raise ValueError("contamination levels must be in [0, 1)")
    cfg["datasets"] = [("csv:" + resolve(d[4:])) if d.startswith("csv:") else d for d in cfg["datasets"]]
    cfg["out_dir"] = resolve(cfg["out_dir"])
    cfg.setdefault("method_params", {})
    cfg["method_params"] = cfg["method_params"] or {}
    if "AE+FCM" in cfg["methods"] and not torch_available():
        cfg["methods"] = [m for m in cfg["methods"] if m != "AE+FCM"]
        if not quiet:
            print("WARNING: PyTorch is not installed, so AE+FCM is skipped. "
                  "Install it with:  pip install torch   (see README, section 1)")
    cfg["_config_path"] = path
    return cfg
