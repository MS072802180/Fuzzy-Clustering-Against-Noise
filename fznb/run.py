"""Main experiment: every method x dataset x noise type x contamination level x seed.

Usage:
    python -m fznb.run --config configs/full.yaml
    python -m fznb.run --config configs/quick.yaml --jobs 4

Writes one row per fit to <out_dir>/raw_results.csv. Re-running resumes: fits already in the
CSV are skipped, so an interrupted run can be continued with the same command.
"""
import argparse, os, time, json, traceback, warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import data, noise, metrics
from .methods import make, REGISTRY
from .config import load_config

KEYS = ["dataset", "noise", "level", "seed", "method"]


def conditions(cfg):
    """Level 0 is the same clean data for every noise type, so it is run once (noise='none')."""
    for ds in cfg["datasets"]:
        for seed in range(cfg["seeds"]):
            yield ds, "none", 0.0, seed
            for nt in cfg["noise_types"]:
                for lv in cfg["levels"]:
                    if lv > 0:
                        yield ds, nt, float(lv), seed


def run_one(ds, nt, lv, seed, method_names, method_params):
    warnings.filterwarnings("ignore")
    D = data.load(ds)
    # noise seed depends on everything except the method, so all methods see identical data
    nseed = int(np.random.default_rng([seed, int(round(lv * 1000)), sum(map(ord, ds)), sum(map(ord, nt))]).integers(2**31))
    X, injected, contaminated = noise.inject(D.X, D.y, nt if nt != "none" else "uniform", lv, nseed)
    rows = []
    for mn in method_names:
        row = dict(dataset=ds, noise=nt, level=lv, seed=seed, method=mn,
                   paradigm=REGISTRY[mn].paradigm, mechanism=REGISTRY[mn].mechanism,
                   n=len(X), d=X.shape[1], c=D.n_clusters)
        try:
            model = make(mn, seed=seed, **method_params.get(mn, {}))
            t0 = time.perf_counter()
            fit = model.fit(X, D.n_clusters)
            row["runtime_s"] = time.perf_counter() - t0
            row.update(metrics.evaluate(D.y, fit.labels, fit.outlier_score, fit.outlier_flag,
                                        injected, contaminated))
            row["fpc"] = metrics.partition_coefficient(fit.info["U"]) if "U" in fit.info else np.nan
            row["iterations"] = fit.info.get("iterations", np.nan)
            row["error"] = ""
        except Exception as e:  # keep going; record the failure
            row["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        rows.append(row)
    return rows


def run(cfg, jobs=None, fresh=False):
    out = cfg["out_dir"]; os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "raw_results.csv")
    if fresh and os.path.exists(path):
        os.remove(path)
    methods = cfg["methods"]
    params = cfg.get("method_params", {}) or {}
    done = set()
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[old["error"].fillna("") == ""]
        done = set(map(tuple, old[KEYS].astype(str).values))
        if done:
            print(f"resuming: {len(done)} fits already in {path} (use --fresh to start over)")
    todo = []
    for ds, nt, lv, seed in conditions(cfg):
        ms = [m for m in methods if (ds, nt, str(lv), str(seed), m) not in done]
        if ms:
            todo.append((ds, nt, lv, seed, ms))
    jobs = jobs or cfg.get("jobs", -1)
    print(f"{len(todo)} data conditions to run ({sum(len(t[4]) for t in todo)} fits), jobs={jobs}")
    with open(os.path.join(out, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    if not todo:
        print("nothing to do ->", path)
        return path
    batch = max(1, len(todo) // 20)
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        res = Parallel(n_jobs=jobs)(delayed(run_one)(*t, params) for t in chunk)
        rows = [r for rs in res for r in rs]
        pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        print(f"  {min(i + batch, len(todo))}/{len(todo)} conditions written", flush=True)
    print("done ->", path)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/quick.yaml")
    ap.add_argument("--jobs", type=int, default=None, help="parallel workers (default: config or all cores)")
    ap.add_argument("--fresh", action="store_true", help="delete existing raw_results.csv first")
    args = ap.parse_args(argv)
    run(load_config(args.config), args.jobs, args.fresh)


if __name__ == "__main__":
    main()
