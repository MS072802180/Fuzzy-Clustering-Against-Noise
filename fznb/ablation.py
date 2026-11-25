"""Ablations: PFCM/FCM parameters with label-free selection, and the number of clusters c.

1. Parameter grid. For every dataset x noise type x seed (at the ablation level), PFCM is fitted on a
   grid of (m, eta, a, b) and FCM on the same m values. Three label-free criteria pick one PFCM
   setting per condition; we report the ARI each rule achieves against the oracle (best ARI in the
   grid, which needs labels) and the default (m=eta=2, a=b=1).
2. Number of clusters. Each method in `c_methods` is fitted with c = true c + each of `c_offsets`,
   on clean data and on each noise type in `c_noise`. We record ARI and the label-free indices each
   paradigm would use to choose c (FPC and Xie-Beni for fuzzy methods, BIC for GMM, silhouette for
   all) and report how often each index recovers the true c.

Usage:
    python -m fznb.ablation --config configs/full.yaml
"""
import argparse, os, warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score

from . import data, noise
from .methods.pfcm import PFCM
from .methods.fcm import FCM
from .methods import make
from .selection import criteria, xie_beni, _sil
from . import metrics
from .config import load_config


def _noisy(D, ds, nt, lv, seed):
    """Same noise realisation as fznb.run for this (dataset, noise, level, seed)."""
    if nt == "none" or lv <= 0:
        n = len(D.y)
        return D.X.copy(), np.zeros(n, bool), np.zeros(n, bool)
    nseed = int(np.random.default_rng([seed, int(round(lv * 1000)), sum(map(ord, ds)), sum(map(ord, nt))]).integers(2**31))
    return noise.inject(D.X, D.y, nt, lv, nseed)


def run_condition(ds, nt, lv, seed, grid, m_values=()):
    warnings.filterwarnings("ignore")
    D = data.load(ds)
    X, inj, con = _noisy(D, ds, nt, lv, seed)
    rows = []
    for m, eta, a, b in grid:
        fit = PFCM(m=m, eta=eta, a=a, b=b, seed=seed).fit(X, D.n_clusters)
        r = dict(method="PFCM", dataset=ds, noise=nt, level=lv, seed=seed, m=m, eta=eta, a=a, b=b,
                 ari=adjusted_rand_score(D.y, fit.labels[~inj]),
                 fpc=metrics.partition_coefficient(fit.info["U"]))
        r.update(criteria(X, fit, m, seed=seed))
        rows.append(r)
    for m in m_values:                       # FCM fuzzifier sweep (no eta, a, b)
        fit = FCM(m=m, seed=seed).fit(X, D.n_clusters)
        rows.append(dict(method="FCM", dataset=ds, noise=nt, level=lv, seed=seed, m=m, eta=np.nan, a=np.nan,
                         b=np.nan, ari=adjusted_rand_score(D.y, fit.labels[~inj]),
                         fpc=metrics.partition_coefficient(fit.info["U"]),
                         n_pred_clusters=len(np.unique(fit.labels[~inj]))))
    return rows


def run_c_condition(ds, nt, lv, seed, offsets, methods, params):
    warnings.filterwarnings("ignore")
    D = data.load(ds)
    X, inj, con = _noisy(D, ds, nt, lv, seed)
    rng = np.random.default_rng(seed)
    cs = sorted({D.n_clusters + o for o in offsets if D.n_clusters + o >= 2})
    rows = []
    for mn in methods:
        for c in cs:
            model = make(mn, seed=seed, **params.get(mn, {}))
            fit = model.fit(X, c)
            m = getattr(model, "m", None)            # fuzzifier, for Xie-Beni (fuzzy methods only)
            U, V = fit.info.get("U"), fit.info.get("centers")
            rows.append(dict(
                dataset=ds, noise=nt, level=lv, seed=seed, method=mn, c=c, true_c=D.n_clusters,
                ari=adjusted_rand_score(D.y, fit.labels[~inj]),
                fpc=metrics.partition_coefficient(U) if U is not None else np.nan,
                xie_beni=xie_beni(X, U, V, m) if (m is not None and U is not None and V is not None) else np.nan,
                silhouette=_sil(X, fit.labels, rng),
                bic=fit.info.get("bic", np.nan)))
    return rows


def summarize(df, out):
    if "method" in df:
        fcm_rows = df[df.method == "FCM"]
        if len(fcm_rows):
            fcm_rows.groupby(["dataset", "m"]).ari.mean().unstack("m").round(3).to_csv(
                os.path.join(out, "ablation_fcm_m.csv"))
        df = df[df.method == "PFCM"]
    keys = ["dataset", "noise", "level", "seed"]
    is_default = (df.m == 2.0) & (df.eta == 2.0) & (df.a == 1.0) & (df.b == 1.0)
    recs = []
    for k, g in df.groupby(keys):
        rec = dict(zip(keys, k))
        rec["oracle"] = g.ari.max()
        d = g[is_default.loc[g.index]]
        rec["default"] = d.ari.iloc[0] if len(d) else np.nan   # NaN if the grid omits m=eta=2, a=b=1
        for crit in ["neg_xie_beni", "silhouette", "trimmed_silhouette"]:
            rec[crit] = g.loc[g[crit].idxmax(), "ari"]
        rec["worst"] = g.ari.min()
        recs.append(rec)
    sel = pd.DataFrame(recs)
    sel.to_csv(os.path.join(out, "ablation_selection.csv"), index=False)
    cols = ["oracle", "default", "neg_xie_beni", "silhouette", "trimmed_silhouette", "worst"]
    tab = sel.groupby(["dataset"])[cols].mean().round(3)
    tab.loc["MEAN"] = sel[cols].mean().round(3)
    tab.to_csv(os.path.join(out, "ablation_summary.csv"))
    # sensitivity surface: mean ARI per (m, eta), averaged over everything else
    surf = df.groupby(["m", "eta"]).ari.mean().unstack().round(3)
    surf.to_csv(os.path.join(out, "ablation_m_eta_surface.csv"))
    print(tab.to_string())
    return sel, tab, surf


def summarize_c(df, out):
    """How often each label-free index picks the true c, and the ARI at the c it picks."""
    rules = {"FPC (max)": ("fpc", "max"), "Xie-Beni (min)": ("xie_beni", "min"),
             "Silhouette (max)": ("silhouette", "max"), "BIC (min)": ("bic", "min")}
    recs = []
    for (ds, nt, seed, mn), g in df.groupby(["dataset", "noise", "seed", "method"]):
        for rule, (col, how) in rules.items():
            v = g[col]
            if v.isna().all():
                continue
            i = v.idxmax() if how == "max" else v.idxmin()
            recs.append(dict(dataset=ds, noise=nt, seed=seed, method=mn, rule=rule, chosen_c=g.loc[i, "c"],
                             true_c=g.loc[i, "true_c"], correct=float(g.loc[i, "c"] == g.loc[i, "true_c"]),
                             ari_at_chosen=g.loc[i, "ari"], ari_at_true=g[g.c == g.true_c].ari.mean()))
    sel = pd.DataFrame(recs)
    sel.to_csv(os.path.join(out, "ablation_c_selection.csv"), index=False)
    tab = sel.groupby(["method", "rule", "noise"]).correct.mean().unstack("noise").round(3)
    tab.to_csv(os.path.join(out, "ablation_c_summary.csv"))
    print("\nfraction of runs in which the index recovers the true c:\n" + tab.to_string())
    return sel, tab


def run_c(cfg, jobs=None):
    acfg = cfg.get("ablation") or {}
    if not acfg.get("c_offsets"):
        return None
    out = cfg["out_dir"]
    methods = acfg.get("c_methods", ["FCM", "PFCM", "NC", "GMM", "GMM+noise"])
    noises = acfg.get("c_noise", ["none", "uniform"])
    conds = [(ds, nt, 0.0 if nt == "none" else float(acfg["level"]), s) for ds in cfg["datasets"]
             for nt in noises for s in range(acfg.get("c_seeds", acfg["seeds"]))]
    print(f"c-ablation: {len(conds)} conditions x {len(methods)} methods x {len(acfg['c_offsets'])} values of c")
    res = Parallel(n_jobs=jobs or cfg.get("jobs", -1), verbose=5)(
        delayed(run_c_condition)(*cd, acfg["c_offsets"], methods, cfg.get("method_params", {})) for cd in conds)
    df = pd.DataFrame([r for rs in res for r in rs])
    df.to_csv(os.path.join(out, "ablation_c_raw.csv"), index=False)
    return summarize_c(df, out)


def run_components(cfg, jobs=None):
    """Component ablation of the proposed model: NC -> RNC (noise distance) / ST-NC (fuzzifier
    selection) -> ST-RNC (both). Uses exactly the main study's conditions and noise realisations;
    only the variants not already in the main run are fitted here."""
    acfg = cfg.get("ablation") or {}
    comps = acfg.get("components")
    if not comps:
        return None
    from .run import conditions, run_one
    out = cfg["out_dir"]
    extra = [m for m in comps if m not in cfg["methods"]]
    params = cfg.get("method_params", {}) or {}
    conds = list(conditions(cfg))
    print(f"component ablation: {len(conds)} conditions x {extra}")
    res = Parallel(n_jobs=jobs or cfg.get("jobs", -1), verbose=2)(
        delayed(run_one)(ds, nt, lv, seed, extra, params) for ds, nt, lv, seed in conds)
    df = pd.DataFrame([r for rs in res for r in rs])
    df.to_csv(os.path.join(out, "ablation_components_raw.csv"), index=False)
    return summarize_components(cfg)


def summarize_components(cfg):
    out = cfg["out_dir"]
    comps = (cfg.get("ablation") or {}).get("components", [])
    p_extra, p_main = os.path.join(out, "ablation_components_raw.csv"), os.path.join(out, "raw_results.csv")
    if not comps or not os.path.exists(p_extra) or not os.path.exists(p_main):
        return None
    main = pd.read_csv(p_main)
    df = pd.concat([main[main.method.isin(comps)], pd.read_csv(p_extra)], ignore_index=True)
    df = df[df.error.fillna("") == ""]
    clean = df[df.noise == "none"].groupby("method").ari.mean()
    noisy = df[df.noise != "none"]
    hi = noisy.level.max()
    tab = noisy[noisy.level == hi].groupby(["method", "noise"]).ari.mean().unstack("noise")
    tab.insert(0, "clean", clean)
    tab["all noisy"] = noisy.groupby("method").ari.mean()
    tab = tab.loc[[m for m in comps if m in tab.index]].round(3)
    tab.to_csv(os.path.join(out, "ablation_components_summary.csv"))
    print("\ncomponent ablation (mean ARI; noise columns at r = %.2f):\n" % hi + tab.to_string())
    return tab


def run(cfg, jobs=None):
    acfg = cfg.get("ablation")
    if not acfg:
        print("no 'ablation' section in the config; skipping")
        return None
    out = cfg["out_dir"]; os.makedirs(out, exist_ok=True)
    grid = [(m, e, a, b) for m in acfg["m"] for e in acfg["eta"] for a, b in acfg["ab"]]
    conds = [(ds, nt, float(acfg["level"]), s) for ds in cfg["datasets"] for nt in cfg["noise_types"]
             for s in range(acfg["seeds"])]
    print(f"ablation: {len(conds)} conditions x {len(grid)} settings")
    res = Parallel(n_jobs=jobs or cfg.get("jobs", -1), verbose=5)(
        delayed(run_condition)(*c, grid, acfg.get("m", [])) for c in conds)
    df = pd.DataFrame([r for rs in res for r in rs])
    df.to_csv(os.path.join(out, "ablation_raw.csv"), index=False)
    result = summarize(df, out)
    run_c(cfg, jobs)
    run_components(cfg, jobs)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/quick.yaml")
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args(argv)
    run(load_config(args.config), args.jobs)


if __name__ == "__main__":
    main()
