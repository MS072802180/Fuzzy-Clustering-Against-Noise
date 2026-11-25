"""Statistics, figures and tables from raw_results.csv (and ablation outputs, if present).

Usage:
    python -m fznb.analyze --config configs/full.yaml

Outputs (in out_dir):
    summary.csv                 mean/sd of every metric per method x dataset x noise x level
    ranks.csv                   average rank of each method on ARI (lower is better)
    stats_friedman.txt          Friedman test + Nemenyi critical difference
    stats_pairwise.csv          Wilcoxon signed-rank tests (Holm-corrected), incl. mechanism pairs
    runtime_median_s.csv        median fit time per dataset and method
    REPORT.md                   the key numbers in prose
    figures/, tables/, all_figures_and_tables.pdf   -> written by fznb.report (see that module)
"""
import argparse, os, itertools
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, rankdata

from .methods import MECHANISM_PAIRS
from .config import load_config

# Nemenyi q_{0.05} for k = 2..10 methods (Demsar 2006, Table 5)
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}



def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i]); adj[i] = min(1.0, running)
    return adj


def load(out):
    path = os.path.join(out, "raw_results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run the experiment first (python main.py, or python -m fznb.run)")
    df = pd.read_csv(path)
    df["error"] = df["error"].fillna("")
    bad = df[df.error != ""]
    if len(bad):
        print(f"warning: {len(bad)} failed fits (see 'error' column); they are excluded")
    df = df[df.error == ""].drop_duplicates(["dataset", "noise", "level", "seed", "method"], keep="last")
    return df


def with_clean_baseline(df):
    """Attach each fit's clean-data ARI (same dataset, seed, method) to compute the robustness drop."""
    clean = df[df.noise == "none"][["dataset", "seed", "method", "ari"]].rename(columns={"ari": "ari_clean"})
    noisy = df[df.noise != "none"].merge(clean, on=["dataset", "seed", "method"], how="left")
    noisy["ari_drop"] = noisy["ari_clean"] - noisy["ari"]
    return noisy


def block_matrix(noisy, methods, metric="ari"):
    """Rows = blocks (dataset x noise x level), columns = methods, values = seed-mean metric."""
    M = noisy.groupby(["dataset", "noise", "level", "method"])[metric].mean().unstack("method")
    return M[methods].dropna()


def stats(noisy, methods, out):
    M = block_matrix(noisy, methods)
    ranks = M.apply(lambda r: pd.Series(rankdata(-r.values), index=r.index), axis=1)
    avg = ranks.mean().sort_values()
    k, N = len(methods), len(M)
    if k >= 3 and N >= 2:
        fr = friedmanchisquare(*[M[m].values for m in methods])
    else:  # Friedman needs at least 3 methods; report a neutral result instead of crashing
        from types import SimpleNamespace
        fr = SimpleNamespace(statistic=float("nan"), pvalue=float("nan"))
    cd = Q05.get(k, np.nan) * np.sqrt(k * (k + 1) / (6.0 * N))
    with open(os.path.join(out, "stats_friedman.txt"), "w", encoding="utf-8") as f:
        f.write(f"Blocks (dataset x noise type x contamination level > 0): N = {N}\n")
        f.write(f"Methods: k = {k}\nFriedman chi2 = {fr.statistic:.2f}, p = {fr.pvalue:.3g}\n")
        f.write(f"Nemenyi critical difference (alpha = 0.05): CD = {cd:.3f}\n\nAverage rank on ARI (1 = best):\n")
        for m, r in avg.items():
            f.write(f"  {m:10s} {r:.2f}\n")
    pd.DataFrame({"avg_rank": avg}).to_csv(os.path.join(out, "ranks.csv"))
    # pairwise Wilcoxon on block means
    rows = []
    for a, b in itertools.combinations(methods, 2):
        d = M[a] - M[b]
        p = wilcoxon(M[a], M[b]).pvalue if (np.count_nonzero(d) >= 1 and len(d) >= 2) else 1.0
        tag = next((t for x, y, t in MECHANISM_PAIRS if {x, y} == {a, b}), "")
        rows.append(dict(method_a=a, method_b=b, mean_ari_a=M[a].mean(), mean_ari_b=M[b].mean(),
                         wins_a=int((d > 0).sum()), wins_b=int((d < 0).sum()), p=p, mechanism_pair=tag))
    pw = pd.DataFrame(rows)
    pw["p_holm"] = holm(pw.p)
    pw.sort_values(["mechanism_pair", "p_holm"], ascending=[False, True]).to_csv(
        os.path.join(out, "stats_pairwise.csv"), index=False)
    return M, avg, fr, cd, pw


def report(df, noisy, methods, avg, fr, cd, pw, abl, out):
    hi = noisy.level.max()
    t = noisy[noisy.level == hi].groupby(["noise", "method"]).ari.mean().unstack("method")[methods]
    clean = df[df.noise == "none"].groupby("method").ari.mean()[methods]
    auc = noisy.groupby("method").outlier_auc.mean()[methods]
    rt = df.groupby("method").runtime_s.median()[methods]
    cl = df[df.noise == "none"]
    ncl = cl.groupby(["dataset", "method"]).n_pred_clusters.mean().unstack("method")[methods].round(1)
    ncl["true c"] = cl.groupby("dataset").c.first()
    L = ["# Benchmark report (auto-generated by `python -m fznb.analyze`)", "",
         f"Datasets: {', '.join(dict.fromkeys(df.dataset))}. Seeds: {df.seed.nunique()}. "
         f"Noise types: {', '.join(sorted(noisy.noise.unique()))}. Levels: {[float(x) for x in sorted(noisy.level.unique())]}.", "",
         "## Mean ARI on clean data (level 0)", "", clean.round(3).to_frame("ARI").T.to_markdown(), "",
         "## Mean number of distinct predicted clusters on clean data (true c in last column)", "",
         "A value below c means clusters collapsed (e.g. coincident PFCM clusters, or FCM with m = 2 in high dimension).", "",
         ncl.to_markdown(), "",
         f"## Mean ARI at r = {hi:.2f}, by noise type (averaged over datasets and seeds)", "",
         t.round(3).to_markdown(), "",
         "## Average rank on ARI over all noisy conditions (1 = best)", "",
         avg.round(2).to_frame("avg rank").T.to_markdown(), "",
         f"Friedman chi2 = {fr.statistic:.2f}, p = {fr.pvalue:.3g}; Nemenyi CD (alpha 0.05) = {cd:.2f}. "
         "Methods whose average ranks differ by less than CD are not significantly different.", "",
         "## Mechanism pairs: Wilcoxon signed-rank on condition means, Holm-corrected", ""]
    mp = pw[pw.mechanism_pair != ""][["mechanism_pair", "method_a", "method_b", "mean_ari_a", "mean_ari_b",
                                      "wins_a", "wins_b", "p_holm"]]
    L += [mp.round(4).to_markdown(index=False), "",
          "## Outlier detection (mean ROC-AUC over noisy conditions) and median fit time", "",
          pd.DataFrame({"outlier AUC": auc.round(3), "median time (s)": rt.round(3)}).T.to_markdown(), ""]
    if abl is not None:
        L += ["## PFCM parameter selection without labels", "",
              "Mean ARI of the setting chosen by each rule; `oracle` needs labels, `default` is m = eta = 2, a = b = 1.", "",
              abl.to_markdown(), ""]
    with open(os.path.join(out, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def analyze(cfg, figures=True):
    out = cfg["out_dir"]
    df = load(out)
    methods = [m for m in cfg["methods"] if m in set(df.method)]
    if (df.noise == "none").sum() == 0 or (df.noise != "none").sum() == 0:
        raise ValueError("the results need both clean (level 0) and noisy runs; include 0.0 and a level > 0")
    noisy = with_clean_baseline(df)
    summ = df.groupby(["method", "dataset", "noise", "level"])[
        ["ari", "ami", "nmi", "acc", "outlier_auc", "outlier_f1", "runtime_s"]].agg(["mean", "std"]).round(4)
    summ.to_csv(os.path.join(out, "summary.csv"))
    M, avg, fr, cd, pw = stats(noisy, methods, out)
    rt = df.groupby(["dataset", "method"]).runtime_s.median().unstack("method")[methods]
    rt.round(4).to_csv(os.path.join(out, "runtime_median_s.csv"))
    p = os.path.join(out, "ablation_summary.csv")
    abl = pd.read_csv(p, index_col=0) if os.path.exists(p) else None
    report(df, noisy, methods, avg, fr, cd, pw, abl, out)
    with open(os.path.join(out, "REPORT.md"), encoding="utf-8") as f:
        print(f.read())
    if figures:
        from .report import build
        build(cfg)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/quick.yaml")
    ap.add_argument("--no-figures", action="store_true", help="statistics and REPORT.md only")
    args = ap.parse_args(argv)
    analyze(load_config(args.config), figures=not args.no_figures)


if __name__ == "__main__":
    main()
