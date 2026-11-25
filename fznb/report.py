"""Publication-ready figures (PDF + PNG) and tables (LaTeX + CSV) for reports and documents.

Usage:
    python -m fznb.report --config configs/full.yaml      (also run automatically by fznb.analyze and main.py)

Writes to <out_dir>/figures/, <out_dir>/tables/ and <out_dir>/all_figures_and_tables.pdf.
Figures are vector PDFs sized for two-column layouts (3.5 in single column, 7.16 in double).
Every figure also uses a distinct marker per method, so it stays readable in grayscale.

Include in LaTeX with, e.g.:
    \\includegraphics[width=\\columnwidth]{figures/fig_runtime.pdf}          (single column)
    \\begin{figure*} \\includegraphics[width=\\textwidth]{figures/fig_ari_vs_noise.pdf} \\end{figure*}
    \\input{tables/tab_ari_r20.tex}                                          (needs \\usepackage{booktabs})
"""
import argparse, os, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from . import data, noise
from .methods import REGISTRY, MECHANISM_PAIRS, make
from .config import load_config

ONE_COL, TWO_COL = 3.5, 7.16
COLORS = {"FCM": "#08519c", "PFCM": "#4292c6", "NC": "#17becf", "GMM": "#cb181d",
          "GMM+noise": "#fd8d3c", "t-mixture": "#c51b8a", "AE+FCM": "#238b45",
          "ST-RNC": "#000000", "RNC": "#636363", "ST-NC": "#969696"}
MARKERS = {"FCM": "o", "PFCM": "s", "NC": "D", "GMM": "^", "GMM+noise": "v", "t-mixture": "P", "AE+FCM": "X",
           "ST-RNC": "*", "RNC": "h", "ST-NC": "p"}
STYLE = {"fuzzy": "-", "probabilistic": "--", "neural/hybrid": ":"}
NOISE_LABEL = {"uniform": "Uniform background", "heavy": "Heavy-tailed", "rowwise": "Row-wise", "cellwise": "Cellwise"}
NOISE_SHORT = {"uniform": "Uniform", "heavy": "Heavy", "rowwise": "Row-wise", "cellwise": "Cellwise"}
REFS = {"FCM": "Bezdek 1981", "PFCM": "Pal et al. 2005", "NC": "Dav\\'e 1991", "GMM": "Dempster et al. 1977",
        "GMM+noise": "Banfield \\& Raftery 1993", "t-mixture": "Peel \\& McLachlan 2000", "AE+FCM": "Hinton \\& Salakhutdinov 2006 + FCM",
        "ST-RNC": "proposed (this work)", "RNC": "ablation of ST-RNC", "ST-NC": "ablation of ST-RNC"}
PARAMS = {"FCM": "$m=2$", "PFCM": "$m=\\eta=2$, $a=b=1$", "NC": "$m=2$, $\\lambda=0.5$", "GMM": "full cov., 5 inits",
          "GMM+noise": "uniform comp., init. share 0.05", "t-mixture": "$\\nu$ estimated", "AE+FCM": "MLP 128-64-$z$, 2,000 steps",
          "ST-RNC": "$q=0.99$, $m\\in\\{1.2,1.5,2\\}$", "RNC": "$q=0.99$, $m=2$", "ST-NC": "$\\lambda=0.5$, $m\\in\\{1.2,1.5,2\\}$"}


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def setup_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "lines.linewidth": 1.2,
        "lines.markersize": 4, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "figure.dpi": 150})


class Out:
    """Collects every figure and table so they can also be bundled into one PDF."""
    def __init__(self, root):
        self.fig_dir = os.path.join(root, "figures"); self.tab_dir = os.path.join(root, "tables")
        os.makedirs(self.fig_dir, exist_ok=True); os.makedirs(self.tab_dir, exist_ok=True)
        self.root, self.figs, self.tables = root, [], []

    def fig(self, fig, name, caption):
        fig.savefig(os.path.join(self.fig_dir, name + ".pdf"))
        fig.savefig(os.path.join(self.fig_dir, name + ".png"), dpi=300)
        self.figs.append((name, caption, fig))

    def table(self, df, name, caption, label, bold_max_cols=None, wide=False, fmt="{:.3f}",
              col_format=None, index=True, notes=None):
        df.to_csv(os.path.join(self.tab_dir, name + ".csv"))
        body = df.copy()
        if bold_max_cols is not None:
            for c in bold_max_cols:
                col = pd.to_numeric(df[c], errors="coerce")
                if col.notna().any():
                    best = col.max()
                    body[c] = [f"\\textbf{{{fmt.format(v)}}}" if np.isclose(v, best) else fmt.format(v)
                               if pd.notna(v) else "---" for v in col]
        body = body.map(lambda v: fmt.format(v) if isinstance(v, (float, np.floating)) and pd.notna(v) else
                        ("---" if isinstance(v, float) and pd.isna(v) else v))
        cols = list(body.columns)
        ncol = len(cols) + (1 if index else 0)
        col_format = col_format or ("l" * (1 if index else 0) + "r" * len(cols))
        env = "table*" if wide else "table"
        head = ([body.index.name or ""] if index else []) + [str(c) for c in cols]
        lines = [f"\\begin{{{env}}}[t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}", f"\\caption{{{caption}}}",
                 f"\\label{{{label}}}", f"\\begin{{tabular}}{{{col_format}}}", "\\toprule",
                 " & ".join(head) + " \\\\", "\\midrule"]
        for idx, row in body.iterrows():
            cells = ([str(idx)] if index else []) + [str(v) for v in row.values]
            lines.append(" & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        if notes:
            lines.append(f"\\par\\smallskip\\parbox{{\\linewidth}}{{\\footnotesize {notes}}}")
        lines.append(f"\\end{{{env}}}")
        _write(os.path.join(self.tab_dir, name + ".tex"), "\n".join(lines) + "\n")
        self.tables.append((name, caption, df))


def _legend_handles(methods, paradigm):
    return [plt.Line2D([], [], color=COLORS[m], marker=MARKERS[m], ls=STYLE[paradigm[m]], label=m) for m in methods]


def _curve(ax, x, y, m, paradigm, **kw):
    ax.plot(x, y, STYLE[paradigm[m]], color=COLORS[m], marker=MARKERS[m], label=m, **kw)


# ----------------------------------------------------------------------------------------- figures
def fig_ari_vs_noise(o, df, noisy, methods, paradigm, ntypes):
    # band = 95% CI over seeds of the dataset-averaged ARI (seed-to-seed uncertainty, not between-dataset spread)
    cl = df[df.noise == "none"].groupby(["method", "seed"]).ari.mean().groupby("method")
    clean, clean_ci = cl.mean(), 1.96 * cl.sem()
    gg = noisy.groupby(["noise", "level", "method", "seed"]).ari.mean().groupby(["noise", "level", "method"])
    g, ci = gg.mean(), 1.96 * gg.sem()
    fig, axes = plt.subplots(1, len(ntypes), figsize=(TWO_COL, 2.2), sharey=True)
    for ax, nt in zip(np.atleast_1d(axes), ntypes):
        lv = sorted(noisy[noisy.noise == nt].level.unique())
        for m in methods:
            y = np.array([clean[m]] + [g[(nt, l, m)] for l in lv])
            e = np.array([clean_ci[m]] + [ci[(nt, l, m)] for l in lv])
            _curve(ax, [0] + lv, y, m, paradigm)
            ax.fill_between([0] + lv, y - e, y + e, color=COLORS[m], alpha=0.12, lw=0)
        ax.set_title(NOISE_LABEL[nt]); ax.set_xlabel("contamination level $r$")
        ax.set_xticks([0] + lv)
    np.atleast_1d(axes)[0].set_ylabel(f"mean ARI ({df.dataset.nunique()} datasets)")
    fig.legend(handles=_legend_handles(methods, paradigm), loc="upper center", ncol=len(methods),
               bbox_to_anchor=(0.5, 1.08), frameon=False)
    o.fig(fig, "fig_ari_vs_noise", "Mean ARI on the original samples as contamination grows, averaged over datasets and seeds; "
          "shaded bands are 95\\% confidence intervals over seeds. Solid lines: fuzzy; dashed: probabilistic; "
          "dotted: neural/hybrid.")


def fig_ari_by_dataset(o, df, noisy, methods, paradigm, ntypes):
    datasets = list(dict.fromkeys(df.dataset))
    g = df.groupby(["dataset", "noise", "level", "method"]).ari.mean()
    fig, axes = plt.subplots(len(datasets), len(ntypes), figsize=(TWO_COL, 1.45 * len(datasets) + 0.4),
                             sharex=True, squeeze=False)
    for i, ds in enumerate(datasets):
        for j, nt in enumerate(ntypes):
            ax = axes[i, j]
            lv = sorted(noisy[(noisy.dataset == ds) & (noisy.noise == nt)].level.unique())
            for m in methods:
                try:
                    _curve(ax, [0] + lv, [g[(ds, "none", 0.0, m)]] + [g[(ds, nt, l, m)] for l in lv], m, paradigm,
                           ms=3, lw=1)
                except KeyError:
                    pass
            if i == 0: ax.set_title(NOISE_LABEL[nt])
            if j == 0: ax.set_ylabel(f"{ds}\nARI", fontsize=7)
            if i == len(datasets) - 1: ax.set_xlabel("contamination level $r$")
    fig.legend(handles=_legend_handles(methods, paradigm), loc="upper center", ncol=len(methods),
               bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    o.fig(fig, "fig_ari_by_dataset", "ARI versus contamination for every dataset (rows) and noise type (columns).")


def fig_robustness_drop(o, noisy, methods, paradigm, ntypes):
    g = noisy.groupby(["noise", "level", "method"]).ari_drop.mean()
    fig, axes = plt.subplots(1, len(ntypes), figsize=(TWO_COL, 2.2), sharey=True)
    for ax, nt in zip(np.atleast_1d(axes), ntypes):
        lv = sorted(noisy[noisy.noise == nt].level.unique())
        for m in methods:
            _curve(ax, lv, [g[(nt, l, m)] for l in lv], m, paradigm)
        ax.axhline(0, color="k", lw=0.6); ax.set_title(NOISE_LABEL[nt]); ax.set_xlabel("contamination level $r$")
        ax.set_xticks(lv)
    np.atleast_1d(axes)[0].set_ylabel("ARI lost vs. clean data")
    fig.legend(handles=_legend_handles(methods, paradigm), loc="upper center", ncol=len(methods),
               bbox_to_anchor=(0.5, 1.08), frameon=False)
    o.fig(fig, "fig_robustness_drop", "Robustness drop: clean-data ARI minus noisy ARI for the same dataset, seed and method "
          "(lower is more robust).")


def fig_ranks(o, avg, cd, pval, N):
    k = len(avg)
    fig, ax = plt.subplots(figsize=(ONE_COL, 1.55))
    order = avg.sort_values()
    ax.hlines(0, 1, k, color="k", lw=0.8)
    for x in range(1, k + 1):
        ax.vlines(x, -0.05, 0.05, color="k", lw=0.8); ax.text(x, -0.18, str(x), ha="center", va="top", fontsize=7)
    half = int(np.ceil(k / 2))
    for i, (m, r) in enumerate(order.items()):
        left = i < half
        y = 0.25 + 0.17 * (i if left else (k - 1 - i))
        xt = 0.4 if left else k + 0.6
        ax.plot([r, r, xt], [0, y, y], color=COLORS.get(m, "k"), lw=0.8)
        ax.plot(r, 0, MARKERS.get(m, "o"), color=COLORS.get(m, "k"), ms=4)
        ax.text(xt + (-0.05 if left else 0.05), y, f"{m} ({r:.2f})", ha="right" if left else "left", va="center", fontsize=6.5)
    ax.plot([1, 1 + cd], [-0.42, -0.42], "k-", lw=1.5)
    ax.text(1 + cd / 2, -0.47, f"CD = {cd:.2f}", ha="center", va="top", fontsize=6.5)
    ax.set_xlim(-1.6, k + 2.6); ax.set_ylim(-0.65, 0.25 + 0.17 * half + 0.05); ax.axis("off")
    ax.set_title(f"Average rank on ARI (Friedman $p$ = {pval:.1e}, $N$ = {N})", fontsize=7.5)
    o.fig(fig, "fig_ranks", "Critical-difference diagram: average rank of each method over all noisy conditions "
          "(1 = best). Methods closer than the Nemenyi CD are not significantly different at $\\alpha$ = 0.05.")


def fig_mechanism_pairs(o, noisy, ntypes):
    M = noisy.groupby(["dataset", "noise", "level", "method"]).ari.mean().unstack("method")
    pairs = [(a, b, t) for a, b, t in MECHANISM_PAIRS if a in M and b in M]
    fig, axes = plt.subplots(1, len(pairs), figsize=(TWO_COL, 2.45))
    nmark = {"uniform": "o", "heavy": "^", "rowwise": "D", "cellwise": "s"}
    for ax, (a, b, t) in zip(np.atleast_1d(axes), pairs):
        for nt in ntypes:
            s = M.xs(nt, level="noise")
            ax.scatter(s[a], s[b], s=12, marker=nmark[nt], alpha=0.75, label=NOISE_LABEL[nt],
                       edgecolor="k", linewidth=0.3)
        ax.plot([0, 1], [0, 1], "k--", lw=0.7); ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
        ax.set_aspect("equal"); ax.set_xlabel(f"{a} ARI"); ax.set_ylabel(f"{b} ARI")
        wins = int((M[a] > M[b]).sum()); n = int(M[[a, b]].dropna().shape[0])
        ax.set_title("\n".join(textwrap.wrap(t, 32)) + f"\n{a} better in {wins}/{n}", fontsize=7)
    np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=6, frameon=True)
    fig.tight_layout()
    o.fig(fig, "fig_mechanism_pairs", "Mechanism pairs. Each point is one condition (dataset $\\times$ noise "
          "type $\\times$ level); points below the diagonal favour the method on the x-axis.")


def fig_outlier_auc(o, noisy, methods, paradigm, ntypes):
    g = noisy.groupby(["noise", "level", "method"]).outlier_auc.mean()
    fig, axes = plt.subplots(1, len(ntypes), figsize=(TWO_COL, 2.2), sharey=True)
    for ax, nt in zip(np.atleast_1d(axes), ntypes):
        lv = sorted(noisy[noisy.noise == nt].level.unique())
        for m in methods:
            _curve(ax, lv, [g[(nt, l, m)] for l in lv], m, paradigm)
        ax.axhline(0.5, color="gray", lw=0.6, ls=":"); ax.set_title(NOISE_LABEL[nt])
        ax.set_xlabel("contamination level $r$"); ax.set_xticks(lv)
    np.atleast_1d(axes)[0].set_ylabel("outlier ROC-AUC")
    fig.legend(handles=_legend_handles(methods, paradigm), loc="upper center", ncol=len(methods),
               bbox_to_anchor=(0.5, 1.08), frameon=False)
    o.fig(fig, "fig_outlier_auc", "Outlier detection: ROC-AUC of each method's outlier score against the injected "
          "points (uniform, heavy) or corrupted rows (cellwise). 0.5 = chance.")


def fig_runtime(o, df, methods):
    rt = df.groupby(["dataset", "method"]).runtime_s.median().unstack("method")[methods]
    fig, ax = plt.subplots(figsize=(TWO_COL, 2.1))
    w = 0.8 / len(methods); x = np.arange(len(rt))
    for i, m in enumerate(methods):
        ax.bar(x + (i - (len(methods) - 1) / 2) * w, rt[m], w, color=COLORS[m], label=m, edgecolor="k", lw=0.3)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(rt.index, fontsize=7)
    ax.set_ylabel("median fit time (s)"); ax.legend(ncol=len(methods), loc="upper center", bbox_to_anchor=(0.5, 1.2),
                                                  frameon=False)
    o.fig(fig, "fig_runtime", "Median wall-clock fit time per dataset (log scale), over all noise conditions and seeds.")


def fig_clean_vs_noisy(o, df, noisy, methods, paradigm):
    clean = df[df.noise == "none"].groupby("method").ari.mean()
    hi = noisy.level.max()
    nz = noisy[noisy.level == hi].groupby("method").ari.mean()
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.6))
    for m in methods:
        ax.scatter(clean[m], nz[m], color=COLORS[m], marker=MARKERS[m], s=40, edgecolor="k", lw=0.4, zorder=3)
        ax.annotate(m, (clean[m], nz[m]), textcoords="offset points", xytext=(4, 3), fontsize=6.5)
    lo = min(clean.min(), nz.min()) - 0.05; up = max(clean.max(), nz.max()) + 0.05
    ax.plot([lo, up], [lo, up], "k--", lw=0.6); ax.set_xlim(lo, up); ax.set_ylim(lo, up)
    ax.set_xlabel("mean ARI, clean data"); ax.set_ylabel(f"mean ARI, $r$ = {hi:.2f} (all noise types)")
    o.fig(fig, "fig_clean_vs_noisy", "Accuracy-robustness trade-off: clean-data ARI against ARI at the highest "
          "contamination. Distance below the diagonal is the ARI lost to noise.")


def fig_noise_protocol(o, r=0.10):
    D = data.load("S1_2d_unequal")
    fig, axes = plt.subplots(1, 1 + len(noise.NOISE_TYPES), figsize=(TWO_COL, 1.7), sharex=True, sharey=True)
    titles = ["Clean data"] + [f"{NOISE_LABEL[n]}\n$r$ = {r:.2f}" for n in noise.NOISE_TYPES]
    cmap = ["#4292c6", "#fd8d3c", "#74c476"]
    for ax, nt, title in zip(axes, [None] + list(noise.NOISE_TYPES), titles):
        if nt is None:
            X, inj, con = D.X, np.zeros(len(D.y), bool), np.zeros(len(D.y), bool)
        else:
            X, inj, con = noise.inject(D.X, D.y, nt, r, seed=1)
        n0 = len(D.y)
        for k in range(D.n_clusters):
            sel = (D.y == k) & ~con[:n0]
            ax.scatter(X[:n0][sel, 0], X[:n0][sel, 1], s=3, color=cmap[k], lw=0)
        ax.scatter(X[con, 0], X[con, 1], s=10, marker="x", color="k", lw=0.7, label="noise / corrupted")
        ax.set_title(title, fontsize=7); ax.set_aspect("equal", adjustable="box")
    axes[0].set_xlim(-4, 4); axes[0].set_ylim(-4, 4)
    axes[-1].legend(loc="lower right", fontsize=5.5, markerscale=0.8)
    o.fig(fig, "fig_noise_protocol", "The shared noise protocol on the 2-D synthetic data (S1). Crosses mark injected "
          "points or corrupted samples. Axes are clipped to $[-4, 4]$, so the farthest heavy-tailed and row-wise points, "
          "and all cellwise-corrupted samples (values of $\\pm$5--10 SD), lie outside the view.")


def fig_partitions(o, r=0.10, show=("FCM", "PFCM", "NC", "GMM", "GMM+noise", "t-mixture")):
    D = data.load("S1_2d_unequal")
    X, inj, con = noise.inject(D.X, D.y, "uniform", r, seed=1)
    fig, axes = plt.subplots(2, 3, figsize=(TWO_COL, 4.3), sharex=True, sharey=True)
    cmap = np.array(["#4292c6", "#fd8d3c", "#74c476"])
    from sklearn.metrics import adjusted_rand_score
    for ax, m in zip(axes.ravel(), show):
        fit = make(m, seed=0).fit(X, D.n_clusters)
        ax.scatter(X[:, 0], X[:, 1], s=4, c=cmap[fit.labels % 3], lw=0)
        if fit.outlier_flag is not None:
            f = fit.outlier_flag
            ax.scatter(X[f, 0], X[f, 1], s=14, facecolor="none", edgecolor="k", lw=0.6, label="flagged as noise")
            ax.legend(loc="lower left", fontsize=5.5)
        if "centers" in fit.info:
            V = fit.info["centers"]; ax.scatter(V[:, 0], V[:, 1], marker="*", s=90, color="k", edgecolor="w", lw=0.5)
        ari = adjusted_rand_score(D.y, fit.labels[~inj])
        ax.set_title(f"{m} (ARI = {ari:.2f})", fontsize=7.5); ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(X[:, 0].min() - 0.2, X[:, 0].max() + 0.2); ax.set_ylim(X[:, 1].min() - 0.2, X[:, 1].max() + 0.2)
    fig.tight_layout()
    o.fig(fig, "fig_partitions_example", f"Example partitions of S1 with {int(r * 100)}% uniform noise (seed 1). "
          "Colours: assigned cluster; stars: fuzzy cluster centres; circles: points the method flags as noise.")


def fig_ablation(o, root):
    p = os.path.join(root, "ablation_raw.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    if "method" in df:
        df = df[df.method == "PFCM"]
    surf = df.groupby(["m", "eta"]).ari.mean().unstack()
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.4))
    im = ax.imshow(surf.values, cmap="viridis", origin="lower", aspect="auto")
    ax.set_xticks(range(surf.shape[1])); ax.set_xticklabels([f"{v:g}" for v in surf.columns])
    ax.set_yticks(range(surf.shape[0])); ax.set_yticklabels([f"{v:g}" for v in surf.index])
    vmid = np.nanmean(surf.values)
    for i in range(surf.shape[0]):
        for j in range(surf.shape[1]):
            v = surf.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="w" if v < vmid else "k")
    ax.set_xlabel(r"typicality exponent $\eta$"); ax.set_ylabel("fuzzifier $m$"); ax.grid(False)
    fig.colorbar(im, ax=ax, label="mean ARI")
    o.fig(fig, "fig_ablation_surface", "PFCM sensitivity: mean ARI over the fuzzifier $m$ and typicality exponent "
          "$\\eta$, averaged over datasets, noise types, $(a, b)$ and seeds ($r$ = 0.10).")
    s = pd.read_csv(os.path.join(root, "ablation_selection.csv"))
    cols = ["default", "neg_xie_beni", "silhouette", "trimmed_silhouette", "oracle"]
    names = ["Default", "Xie–Beni", "Silhouette", "Trimmed sil.", "Oracle"]
    tab = s.groupby("dataset")[cols].mean()
    fig, ax = plt.subplots(figsize=(TWO_COL, 2.1))
    w = 0.8 / len(cols); x = np.arange(len(tab))
    shades = ["#bdbdbd", "#6baed6", "#3182bd", "#08519c", "#252525"]
    for i, (c, nm) in enumerate(zip(cols, names)):
        ax.bar(x + (i - (len(cols) - 1) / 2) * w, tab[c], w, label=nm, color=shades[i], edgecolor="k", lw=0.3,
               hatch="//" if c == "oracle" else None)
    ax.set_xticks(x); ax.set_xticklabels(tab.index, fontsize=7); ax.set_ylabel("PFCM ARI")
    ax.legend(ncol=len(cols), loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    o.fig(fig, "fig_parameter_selection", "Label-free parameter selection for PFCM: ARI of the setting chosen by "
          "each rule, compared with the default ($m=\\eta=2$, $a=b=1$) and the oracle (best setting, requires labels).")


# ------------------------------------------------------------------------------------------ tables
def tables(o, df, noisy, methods, root):
    ntypes = [n for n in noise.NOISE_TYPES if n in set(noisy.noise)]
    hi = noisy.level.max()
    # T1 datasets
    rows = []
    for ds in dict.fromkeys(df.dataset):
        D = data.load(ds)
        rows.append({"Dataset": ds.replace("_", "\\_"), "$n$": len(D.y), "$d$": D.X.shape[1], "$c$": D.n_clusters,
                     "Source": D.source.replace("&", "\\&").replace("_", "\\_").replace(" (sklearn bundled)", "")})
    t = pd.DataFrame(rows).set_index("Dataset")
    o.table(t, "tab_datasets", "Datasets (z-scored; constant features removed). Real datasets are the UCI versions "
            "bundled with scikit-learn", "tab:datasets", col_format="lrrr>{\\raggedright\\arraybackslash}p{3.3cm}")
    # T2 methods
    t = pd.DataFrame([{"Method": m, "Paradigm": REGISTRY[m].paradigm, "Mechanism": REGISTRY[m].mechanism,
                       "Default parameters": PARAMS.get(m, ""), "Reference": REFS.get(m, "")} for m in methods]).set_index("Method")
    o.table(t, "tab_methods", "Methods compared, by paradigm and noise-handling mechanism", "tab:methods",
            wide=True, col_format="lllll")
    # T3 clean ARI by dataset
    t = df[df.noise == "none"].groupby(["dataset", "method"]).ari.mean().unstack("method")[methods]
    t.loc["Mean"] = t.mean()
    t.index = [i.replace("_", "\\_") for i in t.index]; t.index.name = "Dataset"
    o.table(t, "tab_ari_clean", "Mean ARI on clean data (no noise), 10 seeds; best per row in bold",
            "tab:clean", wide=True)
    _bold_rows(o, "tab_ari_clean")
    # T4 ARI at highest level, mean (sd), plus average rank
    a = noisy[noisy.level == hi].groupby(["method", "noise"]).ari.agg(["mean", "std"])
    ranks = pd.read_csv(os.path.join(root, "ranks.csv"), index_col=0)["avg_rank"]
    rows = {}
    for m in methods:
        rows[m] = {NOISE_SHORT[nt]: a.loc[(m, nt), "mean"] for nt in ntypes}
        rows[m]["Avg. rank"] = ranks[m]
    t = pd.DataFrame(rows).T; t.index.name = "Method"
    sd = {m: {NOISE_SHORT[nt]: a.loc[(m, nt), "std"] for nt in ntypes} for m in methods}
    o.table(t, "tab_ari_r20", f"Mean ARI at contamination $r$ = {hi:.2f} over all datasets and seeds, and average rank "
            "over all noisy conditions (lower rank is better)", "tab:ari", fmt="{:.3f}")
    _rewrite_with_sd(o, "tab_ari_r20", t, sd, [NOISE_SHORT[nt] for nt in ntypes], rank_col="Avg. rank")
    # T5 robustness drop by noise and level
    t = noisy.groupby(["noise", "level", "method"]).ari_drop.mean().unstack("method")[methods]
    t.index = [f"{NOISE_LABEL[n]}, {l:.2f}" for n, l in t.index]; t.index.name = "Noise, $r$"
    o.table(t, "tab_robustness_drop", "Robustness drop (clean ARI $-$ noisy ARI), mean over datasets and seeds; "
            "smallest drop per row in bold", "tab:drop", wide=True)
    _bold_rows(o, "tab_robustness_drop", best="min")
    # T6 mechanism pairs
    pw = pd.read_csv(os.path.join(root, "stats_pairwise.csv"))
    mp = pw[pw.mechanism_pair.fillna("") != ""][["mechanism_pair", "method_a", "method_b", "mean_ari_a",
                                                  "mean_ari_b", "wins_a", "wins_b", "p_holm"]].copy()
    mp.columns = ["Pair", "A", "B", "ARI A", "ARI B", "A wins", "B wins", "$p$ (Holm)"]
    mp["$p$ (Holm)"] = mp["$p$ (Holm)"].map(lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}")
    mp = mp.set_index("Pair")
    o.table(mp, "tab_mechanism_pairs", "Mechanism pairs across paradigms: Wilcoxon signed-rank tests on "
            "condition means, Holm-corrected", "tab:pairs", wide=True, col_format="lllrrrrr")
    # T7 Friedman + all pairwise
    pw2 = pw[["method_a", "method_b", "mean_ari_a", "mean_ari_b", "wins_a", "wins_b", "p_holm"]].copy()
    pw2.columns = ["A", "B", "ARI A", "ARI B", "A wins", "B wins", "$p$ (Holm)"]
    pw2["$p$ (Holm)"] = pw2["$p$ (Holm)"].map(lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}")
    o.table(pw2.set_index("A"), "tab_pairwise_all", "All pairwise Wilcoxon signed-rank tests on ARI "
            "(Holm-corrected)", "tab:pairwise", col_format="llrrrrr",
            notes=_read(os.path.join(root, "stats_friedman.txt")).split("\n\n")[0].replace("\n", "; "))
    # T8 outlier detection + runtime
    t = pd.DataFrame({"Outlier AUC": noisy.groupby("method").outlier_auc.mean(),
                      "Outlier F1 (native flag)": noisy.groupby("method").outlier_f1.mean(),
                      "Median time (s)": df.groupby("method").runtime_s.median()}).loc[methods]
    t.index.name = "Method"
    o.table(t, "tab_outliers_runtime", "Outlier detection (mean over noisy conditions) and median fit time. "
            "F1 is shown only for methods with a built-in outlier decision.", "tab:outliers")
    # T9 ablation
    p = os.path.join(root, "ablation_summary.csv")
    if os.path.exists(p):
        t = pd.read_csv(p, index_col=0)
        t = t.rename(columns={"oracle": "Oracle", "default": "Default", "neg_xie_beni": "Xie--Beni",
                              "silhouette": "Silhouette", "trimmed_silhouette": "Trimmed sil.", "worst": "Worst"})
        t.index = [i.replace("_", "\\_") for i in t.index]; t.index.name = "Dataset"
        o.table(t, "tab_parameter_selection", "PFCM parameter selection without labels ($r$ = 0.10): ARI of the "
                "setting each rule picks; Oracle requires labels", "tab:selection", wide=True)


def _bold_rows(o, name, best="max"):
    """Re-write a table with the best value in each row in bold."""
    df = [d for n, c, d in o.tables if n == name][0]
    path = os.path.join(o.tab_dir, name + ".tex")
    lines = _read(path).split("\n")
    start = lines.index("\\midrule") + 1
    for k, (idx, row) in enumerate(df.iterrows()):
        vals = pd.to_numeric(row, errors="coerce")
        b = vals.max() if best == "max" else vals.min()
        cells = [str(idx)] + [f"\\textbf{{{v:.3f}}}" if np.isclose(v, b) else f"{v:.3f}" for v in vals]
        lines[start + k] = " & ".join(cells) + " \\\\"
    _write(path, "\n".join(lines))


def _rewrite_with_sd(o, name, t, sd, cols, rank_col):
    path = os.path.join(o.tab_dir, name + ".tex")
    lines = _read(path).split("\n")
    start = lines.index("\\midrule") + 1
    best = {c: t[c].max() for c in cols}; best_rank = t[rank_col].min()
    for k, m in enumerate(t.index):
        cells = [m]
        for c in cols:
            s = f"{t.loc[m, c]:.2f} ({sd[m][c]:.2f})"
            cells.append(f"\\textbf{{{s}}}" if np.isclose(t.loc[m, c], best[c]) else s)
        r = f"{t.loc[m, rank_col]:.2f}"
        cells.append(f"\\textbf{{{r}}}" if np.isclose(t.loc[m, rank_col], best_rank) else r)
        lines[start + k] = " & ".join(cells) + " \\\\"
    _write(path, "\n".join(lines))


# ------------------------------------------------------------------------------------------ bundle
def _strip_tex(s):
    for a, b in [("\\textbf{", ""), ("$", ""), ("\\_", "_"), ("\\&", "&"), ("\\'e", "é"), ("\\eta", "η"),
                 ("\\lambda", "λ"), ("\\nu", "ν"), ("\\times", "×"), ("\\alpha", "α"), ("--", "–"), ("\\", "")]:
        s = s.replace(a, b)
    return s.replace("}", "").replace("{", "")


def bundle(o, title):
    path = os.path.join(o.root, "all_figures_and_tables.pdf")
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.5, 11)); fig.text(0.5, 0.6, title, ha="center", fontsize=16)
        fig.text(0.5, 0.55, f"{len(o.figs)} figures and {len(o.tables)} tables. Vector PDFs of each figure are in "
                 "figures/;\nLaTeX and CSV versions of each table are in tables/.", ha="center", fontsize=9)
        pdf.savefig(fig); plt.close(fig)
        for i, (name, cap, f) in enumerate(o.figs, 1):
            w, h = f.get_size_inches()
            page = plt.figure(figsize=(8.5, 11))
            page.text(0.08, 0.95, f"Figure {i}: {name}.pdf", fontsize=10, weight="bold")
            page.text(0.08, 0.93, "\n".join(textwrap.wrap(_strip_tex(cap), 110)), fontsize=8, va="top")
            f.savefig(os.path.join(o.fig_dir, "_tmp.png"), dpi=200)
            img = plt.imread(os.path.join(o.fig_dir, "_tmp.png"))
            scale = min(7.2 / w, 8.5 / h)
            ax = page.add_axes([(8.5 - w * scale) / 2 / 8.5, (0.86 - h * scale / 11), w * scale / 8.5, h * scale / 11])
            ax.imshow(img); ax.axis("off")
            pdf.savefig(page); plt.close(page); plt.close(f)
        os.remove(os.path.join(o.fig_dir, "_tmp.png"))
        for i, (name, cap, df) in enumerate(o.tables, 1):
            page = plt.figure(figsize=(11, 8.5))
            page.text(0.05, 0.94, f"Table {i}: {name}.tex", fontsize=10, weight="bold")
            page.text(0.05, 0.91, "\n".join(textwrap.wrap(_strip_tex(cap), 150)), fontsize=8, va="top")
            ax = page.add_axes([0.05, 0.05, 0.9, 0.8]); ax.axis("off")
            show = df.copy().map(lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) else _strip_tex(str(v)))
            show.index = [_strip_tex(str(i)) for i in show.index]
            show.columns = [_strip_tex(str(c)) for c in show.columns]
            tb = ax.table(cellText=show.values, rowLabels=show.index, colLabels=show.columns, loc="upper center",
                          cellLoc="center")
            tb.auto_set_font_size(False); tb.set_fontsize(7 if len(show) < 25 else 5.5); tb.scale(1, 1.25)
            pdf.savefig(page); plt.close(page)
    return path


def build(cfg):
    from .analyze import load, with_clean_baseline
    setup_style()
    root = cfg["out_dir"]
    for need in ["raw_results.csv", "ranks.csv", "stats_friedman.txt", "stats_pairwise.csv"]:
        if not os.path.exists(os.path.join(root, need)):
            raise FileNotFoundError(f"{os.path.join(root, need)} not found - run the experiment and "
                                    "`python -m fznb.analyze` first (or just `python main.py`)")
    df = load(root)
    methods = [m for m in cfg["methods"] if m in set(df.method)]
    paradigm = {m: REGISTRY[m].paradigm for m in methods}
    noisy = with_clean_baseline(df)
    ntypes = [n for n in noise.NOISE_TYPES if n in set(noisy.noise)]
    ranks = pd.read_csv(os.path.join(root, "ranks.csv"), index_col=0)["avg_rank"]
    fr = _read(os.path.join(root, "stats_friedman.txt"))
    import re
    N = int(re.search(r"N = (\d+)", fr).group(1))
    p = float(re.search(r"p = ([0-9.eE+-]+|nan)", fr).group(1))
    cd = float(re.search(r"CD = ([0-9.]+|nan)", fr).group(1))
    o = Out(root)
    fig_noise_protocol(o)
    fig_partitions(o)
    fig_ari_vs_noise(o, df, noisy, methods, paradigm, ntypes)
    fig_ari_by_dataset(o, df, noisy, methods, paradigm, ntypes)
    fig_robustness_drop(o, noisy, methods, paradigm, ntypes)
    fig_clean_vs_noisy(o, df, noisy, methods, paradigm)
    fig_ranks(o, ranks, cd, p, N)
    fig_mechanism_pairs(o, noisy, ntypes)
    fig_outlier_auc(o, noisy, methods, paradigm, ntypes)
    fig_runtime(o, df, methods)
    fig_ablation(o, root)
    tables(o, df, noisy, methods, root)
    from . import diagnostics
    diagnostics.build_all(o, cfg, df, methods)
    path = bundle(o, "Fuzzy Clustering Against Noise: Benchmark Figures and Tables")
    print(f"{len(o.figs)} figures -> {o.fig_dir}\n{len(o.tables)} tables -> {o.tab_dir}\nbundle -> {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/quick.yaml")
    build(load_config(ap.parse_args(argv).config))


if __name__ == "__main__":
    main()
