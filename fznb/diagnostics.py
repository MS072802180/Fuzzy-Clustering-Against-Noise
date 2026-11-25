"""Diagnostic and illustrative figures (adapted from the project's original visualize.py).

These complement the aggregate results in fznb.report: they show *why* the methods behave as they
do on single, reproducible runs (seed 0), plus the c- and m-ablations and runtime scaling.
Called automatically by fznb.report.build; each function adds figures/tables to the shared Out.

Figures
    fig_pca_partitions_<dataset>   truth + every method, PCA view, 10% row-wise noise (corrupted samples marked)
    fig_convergence                objective / log-likelihood / AE loss per iteration, clean vs noisy
    fig_membership_typicality      FCM memberships vs PFCM typicality vs NC noise membership (heatmaps)
    fig_outlier_scores             each method's outlier score on a PCA view, injected points circled
    fig_ae_latent                  autoencoder latent space (clean vs noisy) and training loss
    fig_ablation_m                 ARI vs fuzzifier m for FCM and PFCM, per dataset
    fig_ablation_c                 ARI vs number of clusters, and how often each index recovers c
    fig_runtime_scaling            fit time vs n, and iterations to convergence
    fig_ari_clean_bars             clean-data ARI and NMI per dataset and method (values printed)
Tables
    tab_c_selection, tab_fcm_m, tab_fpc_iterations
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from . import data, noise
from .methods import make, REGISTRY

SEED = 0
CLUSTER_COLORS = ["#4292c6", "#fd8d3c", "#74c476", "#9e9ac8", "#e7298a", "#a6761d",
                  "#666666", "#e6ab02", "#1b9e77", "#d95f02", "#7570b3", "#66a61e"]


def _fit(name, X, c, params, **override):
    kw = dict(params.get(name, {})); kw.update(override)
    return make(name, seed=SEED, **kw).fit(X, c)


def _noisy(D, nt, r, seed=1):
    return noise.inject(D.X, D.y, nt, r, seed=seed)


def _colors(labels):
    return np.array(CLUSTER_COLORS)[np.asarray(labels) % len(CLUSTER_COLORS)]


# ---------------------------------------------------------------------------------- single-run figures
def fig_pca_partitions(o, ds, methods, params, nt="rowwise", r=0.10):
    from .report import TWO_COL
    D = data.load(ds)
    X, inj, con = _noisy(D, nt, r)
    pca = PCA(2, random_state=0).fit(D.X)                   # projection fitted on clean data
    P, P0 = pca.transform(X), pca.transform(D.X)
    pad = 0.15 * (P0.max(0) - P0.min(0))
    lo, hi = P0.min(0) - pad, P0.max(0) + pad
    panels = [("Ground truth", np.r_[D.y, -np.ones(inj.sum(), int)], None)]
    for m in methods:
        fit = _fit(m, X, D.n_clusters, params)
        panels.append((f"{m} (ARI {adjusted_rand_score(D.y, fit.labels[~inj]):.2f})", fit.labels, fit.outlier_flag))
    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(TWO_COL, 1.85 * nrow + 0.2), sharex=True, sharey=True)
    for ax, (title, lab, flag) in zip(axes.ravel(), panels):
        ok = lab >= 0
        ax.scatter(P[ok & ~con, 0], P[ok & ~con, 1], s=5, c=_colors(lab[ok & ~con]), lw=0)
        ax.scatter(P[con, 0], P[con, 1], s=14, marker="x", c=_colors(np.where(lab[con] >= 0, lab[con], 6)), lw=0.8)
        if flag is not None:
            ax.scatter(P[flag, 0], P[flag, 1], s=26, facecolor="none", edgecolor="k", lw=0.5)
        ax.set_title(title, fontsize=7); ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("PC 2", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("PC 1", fontsize=7)
    fig.tight_layout()
    o.fig(fig, f"fig_pca_partitions_{D.name}",
          f"Partitions of {D.name} with {int(r * 100)}% {nt} noise, shown on the first two principal components of the "
          "clean data (seed 0). Colour: assigned cluster; crosses: corrupted samples (most lie outside the view); "
          "open circles: samples a method flags as noise.")


def fig_convergence(o, params, datasets=("iris", "wine"), nt="rowwise", r=0.10):
    from .report import TWO_COL
    panels = [("FCM", "objective $J$"), ("PFCM", "objective $J$"), ("NC", "objective $J$"),
              ("GMM+noise", "$-$log-likelihood / $n$"), ("t-mixture", "$-$log-likelihood / $n$"),
              ("AE+FCM", "AE reconstruction loss")]
    panels = [(m, y) for m, y in panels if m in REGISTRY]
    fig, axes = plt.subplots(2, 3, figsize=(TWO_COL, 3.9))
    styles = {}
    for ax, (m, ylab) in zip(axes.ravel(), panels):
        for i, ds in enumerate(datasets):
            D = data.load(ds)
            for noisy in (False, True):
                X = _noisy(D, nt, r)[0] if noisy else D.X
                try:
                    fit = _fit(m, X, D.n_clusters, params)
                except ImportError:
                    continue
                h = np.asarray(fit.info.get("loss_history" if m == "AE+FCM" else "history", []), float)
                if m in ("GMM+noise", "t-mixture"):
                    h = -h / len(X)
                ls = "--" if noisy else "-"
                lab = f"{ds}, {'10% ' + nt if noisy else 'clean'}"
                ax.plot(np.arange(1, len(h) + 1), h, ls, color=CLUSTER_COLORS[i * 2], lw=1, label=lab)
        ax.set_title(m); ax.set_xlabel("epoch" if m == "AE+FCM" else "iteration"); ax.set_ylabel(ylab, fontsize=7)
        ax.set_xscale("log")
        if m in ("FCM", "PFCM", "NC"):
            ax.set_yscale("log")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    o.fig(fig, "fig_convergence", "Convergence of each iterative method for its best restart (seed 0), on clean data "
          "(solid) and with 10% row-wise noise (dashed). Fuzzy methods: objective; probabilistic: negative "
          "log-likelihood per sample; AE+FCM: autoencoder training loss.")


def fig_membership_typicality(o, params, ds="iris", nt="uniform", r=0.10, max_rows=300):
    from .report import TWO_COL
    D = data.load(ds)
    X, inj, con = _noisy(D, nt, r)
    c = D.n_clusters
    fcm = _fit("FCM", X, c, params)
    pf = _fit("PFCM", X, c, params)
    nc = _fit("NC", X, c, params)
    st = _fit("ST-RNC", X, c, params) if "ST-RNC" in REGISTRY else None
    # rows: real samples sorted by true class, then injected points; columns aligned to FCM clusters.
    # On large data, show a random sample of max_rows real rows (and the same 10% share of injected
    # rows) so the heatmap stays readable.
    rng = np.random.default_rng(0)
    real = np.arange(len(D.y))
    injected = np.arange(len(D.y), len(X))
    if len(real) > max_rows:
        real = np.sort(rng.choice(real, max_rows, replace=False))
        k = max(1, int(round(len(injected) * max_rows / len(D.y))))
        injected = np.sort(rng.choice(injected, min(k, len(injected)), replace=False))
    order = np.r_[real[np.argsort(D.y[real], kind="stable")], injected]
    n_real = len(real)

    def align(V, Vref):
        from scipy.optimize import linear_sum_assignment
        C = ((Vref[:, None, :] - V[None, :, :]) ** 2).sum(-1)
        return linear_sum_assignment(C)[1]

    ref = fcm.info["centers"]
    mats = [("FCM membership $u_{ik}$", fcm.info["U"]),
            ("PFCM typicality $t_{ik}$", pf.info["T"][align(pf.info["centers"], ref)]),
            ("NC membership (+ noise)", np.vstack([nc.info["U"][align(nc.info["centers"], ref)], nc.info["u_noise"]]))]
    if st is not None:
        mats.append((f"ST-RNC membership (+ noise), $m$={st.info['m']:g}",
                     np.vstack([st.info["U"][align(st.info["centers"], ref)], st.info["u_noise"]])))
    else:
        mats.insert(1, ("PFCM membership $u_{ik}$", pf.info["U"][align(pf.info["centers"], ref)]))
    fig, axes = plt.subplots(1, 4, figsize=(TWO_COL, 3.0), sharey=True)
    for ax, (title, M) in zip(axes, mats):
        im = ax.imshow(M[:, order].T, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        ax.axhline(n_real - 0.5, color="w", lw=1.2, ls="--")
        k = M.shape[0]
        ax.set_xticks(range(k)); ax.set_xticklabels([str(i + 1) for i in range(c)] + (["N"] if k > c else []))
        ax.set_title(title, fontsize=7); ax.set_xlabel("cluster"); ax.grid(False)
    shown = "" if n_real == len(D.y) else f"{n_real} of {len(D.y)} "
    axes[0].set_ylabel(f"{shown}real samples (by class), then {len(order) - n_real} injected", fontsize=7)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    o.fig(fig, "fig_membership_typicality",
          f"Why the sum-to-one constraint hurts: memberships on {D.name} with {int(r * 100)}% {nt} noise (rows below the "
          "dashed line are injected points). FCM memberships give every outlier a full unit of membership; PFCM "
          "typicality and the noise clusters of NC and the proposed ST-RNC (column N) separate them.")


def fig_outlier_scores(o, params, ds="iris", nt="uniform", r=0.10):
    from .report import TWO_COL
    D = data.load(ds)
    X, inj, con = _noisy(D, nt, r)
    P = PCA(2, random_state=0).fit(D.X).transform(X)
    show = [("PFCM", "1 $-$ max typicality"), ("NC", "noise membership"), ("GMM+noise", "noise posterior"),
            ("t-mixture", "$-$log density"), ("AE+FCM", "reconstruction error")]
    fig, axes = plt.subplots(1, len(show), figsize=(TWO_COL, 1.9), sharex=True, sharey=True)
    from sklearn.metrics import roc_auc_score
    for ax, (m, lab) in zip(axes, show):
        try:
            fit = _fit(m, X, D.n_clusters, params)
        except ImportError:
            ax.axis("off"); continue
        s = np.asarray(fit.outlier_score, float)
        rank = s.argsort().argsort() / (len(s) - 1)          # rank-normalised so all panels share a scale
        sc = ax.scatter(P[:, 0], P[:, 1], c=rank, cmap="plasma", s=6, lw=0)
        ax.scatter(P[con, 0], P[con, 1], s=22, facecolor="none", edgecolor="k", lw=0.5)
        ax.set_title(f"{m}\nAUC = {roc_auc_score(con, s):.2f}", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.01, label="score rank")
    o.fig(fig, "fig_outlier_scores", f"Outlier scores on {D.name} with {int(r * 100)}% {nt} noise (PCA of the clean "
          "data; colour = rank of the score, bright = more outlying; circles = injected points). Scores: PFCM, "
          "1 $-$ max typicality; NC, noise-cluster membership; GMM+noise, noise posterior; t-mixture, $-$log "
          "density; AE+FCM, reconstruction error. AUC: ROC-AUC against the injected points.")


def fig_ae_latent(o, params, ds="wine", nt="rowwise", r=0.10):
    from .report import TWO_COL
    try:
        import torch  # noqa: F401
    except ImportError:
        return
    D = data.load(ds)
    Xn, inj, con = _noisy(D, nt, r)
    fits = [(D.X, np.zeros(len(D.y), bool), _fit("AE+FCM", D.X, D.n_clusters, params)),
            (Xn, con, _fit("AE+FCM", Xn, D.n_clusters, params))]
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL, 2.1))
    for ax, (X, cmask, fit), title in zip(axes, fits, ["clean", f"{int(r * 100)}% {nt} noise"]):
        Z = fit.info["Z"]
        Zp = PCA(2, random_state=0).fit_transform(Z) if Z.shape[1] > 2 else Z
        ax.scatter(Zp[~cmask, 0], Zp[~cmask, 1], s=6, c=_colors(fit.labels[~cmask]), lw=0)
        ax.scatter(Zp[cmask, 0], Zp[cmask, 1], s=16, marker="x", c="k", lw=0.7)
        q = Zp[~cmask]                                      # frame the real samples; far outliers fall outside
        lo, hi = np.percentile(q, 1, axis=0), np.percentile(q, 99, axis=0)
        pad = 0.25 * (hi - lo)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_title(f"latent space, {title} (ARI {adjusted_rand_score(D.y, fit.labels):.2f})", fontsize=7)
        ax.set_xlabel("latent PC 1"); ax.set_ylabel("latent PC 2")
    for (X, cmask, fit), ls, lab in zip(fits, ["-", "--"], ["clean", "noisy"]):
        axes[2].plot(fit.info["loss_history"], ls, color="k", lw=1, label=lab)
    axes[2].set_yscale("log"); axes[2].set_xlabel("epoch"); axes[2].set_ylabel("reconstruction MSE")
    axes[2].set_title("autoencoder training", fontsize=7); axes[2].legend()
    fig.tight_layout()
    o.fig(fig, "fig_ae_latent", f"AE+FCM on {D.name}: latent codes (PCA) coloured by FCM cluster, clean and with "
          f"{int(r * 100)}% {nt} noise (crosses: corrupted samples), and the training loss. Corrupted samples "
          "dominate the MSE loss, so the noisy model converges to a higher loss.")


# ----------------------------------------------------------------------------- ablation & aggregates
def fig_ablation_m(o, root):
    from .report import TWO_COL
    p = os.path.join(root, "ablation_raw.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    if "method" not in df:
        return
    pf = df[(df.method == "PFCM") & (df.eta == 2.0) & (df.a == 1.0) & (df.b == 1.0)]
    fc = df[df.method == "FCM"]
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 2.2), sharey=True)
    for ax, sub, title in [(axes[0], fc, "FCM"), (axes[1], pf, "PFCM ($\\eta$ = 2, $a$ = $b$ = 1)")]:
        g = sub.groupby(["dataset", "m"]).ari.mean().unstack("m")
        for i, (ds, row) in enumerate(g.iterrows()):
            ax.plot(row.index, row.values, "-o", color=CLUSTER_COLORS[i], ms=3, label=ds)
        ax.axvline(2.0, color="gray", ls=":", lw=0.8)
        ax.set_title(title); ax.set_xlabel("fuzzifier $m$")
    axes[0].set_ylabel("mean ARI ($r$ = 0.10, all noise types)")
    axes[1].legend(fontsize=6, ncol=2)
    o.fig(fig, "fig_ablation_m", "Sensitivity to the fuzzifier $m$ (dotted line: the canonical $m$ = 2), averaged over "
          "noise types and seeds at $r$ = 0.10. High-dimensional data (digits) need $m$ close to 1.")
    if len(fc):
        t = fc.groupby(["dataset", "m"]).ari.mean().unstack("m")
        t.columns = [f"$m$={c:g}" for c in t.columns]
        t.index = [i.replace("_", "\\_") for i in t.index]; t.index.name = "Dataset"
        o.table(t, "tab_fcm_m", "FCM mean ARI by fuzzifier $m$ ($r$ = 0.10, all noise types)", "tab:fcm-m")


def fig_ablation_c(o, root):
    from .report import TWO_COL, COLORS, MARKERS
    p = os.path.join(root, "ablation_c_raw.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    df["offset"] = df.c - df.true_c
    noises = list(dict.fromkeys(df.noise))
    fig, axes = plt.subplots(1, len(noises) + 1, figsize=(TWO_COL, 2.3),
                             gridspec_kw={"width_ratios": [1] * len(noises) + [1.4]})
    for ax, nt in zip(axes, noises):
        g = df[df.noise == nt].groupby(["method", "offset"]).ari.mean()
        for m in g.index.get_level_values(0).unique():
            s = g.loc[m]
            ax.plot(s.index, s.values, "-", marker=MARKERS.get(m, "o"), color=COLORS.get(m, "k"), label=m)
        ax.axvline(0, color="gray", ls=":", lw=0.8)
        ax.set_title("clean data" if nt == "none" else f"{nt} noise", fontsize=7)
        ax.set_xlabel("$c$ $-$ true $c$")
    axes[0].set_ylabel("mean ARI"); axes[0].legend(fontsize=6)
    sel = pd.read_csv(os.path.join(root, "ablation_c_selection.csv"))
    t = sel.groupby(["rule", "noise"]).correct.mean().unstack("noise")[noises]
    x = np.arange(len(t)); w = 0.8 / len(noises)
    for i, nt in enumerate(noises):
        axes[-1].bar(x + (i - (len(noises) - 1) / 2) * w, t[nt], w, color=["#9ecae1", "#3182bd"][i % 2],
                     edgecolor="k", lw=0.3, label="clean" if nt == "none" else nt)
    axes[-1].set_xticks(x); axes[-1].set_xticklabels([r.replace(" (", "\n(") for r in t.index], fontsize=6)
    axes[-1].set_ylabel("fraction choosing true $c$"); axes[-1].set_ylim(0, 1); axes[-1].legend(fontsize=6)
    axes[-1].set_title("label-free choice of $c$", fontsize=7)
    fig.tight_layout()
    o.fig(fig, "fig_ablation_c", "Number of clusters. Left: ARI when $c$ is mis-specified (offset from the true $c$). "
          "Right: how often each label-free index (averaged over the methods it applies to) picks the true $c$.")
    tab = pd.read_csv(os.path.join(root, "ablation_c_summary.csv"))
    tab = tab.rename(columns={"none": "clean"}).set_index(["method", "rule"])
    tab.index = [f"{m}, {r}" for m, r in tab.index]; tab.index.name = "Method, index"
    o.table(tab, "tab_c_selection", "Fraction of runs in which a label-free index recovers the true number of "
            "clusters ($c \\in$ true $c \\pm 2$)", "tab:c-selection")


def fig_runtime_scaling(o, df, methods):
    from .report import TWO_COL, COLORS, MARKERS
    g = df.groupby(["dataset", "method"]).agg(n=("n", "median"), d=("d", "median"), t=("runtime_s", "median"),
                                              it=("iterations", "median")).reset_index()
    g["nd"] = g.n * g.d
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 2.2))
    for m in methods:
        s = g[g.method == m].sort_values("nd")
        axes[0].plot(s.nd, s.t, "-", marker=MARKERS[m], color=COLORS[m], label=m)
        axes[1].plot(s.nd, s.it, "-", marker=MARKERS[m], color=COLORS[m], label=m)
    for ax in axes:
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("data size $n \\times d$")
    axes[0].set_ylabel("median fit time (s)"); axes[1].set_ylabel("median iterations (best restart)")
    axes[1].legend(fontsize=6, ncol=2)
    o.fig(fig, "fig_runtime_scaling", "Cost scaling with data size (one point per dataset, medians over all conditions "
          "and seeds). AE+FCM iterations are those of FCM on the latent codes; GMM counts EM iterations.")
    t = df.assign(clean=df.noise == "none").groupby(["method", "clean"])[["fpc", "iterations"]].mean().unstack("clean")
    t.columns = [f"{'FPC' if a == 'fpc' else 'Iterations'} ({'clean' if b else 'noisy'})" for a, b in t.columns]
    t = t.loc[methods]; t.index.name = "Method"
    o.table(t, "tab_fpc_iterations", "Mean partition coefficient (FPC; 1 = crisp) and iterations to convergence, "
            "on clean and on noisy data", "tab:fpc")


def fig_ari_clean_bars(o, df, methods):
    from .report import TWO_COL, COLORS
    clean = df[df.noise == "none"]
    fig, axes = plt.subplots(2, 1, figsize=(TWO_COL, 3.6), sharex=True)
    for ax, metric, lab in [(axes[0], "ari", "ARI"), (axes[1], "nmi", "NMI")]:
        t = clean.groupby(["dataset", "method"])[metric].mean().unstack("method")[methods]
        x = np.arange(len(t)); w = 0.85 / len(methods)
        for i, m in enumerate(methods):
            bars = ax.bar(x + (i - (len(methods) - 1) / 2) * w, t[m], w, color=COLORS[m], edgecolor="k", lw=0.3, label=m)
            for b, v in zip(bars, t[m]):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=4.2, rotation=90)
        ax.set_ylabel(f"{lab}, clean data"); ax.set_ylim(0, 1.12)
        ax.set_xticks(x); ax.set_xticklabels(t.index, fontsize=7)
    axes[0].legend(ncol=len(methods), loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False, fontsize=6.5)
    fig.tight_layout()
    o.fig(fig, "fig_ari_clean_bars", "Clean-data agreement with the true classes (mean over seeds), before any noise "
          "is added.")


def build_all(o, cfg, df, methods):
    """Datasets for the illustrations come from the config's `diagnostics:` section; if it is absent,
    sensible defaults are taken from the datasets in the run."""
    params = cfg.get("method_params", {}) or {}
    names = list(dict.fromkeys(df.dataset)) if len(df) else list(cfg["datasets"])
    real = [n for n in names if not n.startswith(("S1", "S2"))] or names
    synth = [n for n in names if n.startswith("S1")] or names
    dg = cfg.get("diagnostics") or {}
    for ds in dg.get("partitions", real[:2]):
        fig_pca_partitions(o, ds, methods, params)
    fig_convergence(o, params, datasets=tuple(dg.get("convergence", real[:2])))
    fig_membership_typicality(o, params, ds=dg.get("heatmap", synth[0]))
    fig_outlier_scores(o, params, ds=dg.get("outliers", synth[0]))
    if "AE+FCM" in methods:
        fig_ae_latent(o, params, ds=dg.get("ae_latent", real[0]))
    fig_ablation_m(o, cfg["out_dir"])
    fig_ablation_c(o, cfg["out_dir"])
    if "iterations" in df:
        fig_runtime_scaling(o, df, methods)
    fig_ari_clean_bars(o, df, methods)
