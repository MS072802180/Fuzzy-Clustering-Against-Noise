[README.md](https://github.com/user-attachments/files/35436641/README.md)
# Fuzzy-Clustering-Against-Noise # ST-RNC: Self-Tuning Robust Noise Clustering, with a Noise-Robustness Benchmark

This repository contains **ST-RNC**, a fuzzy clustering method for noisy data, and a benchmark that compares it with seven fuzzy, probabilistic and neural clustering methods under a common noise protocol.

**ST-RNC** (`fznb/methods/strnc.py`) modifies Davé's noise clustering (NC) in two ways:

1. **A χ²-scaled noise distance.** δ² = median nearest-centre squared distance × χ²₀.₉₉(d) / χ²₀.₅(d), re-estimated at every iteration. It replaces NC's δ² = λ·mean d², which needs a user-chosen λ and grows with the outliers it is meant to reject.
2. **Fuzzifier selection without labels.** m is chosen from {1.2, 1.5, 2} by a Xie–Beni index computed only on the samples the noise cluster does not claim. With m = 2, FCM-type methods can merge clusters when the data have many dimensions or many classes.

**Results on the reference run** (seven real datasets of 2,000–10,992 samples, four noise types, three contamination levels, five seeds; details in `results/full/REPORT.md`):

* Mean ARI over all noisy conditions: ST-RNC 0.48, t-mixture 0.39, AE+FCM 0.36, NC and FCM 0.31.
* ST-RNC had the best average rank (2.30; next, the t-mixture at 3.54; Nemenyi CD 1.15). Holm-corrected Wilcoxon tests favoured it over each other method.
* Most of the gain comes from the fuzzifier selection on the three ten-class digit datasets (Optdigits, Pendigits, Multiple Features), where m = 2 merges clusters. On Landsat and Image Segmentation it performs about like NC; on Pendigits, Texture and Waveform the t-mixture is better.
* All baselines use their published default settings. In the PFCM parameter grid at 10% noise, the best setting chosen with the labels reaches about the same mean ARI as ST-RNC (0.48), so tuned baselines can change the ranking.

No API keys or accounts are needed. The main datasets are bundled in `data/pmlb/` (under 1 MB) and verified by SHA-256 on every load. Only the optional extended study downloads data (MNIST, 15 MB, once).

---

## 1. Install

You need Python 3.10 or newer (tested on 3.13).

**Easiest (creates `.venv`, installs everything, runs the tests):**

```bash
git clone https://github.com/MS072802180/Fuzzy-Clustering-Against-Noise.git
cd Fuzzy-Clustering-Against-Noise
bash scripts/setup.sh          # macOS / Linux
scripts\setup.bat              # Windows
```

**By hand:**

```bash
git clone https://github.com/MS072802180/Fuzzy-Clustering-Against-Noise.git
cd Fuzzy-Clustering-Against-Noise
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build; macOS/Windows: pip install torch
pip install -r requirements.txt
python -m pytest -q tests            # 37 tests, under a minute
```

PyTorch is only needed for AE+FCM. If it is missing, the run skips AE+FCM with a warning instead of failing.

**VS Code:** open the folder, choose the `.venv` interpreter (Ctrl+Shift+P → *Python: Select Interpreter*), then open the *Run and Debug* panel (Ctrl+Shift+D). It offers launch configurations for:

* the quick run;
* the full study;
* rebuilding statistics, figures and tables only;
* the development and extended suites;
* a quick run from scratch;
* a one-worker debug run in which breakpoints work inside the methods;
* the tests.

**Google Colab:** run `!git clone https://github.com/MS072802180/Fuzzy-Clustering-Against-Noise.git && cd Fuzzy-Clustering-Against-Noise && pip install -r requirements.txt && python main.py`.

## 2. Run

`main.py` runs every step in order: the benchmark, the ablations, the statistics (`REPORT.md`), and the figures and tables.

```bash
python main.py                                   # quick smoke test (2 datasets, 2 seeds, ~3 min)
python main.py --config configs/full.yaml        # the full study
python main.py --config configs/full.yaml --steps analyze report   # rebuild statistics, figures, tables only
python main.py --config configs/full.yaml --steps report           # rebuild figures and tables only
python main.py --config configs/full.yaml --fresh                  # discard earlier results and start over
```

* `--jobs N` sets the number of parallel workers; the default is all cores.
* The benchmark resumes: if it is interrupted, run the same command again and it skips fits already saved.
* **Configs:**
  * `configs/full.yaml` (the reference results): 7 datasets × (clean + 4 noise types × 3 levels) × 5 seeds × 8 methods = **3,640 fits**, plus the ablations. About 6 h in total on 2 CPU cores (~4 h for the main grid); more cores scale it down almost linearly. The t-mixture on the 216-dimensional Multiple Features data is the slowest part.
  * `configs/small.yaml`: the **development suite** (Iris, Wine, WDBC, Digits-1,797, two small synthetic sets) on which ST-RNC's settings were fixed. It shares no data with the evaluation suite.
  * `configs/extended.yaml`: adds Letter (20,000 samples, 26 classes) and a 10,000-image MNIST subset.
  * `configs/quick.yaml`: a smoke test.

The steps also run on their own: `python -m fznb.run`, `python -m fznb.ablation`, `python -m fznb.analyze`, `python -m fznb.report`, each with `--config`.

## 3. Outputs (in `results/<config name>/`)

`python -m fznb.analyze` computes the statistics and then calls `python -m fznb.report`, which writes the figures and tables. To change only the styling, edit `fznb/report.py` and re-run `python -m fznb.report --config configs/full.yaml`; no experiment is re-run.

**Start with** `REPORT.md` (the key numbers) and `all_figures_and_tables.pdf` (every figure and table on its own page, with its caption).

### Figures (`figures/`, each as vector `.pdf` and 300-dpi `.png`)
Widths are 3.5 in ("1 col") or 7.16 in ("2 col"), for two-column layouts. Each method has its own marker and line style, so the figures also read in grayscale.

**Aggregate results** (`fznb/report.py`):

| File | Size | What it shows |
|---|---|---|
| `fig_noise_protocol` | 2 col | The four noise types, illustrated on small 2-D synthetic data (not part of the evaluation) |
| `fig_partitions_example` | 2 col | Six methods' partitions of the same 2-D illustration data with 10% uniform noise, flagged outliers circled |
| `fig_ari_vs_noise` | 2 col | Mean ARI vs contamination per noise type, with 95% CI bands over seeds |
| `fig_ari_by_dataset` | 2 col, tall | ARI vs contamination for every dataset × noise type |
| `fig_robustness_drop` | 2 col | ARI lost relative to clean data |
| `fig_ari_clean_bars` | 2 col | Clean-data ARI and NMI per dataset and method |
| `fig_clean_vs_noisy` | 1 col | Clean vs noisy ARI per method |
| `fig_ranks` | 1 col | Critical-difference diagram (Friedman + Nemenyi) |
| `fig_mechanism_pairs` | 2 col | NC vs GMM+noise, PFCM vs t-mixture, FCM vs GMM: one point per condition |
| `fig_outlier_auc` | 2 col | Outlier-detection ROC-AUC vs contamination |
| `fig_runtime` | 2 col | Median fit time per dataset (log scale) |
| `fig_runtime_scaling` | 2 col | Fit time and iterations vs data size n×d |
| `fig_ablation_surface` | 1 col | PFCM mean ARI over (m, η) |
| `fig_ablation_m` | 2 col | ARI vs fuzzifier m for FCM and PFCM, per dataset |
| `fig_parameter_selection` | 2 col | Label-free PFCM parameter selection vs default and oracle |
| `fig_ablation_c` | 2 col | ARI when c is mis-specified, and how often each index recovers the true c |

**Diagnostics** (single runs with seed 0; `fznb/diagnostics.py`; datasets set under `diagnostics:` in the config):

| File | Size | What it shows |
|---|---|---|
| `fig_membership_typicality` | 2 col | FCM memberships, PFCM typicality, and NC and ST-RNC memberships with the noise column, on Landsat with 10% uniform noise |
| `fig_outlier_scores` | 2 col | Each method's outlier score on a PCA view, with injected points circled and the ROC-AUC |
| `fig_pca_partitions_<dataset>` | 2 col | Ground truth and every method's partition on a PCA view, with 10% row-wise noise |
| `fig_convergence` | 2 col | Objective, log-likelihood or AE loss per iteration, clean vs noisy |
| `fig_ae_latent` | 2 col | Autoencoder latent space (clean vs noisy) and its training loss |

### Tables (`tables/`, each as `.tex` (booktabs) and `.csv`)
| File | Contents |
|---|---|
| `tab_datasets` | n, d, c and source of each dataset |
| `tab_methods` | Paradigm, mechanism, default parameters and reference for each method |
| `tab_ari_clean` | Clean-data ARI per dataset × method (best in bold) |
| `tab_ari_r20` | Mean (sd) ARI at r = 0.20 per noise type, plus average rank |
| `tab_robustness_drop` | ARI drop per noise type × level × method |
| `tab_mechanism_pairs` | Wilcoxon tests on the mechanism pairs |
| `tab_pairwise_all` | All pairwise tests, with the Friedman/Nemenyi result as a note |
| `tab_outliers_runtime` | Outlier AUC, native outlier F1, median runtime |
| `tab_fpc_iterations` | Fuzzy partition coefficient and iterations, clean vs noisy |
| `tab_parameter_selection` | PFCM label-free selection vs default and oracle |
| `tab_fcm_m` | FCM ARI by fuzzifier m per dataset |
| `tab_c_selection` | How often FPC, Xie–Beni, BIC and silhouette recover the true number of clusters |

The `.tex` tables can be included with `\input{tables/tab_ari_r20.tex}` (they need `booktabs`); the `.png` figures are 300 dpi for Word or Google Docs, and the `.csv` files open in Excel.

### Raw and intermediate files
| File | Contents |
|---|---|
| `raw_results.csv` | One row per fit: dataset, noise, level, seed, method, paradigm, mechanism, ARI, AMI, NMI, acc, predicted clusters, outlier AUC/F1, FPC, iterations, runtime, error |
| `summary.csv` | Mean and sd of every metric per method × dataset × noise × level |
| `stats_friedman.txt`, `ranks.csv`, `stats_pairwise.csv` | Statistical tests |
| `runtime_median_s.csv` | Median fit times |
| `ablation_raw.csv`, `ablation_selection.csv`, `ablation_summary.csv`, `ablation_m_eta_surface.csv`, `ablation_fcm_m.csv` | PFCM grid and FCM m-sweep |
| `ablation_c_raw.csv`, `ablation_c_selection.csv`, `ablation_c_summary.csv` | Number-of-clusters sweep |
| `ablation_components_raw.csv`, `ablation_components_summary.csv` | NC, RNC, ST-NC and ST-RNC on identical data |
| `strnc_m_choices.csv` | Which m ST-RNC selected, per dataset |

**Reference results.** `results/full/` holds the output of the full study (5 seeds, 3,640 fits, 0 failures). A fresh run should reproduce it: the NumPy methods match exactly, and AE+FCM may differ slightly across PyTorch versions.

## 4. Data

Every dataset is z-scored and constant columns are dropped.

**Evaluation suite (`configs/full.yaml`).** Seven real datasets in their original form, obtained through PMLB (Olson et al., *BioData Mining* 2017, doi:10.1186/s13040-017-0154-4): six from UCI, and Texture from the ELENA project. Candidates had to meet criteria fixed in advance: real measurements, continuous features, at least 2,000 samples and 3 classes, and no class above 50% or below 1% of the samples. The files are bundled in `data/pmlb/` and each is checked against the SHA-256 published in PMLB's repository; a mismatch raises an error. If a file is missing, it is downloaded from `media.githubusercontent.com/media/EpistasisLab/pmlb/...` and verified the same way.

| Name in config | n × d, classes | Source |
|---|---|---|
| `waveform` | 5,000 × 21, 3 | UCI Waveform Database Generator (Version 1), doi:10.24432/C5CS3C |
| `segmentation` | 2,310 × 19, 7 | UCI Image Segmentation, doi:10.24432/C5GP4N |
| `satimage` | 6,435 × 36, 6 | UCI Statlog (Landsat Satellite), doi:10.24432/C55887 |
| `optdigits` | 5,620 × 64, 10 | UCI Optical Recognition of Handwritten Digits (full set), doi:10.24432/C50P49 |
| `pendigits` | 10,992 × 16, 10 | UCI Pen-Based Recognition of Handwritten Digits, doi:10.24432/C5MG6K |
| `texture` | 5,500 × 40, 11 | Texture, ELENA project, via KEEL (https://sci2s.ugr.es/keel/dataset.php?cod=72) |
| `mfeat_factors` | 2,000 × 216, 10 | UCI Multiple Features, profile-correlation view, doi:10.24432/C5HC70. Chosen from the four views by highest clean k-means ARI. |

**Extended (`configs/extended.yaml`):**

| Name | Size | Source |
|---|---|---|
| `letter` | 20,000 × 16, 26 classes | UCI Letter Recognition, bundled |
| `mnist10k` | 10,000 × 784 → 50 PCA components | MNIST via PMLB, downloaded once |

Every m = 2 fuzzy method collapses to 2–5 clusters on these two, so they mainly test the fuzzifier rather than noise robustness.

**Development suite (`configs/small.yaml`).** scikit-learn's bundled Iris, Wine, WDBC and Digits (the 1,797-sample subset), plus `S1_2d_unequal` and `S2_10d_blobs`.

**Synthetic sets.** `S1_large` and `S2_large` (3,500 × 2 and 5,000 × 10) are available but not part of the evaluation.

**Your own data:** `csv:path/to/file.csv` (numeric features plus a label column, last by default; example: `data/example_iris.csv`).

**Excluded candidates:**

* Shuttle and Page Blocks: one class holds 79% and 90% of the samples, which makes ARI uninformative.
* Wine Quality: ordinal quality scores, with a smallest class of 5 samples (0.1%).
* MAGIC: two heavily overlapping classes.

## 5. Noise protocol

The contamination level is *r* ∈ {0, 0.05, 0.10, 0.20}, and there are four noise types. Noise is added **after** standardization, so its scale means the same thing on every dataset.

* **uniform** (background noise, Banfield & Raftery 1993): *r/(1−r)·n* points are drawn uniformly from the bounding box of the clean data, enlarged by 50% on each side, so *r* is the fraction of the final data that is noise.
* **heavy** (heavy-tailed scatter): points are drawn from a multivariate Student-t with df = 1, using the mean and covariance of a randomly chosen cluster. A point is kept only if it lies outside that cluster's 99.9% Mahalanobis ellipsoid; its radius is capped at 10× the ellipsoid.
* **rowwise** (sample-level corruption): a fraction *r* of the samples is chosen, and Student-t noise (df = 3) scaled by 3 is added to every feature of each. No samples are added.
* **cellwise** (feature-level corruption, Zaccaria et al. 2026): a fraction *r* of the samples is chosen, and in each one ⌈10% of d⌉ cells are replaced by ±U(5, 10) standard deviations. No samples are added.

**Scoring:** clustering metrics are computed on the **original samples only**, so they measure how much the noise distorted the clustering of the real data. Outlier metrics use the injected points (uniform, heavy) or the corrupted samples (row-wise, cellwise) as positives. Level 0 is the same clean data for every noise type, so it is run once and labelled `noise = none`.

## 6. Methods

| Name | Paradigm, mechanism | Implementation | Default parameters | Outlier score / native flag |
|---|---|---|---|---|
| `FCM` | Fuzzy, baseline | `methods/fcm.py`, NumPy | m = 2, 5 k-means++ restarts | distance to nearest centre / none |
| `PFCM` | Fuzzy, relaxed memberships | `methods/pfcm.py`, Pal et al. 2005; γᵢ estimated once from an FCM run and then held fixed (Krishnapuram & Keller 1996) | m = 2, η = 2, a = b = 1, K = 1 | 1 − max typicality / none |
| `NC` | Fuzzy, explicit noise cluster | `methods/noise_clustering.py`, Davé 1991 | m = 2, λ = 0.5 (δ² = λ·mean d²) | noise membership / noise > all clusters |
| `GMM` | Probabilistic, baseline | scikit-learn `GaussianMixture` | full covariance, 5 inits | −log density / none |
| `GMM+noise` | Probabilistic, explicit noise component | `methods/gmm.py`, own EM; Banfield & Raftery 1993 | uniform component on the bounding box, initial share 0.05 | noise posterior / posterior > 0.5 |
| `t-mixture` | Probabilistic, heavy-tailed likelihood | `methods/gmm.py`, own EM; Peel & McLachlan 2000 | ν estimated per component, 3 inits | −log density / none |
| `AE+FCM` | Neural/hybrid, learned representation | `methods/ae_fcm.py`, PyTorch MLP d→128→64→z, then FCM | z = min(10, max(2, d/2)), 2,000 mini-batch updates (batch 256), Adam 1e-3 | reconstruction error / none |
| **`ST-RNC`** | **Fuzzy, proposed** | `methods/strnc.py` | q = 0.99, m ∈ {1.2, 1.5, 2} chosen by inlier Xie–Beni | noise membership / noise > all clusters |
| `RNC`, `ST-NC` | Fuzzy, ablations of ST-RNC | `methods/strnc.py` | χ² noise distance only (m = 2) / fuzzifier selection only (λ = 0.5) | as NC |

Every method returns its membership (or posterior) matrix, so the fuzzy partition coefficient is reported for all of them, along with iterations to convergence and the objective or log-likelihood history.

Defaults are the settings from each method's original publication and are **not** tuned on the evaluation datasets: λ for NC was chosen on synthetic development data, and ST-RNC's q, m-grid and selection rule on the development suite. Every parameter can be changed in the config under `method_params`.

**Mechanism pairs** (`fznb/methods/__init__.py`): methods in different paradigms that use the same robustness idea. NC ↔ GMM+noise (noise cluster vs noise component), PFCM ↔ t-mixture (typicality down-weighting vs heavy-tailed likelihood), FCM ↔ GMM (unconstrained baselines).

## 7. Statistics

For each noisy condition (dataset × noise type × level > 0), the ARI is averaged over seeds and the methods are ranked. The analysis then runs:

- a **Friedman test** across conditions, with the **Nemenyi critical difference** at α = 0.05 (Demšar 2006);
- pairwise **Wilcoxon signed-rank tests**, Holm-corrected over all pairs.

## 8. Ablations

**PFCM and FCM parameters.** PFCM is fitted on the grid m ∈ {1.1, 1.3, 1.5, 2, 3} × η ∈ {1.5, 2, 3, 4} × (a, b) ∈ {(1,1), (1,3), (3,1)}, and FCM on the same m values, at *r* = 0.10 for every dataset and noise type. Three label-free rules each pick one PFCM setting per condition: the Xie–Beni index, the silhouette of the crisp partition, and a trimmed silhouette that drops the 10% least-typical points. `ablation_summary.csv` compares each rule with the **oracle** (best ARI in the grid, which needs labels) and the **default** (m = η = 2, a = b = 1).

**Number of clusters.** FCM, PFCM, NC, GMM, GMM+noise and ST-RNC are fitted with c = true c − 2 … true c + 2 (never below 2), on clean data and with 10% uniform noise. The sweep records the ARI and how often each index (FPC and Xie–Beni for the fuzzy methods, BIC for the GMM, silhouette for all) picks the true c.

**Components of ST-RNC.** NC, RNC (χ² noise distance only), ST-NC (fuzzifier selection only) and ST-RNC are compared on exactly the main study's conditions and noise realisations.

Methods, offsets and noise types for the ablations are set under `ablation:` in the config.

## 9. Extending

* **New method:** write a class with `fit(X, c) -> FitResult` (see `methods/base.py`), add it to `REGISTRY` in `methods/__init__.py`, and list it in a config.
* **New dataset:** add a loader to `REGISTRY` in `fznb/data.py`, or use `csv:your_file.csv` in a config.
* **New noise type:** add a function to `fznb/noise.py` that returns `(X_noisy, is_injected, is_contaminated)`, and register it in `inject()`.

## 10. Reproducibility notes

* The noise seed depends only on (seed, level, dataset, noise type), so every method sees identical data.
* NumPy methods are deterministic given the seed. PyTorch on CPU is deterministic with one thread; each worker uses one thread.
* `config_used.json` is saved next to the results.

## 11. Known limitations

* Clean-data quality differs by method before any noise is added, so the analysis also reports the **robustness drop** (clean ARI − noisy ARI).
* FCM-type methods with m = 2 merge clusters in high dimension or with many classes (Winkler et al. 2011), visible in the cluster-count table of `REPORT.md`. ST-RNC's fuzzifier selection addresses this, but it triples the cost.
* ST-RNC assumes roughly compact clusters. On Texture and Waveform the mixture models are better, already on clean data.
* Baselines run with default settings; tuning them per dataset can change the ranking (see the PFCM grid).
* The number of clusters is given to every method. None of the tested validity indices recovered it reliably.
* AE+FCM is a two-stage pipeline (pre-train, then cluster). Joint training (DEC/IDEC style) is a natural extension.

## References

Banfield & Raftery, *Biometrics* 1993, doi:10.2307/2532201 · Bezdek, *Pattern Recognition with Fuzzy Objective Function Algorithms* 1981, doi:10.1007/978-1-4757-0450-1 · Davé, *Pattern Recognit. Lett.* 1991, doi:10.1016/0167-8655(91)90002-4 · Demšar, *JMLR* 7:1–30, 2006 · Hubert & Arabie, *J. Classif.* 1985, doi:10.1007/BF01908075 · Krishnapuram & Keller, *IEEE TFS* 1996, doi:10.1109/91.531779 · Olson et al., *BioData Mining* 2017, doi:10.1186/s13040-017-0154-4 · Pal, Pal, Keller & Bezdek, *IEEE TFS* 2005, doi:10.1109/TFUZZ.2004.840099 · Peel & McLachlan, *Stat. Comput.* 2000, doi:10.1023/A:1008981510081 · Vinh, Epps & Bailey, *ICML* 2009, doi:10.1145/1553374.1553511 · Winkler, Klawonn & Kruse, *IJFSA* 2011, doi:10.4018/ijfsa.2011010101 · Xie & Beni, *IEEE TPAMI* 1991, doi:10.1109/34.85677 · Zaccaria et al., *IJAR* 2026, doi:10.1016/j.ijar.2026.109698
