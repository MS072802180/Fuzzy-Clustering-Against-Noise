"""End-to-end check: run -> ablation -> analyze -> report on a tiny config, from any working directory."""
import os
import yaml
import pytest

from fznb.config import load_config, torch_available


def _tiny_config(tmp_path, methods):
    cfg = {
        "out_dir": str(tmp_path / "out"), "seeds": 2, "jobs": 1,
        "datasets": ["S1_2d_unequal", "csv:data/example_iris.csv"],
        "noise_types": ["uniform", "heavy", "rowwise", "cellwise"], "levels": [0.0, 0.1],
        "methods": methods,
        "method_params": {"AE+FCM": {"epochs": 5}},
        "ablation": {"level": 0.1, "seeds": 1, "m": [1.5, 2.0], "eta": [2.0, 3.0], "ab": [[1.0, 1.0]],
                     "c_offsets": [-1, 0, 1], "c_methods": ["FCM", "GMM"], "c_noise": ["none", "uniform"], "c_seeds": 1},
    }
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


def test_full_pipeline(tmp_path, monkeypatch):
    from fznb import run, ablation, analyze, report
    methods = ["FCM", "PFCM", "NC", "GMM", "GMM+noise", "t-mixture", "AE+FCM"]
    monkeypatch.chdir(tmp_path)                       # prove it works outside the project folder
    cfg = load_config(_tiny_config(tmp_path, methods))
    run.run(cfg, jobs=1)
    run.run(cfg, jobs=1)                              # second call must resume and do nothing
    ablation.run(cfg, jobs=1)
    analyze.analyze(cfg, figures=True)
    out = cfg["out_dir"]
    import pandas as pd
    df = pd.read_csv(os.path.join(out, "raw_results.csv"))
    expected_methods = len(cfg["methods"])
    assert len(df) == 2 * 2 * (1 + 4) * expected_methods     # datasets x seeds x conditions x methods
    assert (df.error.fillna("") == "").all()
    for f in ["REPORT.md", "all_figures_and_tables.pdf", "figures/fig_ari_vs_noise.pdf",
              "figures/fig_ranks.pdf", "tables/tab_ari_r20.tex", "tables/tab_mechanism_pairs.csv"]:
        assert os.path.getsize(os.path.join(out, f)) > 0, f
    assert {"fpc", "iterations"} <= set(df.columns) and df.fpc.notna().all()
    figs = {f[:-4] for f in os.listdir(os.path.join(out, "figures")) if f.endswith(".pdf")}
    tabs = {f[:-4] for f in os.listdir(os.path.join(out, "tables")) if f.endswith(".tex")}
    for f in ["fig_convergence", "fig_membership_typicality", "fig_outlier_scores", "fig_ablation_m",
              "fig_ablation_c", "fig_runtime_scaling", "fig_ari_clean_bars", "fig_ae_latent"]:
        assert f in figs, f
    for t in ["tab_ari_r20", "tab_c_selection", "tab_fcm_m", "tab_fpc_iterations"]:
        assert t in tabs, t
    assert len(figs) >= 20 and len(tabs) >= 12


def test_main_entry_point(tmp_path):
    import main
    cfg_path = _tiny_config(tmp_path, ["FCM", "NC", "GMM"])
    main.main(["--config", cfg_path, "--steps", "run", "analyze", "--jobs", "1"])
    assert os.path.exists(os.path.join(tmp_path, "out", "REPORT.md"))


def test_bad_config_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"out_dir": "x", "seeds": 1, "datasets": ["nope"], "noise_types": ["uniform"],
                                 "levels": [0.0], "methods": ["FCM"]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(str(p))


def test_missing_torch_drops_ae(tmp_path, monkeypatch):
    import fznb.config as c
    monkeypatch.setattr(c, "torch_available", lambda: False)
    cfg = c.load_config(_tiny_config(tmp_path, ["FCM", "AE+FCM"]), quiet=True)
    assert cfg["methods"] == ["FCM"]
