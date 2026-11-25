"""One entry point for the whole study.  In VS Code: open this file and press F5 (or the Run button).

    python main.py                                   # quick smoke test (~1 min), configs/quick.yaml
    python main.py --config configs/full.yaml        # the full study (~40 min on 2 cores)
    python main.py --steps analyze                   # re-make statistics, figures, tables only
    python main.py --steps report                    # re-make figures and tables only (seconds)
    python main.py --fresh                           # discard earlier results for this config

Steps, in order: run (main benchmark) -> ablation (PFCM parameter grid) -> analyze (statistics,
REPORT.md) -> report (PDF/PNG figures, LaTeX/CSV tables, all_figures_and_tables.pdf).
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

STEPS = ["run", "ablation", "analyze", "report"]


def check_dependencies():
    missing = []
    for mod, pip_name in [("numpy", "numpy"), ("scipy", "scipy"), ("sklearn", "scikit-learn"),
                          ("pandas", "pandas"), ("matplotlib", "matplotlib"), ("yaml", "pyyaml"),
                          ("joblib", "joblib"), ("tabulate", "tabulate")]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pip_name)
    if missing:
        sys.exit("Missing packages: " + ", ".join(missing) +
                 "\nInstall them with:  python -m pip install -r requirements.txt")
    import pandas
    major, minor = (int(x) for x in pandas.__version__.split(".")[:2])
    if (major, minor) < (2, 1):
        sys.exit(f"pandas {pandas.__version__} is too old; run:  python -m pip install -U \"pandas>=2.1\"")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/quick.yaml", help="YAML config (default: configs/quick.yaml)")
    ap.add_argument("--steps", nargs="+", default=["all"], choices=["all"] + STEPS,
                    help="which steps to run (default: all)")
    ap.add_argument("--jobs", type=int, default=None, help="parallel workers (default: the config's 'jobs')")
    ap.add_argument("--fresh", action="store_true", help="delete this config's earlier raw results first")
    args = ap.parse_args(argv)

    check_dependencies()
    from fznb.config import load_config
    from fznb import run, ablation, analyze, report

    cfg = load_config(args.config)
    steps = STEPS if "all" in args.steps else [s for s in STEPS if s in args.steps]
    print(f"config: {cfg['_config_path']}\noutput: {cfg['out_dir']}\nsteps:  {' -> '.join(steps)}\n")
    t0 = time.time()
    if "run" in steps:
        print("=== 1/4  main benchmark ===")
        run.run(cfg, args.jobs, args.fresh)
    if "ablation" in steps:
        print("\n=== 2/4  PFCM parameter ablation ===")
        ablation.run(cfg, args.jobs)
    if "analyze" in steps:
        print("\n=== 3/4  statistics and REPORT.md ===")
        analyze.analyze(cfg, figures=False)
    if "report" in steps:
        print("\n=== 4/4  figures and tables ===")
        report.build(cfg)
    m, s = divmod(int(time.time() - t0), 60)
    print(f"\nFinished in {m} min {s} s. Open:\n  {os.path.join(cfg['out_dir'], 'REPORT.md')}\n"
          f"  {os.path.join(cfg['out_dir'], 'all_figures_and_tables.pdf')}")


if __name__ == "__main__":
    main()
