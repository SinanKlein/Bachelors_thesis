"""
One-click pipeline runner.

Hit the run button in VS Code and it runs:
  1. preprocess    — load data, filter features, build CV splits
  2. experiment    — main nested-CV experiment AND W-threshold sweep AND
                     lasso stability selection AND XGBoost hyperparameter grid
                     (all in one script; see experiment.py for the four blocks)
  3. plot_data.R   — data-side plots
  4. plot_results.R— results-side plots (incl. slide-21, lasso, xgb-grid)
"""
from __future__ import annotations
import sys
import subprocess
from pathlib import Path

# Make sure this script's folder is on sys.path so we can import paths/utils
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from paths import RESULTS_DIR, R_DIR, GRAPH_DATA_FILE
from utils import make_run_id

# ---------------------------------------------------------------------------
# CONFIGURE
# ---------------------------------------------------------------------------
# Path to the YAML config. Sits next to this script in the flat layout.
CONFIG_PATH = HERE / "default.yaml"

# Path to Rscript on this machine.
R_EXECUTABLE = r"C:/PROGRA~1/R/R-45~1.1/bin/x64/Rscript.exe"

# Set to False to skip the R plotting step
RUN_R_PLOTS = True

# Step 0: rebuild graph_data.npz from the raw X/Y/W data each run.
# True  = always rebuild (safe; the build takes <1 min vs the long experiment).
# False = only build if graph_data.npz is missing.
REBUILD_GRAPH_DATA = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def section(msg: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {msg}")
    print("=" * 70)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a subprocess and stream its output. Raises if it fails."""
    print(f"$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit code {result.returncode}): {cmd}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config not found at {CONFIG_PATH}. "
            f"Edit CONFIG_PATH at the top of run_all.py."
        )

    run_id = make_run_id()
    py = sys.executable  # use the same Python thats running this script

    section(f"PIPELINE START | run_id = {run_id}")

    # 0. build graph_data.npz from raw X/Y/W (data builder)
    if REBUILD_GRAPH_DATA or not GRAPH_DATA_FILE.exists():
        section("STEP 1/5: build graph_data (create_unified_splits_v2.py)")
        run([py, str(HERE / "create_unified_splits_v2.py")])
    else:
        section("STEP 1/5: build graph_data — SKIPPED (already exists)")
        print(f"[info] using existing {GRAPH_DATA_FILE}")

    # 1. preprocess
    section("STEP 2/5: preprocess")
    run([py, str(HERE / "preprocess.py"),
         "--config", str(CONFIG_PATH),
         "--run-id", run_id])

    # 2. experiment (main + W-sweep + lasso + xgb_grid, all inside one script)
    section("STEP 3/5: experiment")
    run([py, str(HERE / "experiment.py"),
         "--config", str(CONFIG_PATH),
         "--run-id", run_id])

    # 3 + 4. R plots
    ran_r = False
    if RUN_R_PLOTS:
        section("STEP 4/5: plot_data.R")
        try:
            run([R_EXECUTABLE, str(R_DIR / "plot_data.R"), run_id], cwd=R_DIR)
            ran_r = True
        except FileNotFoundError:
            print(f"[warn] '{R_EXECUTABLE}' not found. Skipping R plots. "
                  f"Set R_EXECUTABLE in run_all.py to fix.")

        if ran_r:
            section("STEP 5/5: plot_results.R")
            run([R_EXECUTABLE, str(R_DIR / "plot_results.R"), run_id], cwd=R_DIR)
    else:
        print("[info] R plots skipped (RUN_R_PLOTS=False).")

    # Summary
    out_dir = RESULTS_DIR / run_id
    section("DONE")
    print(f"run_id      : {run_id}")
    print(f"all outputs : {out_dir}")
    print(f"  config    : {out_dir / 'config.yaml'}")
    print(f"  preproc   : {out_dir / 'preprocessing'}")
    print(f"  metrics   : {out_dir / 'metrics'}")
    print(f"  predict   : {out_dir / 'predictions'}")
    print(f"  plots     : {out_dir / 'plots'}")


if __name__ == "__main__":
    main()
