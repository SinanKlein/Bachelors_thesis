"""
One-click pipeline runner.

Each stage is one script with one job, run in order:

  1. build_graph_data   graph_data.npz from the raw X / Y / W tables
  2. preprocess         feature selection + transform, splits, labels
  3. experiment         the 10-split train/test experiment
  4. analyses           W-threshold sweep
  5. ablation           4-arm representation ladder + family/genus
  6. shuffle_probe      shuffled-Y control for the joint model
  7. latent             out-of-fold latent export
  8. graph_export       adjacency CSVs for the R interaction plots
  9. descriptives       the numbers the Data chapter reports
 10. R plots            data, results, analyses

Toggle any stage with the flags below. Every stage is independently runnable
with the same --config / --run-id arguments, so a failed step can be repeated
on its own without redoing the run.
"""
from __future__ import annotations
import sys
import time
import functools
import subprocess
from pathlib import Path

# Print everything immediately (flush) so progress is visible live.
print = functools.partial(print, flush=True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from paths import RESULTS_DIR, R_DIR, GRAPH_DATA_FILE, DATASET_NAME
from utils import make_run_id

# ---------------------------------------------------------------------------
# CONFIGURE
# ---------------------------------------------------------------------------
CONFIG_PATH = HERE / "default.yaml"

# Path to Rscript on this machine. A wrong path skips the R plots with a warning.
R_EXECUTABLE = r"C:/PROGRA~1/R/R-45~1.1/bin/x64/Rscript.exe"

# Stage toggles. Set any to False for a faster loop.
REBUILD_GRAPH_DATA = True    # rebuild graph_data.npz (fast; safe to leave on)
RUN_ANALYSES       = True    # analyses.py   W-threshold sweep
RUN_ABLATION       = True    # ablation.py   4 arms + family/genus
RUN_SHUFFLE_PROBE  = False   # shuffle_probe.py  OFF: ~60 extra model fits (hours on GvHD)
RUN_LATENT         = True    # latent.py     latent export (needs torch)
RUN_GRAPH_EXPORT   = True    # graph_export.py  adjacency CSVs for the R plots
RUN_DESCRIPTIVES   = True    # descriptives.py  the Data chapter's numbers
RUN_R_PLOTS        = True    # every plot_*.R


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


def try_run(label: str, cmd: list[str], cwd: Path | None = None) -> bool:
    """Run a non-critical stage. A failure is reported and the run continues."""
    try:
        run(cmd, cwd=cwd)
        return True
    except Exception as e:
        print(f"[warn] {label} failed ({e}); continuing.")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config not found at {CONFIG_PATH}. "
            f"Edit CONFIG_PATH at the top of run_all.py.")

    run_id = make_run_id()
    py = sys.executable          # the same interpreter that is running this file
    cfg_args = ["--config", str(CONFIG_PATH), "--run-id", run_id]

    t_start = time.time()
    section(f"PIPELINE START | dataset = {DATASET_NAME} | run_id = {run_id}")

    # -- 1. data ------------------------------------------------------------
    if REBUILD_GRAPH_DATA or not GRAPH_DATA_FILE.exists():
        section("1/10  build_graph_data.py")
        run([py, str(HERE / "build_graph_data.py")])
    else:
        section("1/10  build_graph_data.py — SKIPPED (already exists)")
        print(f"[info] using existing {GRAPH_DATA_FILE}")

    # -- 2 + 3. the experiment ----------------------------------------------
    section("2/10  preprocess.py")
    run([py, str(HERE / "preprocess.py")] + cfg_args)

    section("3/10  experiment.py")
    run([py, str(HERE / "experiment.py")] + cfg_args)

    # -- 4 + 5. analyses of the experiment ----------------------------------
    if RUN_ANALYSES:
        section("4/10  analyses.py  (W-threshold sweep)")
        try_run("analyses", [py, str(HERE / "analyses.py")] + cfg_args)

    if RUN_ABLATION:
        section("5/10  ablation.py  (4 arms + family/genus)")
        try_run("ablation", [py, str(HERE / "ablation.py")] + cfg_args)

    # -- 6. shuffled-Y control ----------------------------------------------
    # Off by default: n_folds x (1 + n_shuffles) full fits on top of everything
    # else. Reads this run's own splits and features, so it can equally be run
    # on its own afterwards with the same --run-id.
    if RUN_SHUFFLE_PROBE:
        section("6/10  shuffle_probe.py  (shuffled-Y control)")
        try_run("shuffle probe", [py, str(HERE / "shuffle_probe.py")] + cfg_args)
    else:
        print("[info] shuffle probe skipped (RUN_SHUFFLE_PROBE=False). Run it later with:")
        print(f'         "{sys.executable}" shuffle_probe.py --config "{CONFIG_PATH}" --run-id {run_id}')

    # -- 7. latent spaces ----------------------------------------------------
    if RUN_LATENT:
        section("7/10  latent.py  (out-of-fold latent export)")
        run([py, str(HERE / "latent.py")] + cfg_args)
    else:
        print("[info] latent stage skipped (RUN_LATENT=False).")

    # -- 8. graph views ------------------------------------------------------
    if RUN_GRAPH_EXPORT:
        section("8/10  graph_export.py  (adjacency CSVs)")
        try_run("graph export", [py, str(HERE / "graph_export.py")] + cfg_args)

    # -- 9. descriptive statistics -------------------------------------------
    if RUN_DESCRIPTIVES:
        section("9/10  descriptives.py  (the Data chapter's numbers)")
        try_run("descriptives", [py, str(HERE / "descriptives.py")] + cfg_args)

    # -- 10. plots -----------------------------------------------------------
    if RUN_R_PLOTS:
        section("10/10  R plots")
        # Three scripts, one per stage of the pipeline they visualise.
        r_scripts = [
            ("plot_data.R",     "features, labels, interaction matrices, nestedness"),
            ("plot_results.R",  "prediction performance"),
            ("plot_analyses.R", "ablation + W threshold"),
        ]
        if not Path(R_EXECUTABLE).exists():
            print(f"[warn] Rscript not found at '{R_EXECUTABLE}'. Set R_EXECUTABLE "
                  f"at the top of run_all.py. Skipping every R plot.")
        else:
            # Each script is independent, so one failure does not stop the rest:
            # a broken figure should never cost you the other three.
            failed = []
            for script, what in r_scripts:
                print(f"\n--- {script}  ({what})")
                if not try_run(script, [R_EXECUTABLE, str(R_DIR / script),
                                        run_id, DATASET_NAME], cwd=R_DIR):
                    failed.append(script)
            if failed:
                print(f"\n[warn] R plots that failed: {', '.join(failed)}")
                print(f"[warn] Re-run one on its own with:")
                print(f'         "{R_EXECUTABLE}" {failed[0]} {run_id} {DATASET_NAME}')
    else:
        print("[info] R plots skipped (RUN_R_PLOTS=False).")

    # -- summary -------------------------------------------------------------
    out_dir = RESULTS_DIR / DATASET_NAME / run_id
    elapsed = time.time() - t_start
    section("DONE")
    print(f"dataset     : {DATASET_NAME}")
    print(f"run_id      : {run_id}")
    print(f"all outputs : {out_dir}")
    for label, rel in (("config", "config.yaml"), ("preproc", "preprocessing"),
                       ("metrics", "metrics"), ("predict", "predictions"),
                       ("ablation", "ablation"), ("shuffle", "shuffle_probe"),
                       ("latent", "latent"), ("plots", "plots")):
        print(f"  {label:<9}: {out_dir / rel}")
    print(f"total time  : {elapsed:.1f} s  ({elapsed/60:.2f} min)")


if __name__ == "__main__":
    main()
