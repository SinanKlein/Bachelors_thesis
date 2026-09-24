"""Run the whole pipeline for the cohort set in common.py.

  python run.py

Every stage is also runnable on its own with the same --config / --run-id.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

from common import (COHORT_DIR, DATASET_NAME, GRAPH_DATA_FILE, PIPELINE_DIR, RESULTS_DIR,
                    make_run_id)

CONFIG = PIPELINE_DIR / "config.yaml"
# Rscript from the RSCRIPT environment variable, else from PATH.
R_EXECUTABLE = os.environ.get("RSCRIPT") or shutil.which("Rscript")

# Stage toggles.
REBUILD_GRAPH_DATA = True
RUN_W_THRESHOLD = True
RUN_SHUFFLE = True
RUN_DESCRIBE = True
RUN_STABILITY = True
RUN_CKA = True
RUN_R_PLOTS = True


def run(label: str, cmd: list, required: bool = True, cwd=None) -> bool:
    print(f"\n{'=' * 70}\n  {label}\n{'=' * 70}\n$ {' '.join(map(str, cmd))}", flush=True)
    ok = subprocess.run([str(c) for c in cmd], cwd=cwd).returncode == 0
    if not ok and required:
        raise SystemExit(f"{label} failed")
    if not ok:
        print(f"[warn] {label} failed; continuing.")
    return ok


def main() -> None:
    run_id, py, t0 = make_run_id(), sys.executable, time.time()
    args = ["--config", CONFIG, "--run-id", run_id]
    print(f"dataset = {DATASET_NAME} | run_id = {run_id}")
    print(f"inputs  = {COHORT_DIR}\ngraph   = {GRAPH_DATA_FILE}\noutputs = {RESULTS_DIR}")

    if REBUILD_GRAPH_DATA or not GRAPH_DATA_FILE.exists():
        run("build graph data", [py, PIPELINE_DIR / "data.py", "build"])
    run("preprocess", [py, PIPELINE_DIR / "data.py", "preprocess", *args])
    run("experiment + latent export", [py, PIPELINE_DIR / "experiment.py", *args])
    if RUN_CKA:
        run("cka", [py, PIPELINE_DIR / "analyses.py", "cka", *args], required=False)
    for flag, stage in ((RUN_W_THRESHOLD, "w_threshold"), (RUN_SHUFFLE, "shuffle")):
        if flag:
            run(stage, [py, PIPELINE_DIR / "analyses.py", stage, *args], required=False)
    if RUN_DESCRIBE:
        run("describe", [py, PIPELINE_DIR / "data.py", "describe", *args], required=False)
    if RUN_STABILITY:
        run("stability", [py, PIPELINE_DIR / "analyses.py", "stability", *args], required=False)

    if RUN_R_PLOTS and not R_EXECUTABLE:
        print("[warn] Rscript not on PATH (set RSCRIPT to its full path); skipping plots.")
    elif RUN_R_PLOTS:
        for script in ("data.R", "results.R", "analyses.R"):
            run(f"plots/{script}", [R_EXECUTABLE, PIPELINE_DIR / "plots" / script, run_id, DATASET_NAME],
                required=False, cwd=PIPELINE_DIR / "plots")

    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min -> {RESULTS_DIR / DATASET_NAME / run_id}")


if __name__ == "__main__":
    main()
