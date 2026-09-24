"""Shared helpers: paths, IO, seeds, splits, metrics, run loading."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.stdout.reconfigure(line_buffering=True)   # live progress, also when piped

# =============================================================================
# Paths. Defaults are relative to this folder, so a fresh clone works as is:
#
#   <pipeline>/inputs/<cohort>/...                      raw tables (see README)
#   <pipeline>/inputs/<cohort>/graph_data.npz           written by `data.py build`
#   <pipeline>/outputs/<cohort>_outputs/<run_id>/...    every stage's results
#
# Each root can be moved with an environment variable (absolute, or relative to
# the current working directory):
#   PIPELINE_DATA_ROOT    folder holding one subfolder per cohort   (default inputs/)
#   PIPELINE_GRAPH_ROOT   folder holding <cohort>/graph_data.npz    (default = DATA_ROOT)
#   PIPELINE_RESULTS_DIR  folder receiving <cohort>_outputs/        (default outputs/)
#   PIPELINE_COHORT       CRC | GvHD | IBD                          (default GvHD)
# =============================================================================
PIPELINE_DIR = Path(__file__).resolve().parent


def _root(env: str, default: Path) -> Path:
    value = os.environ.get(env)
    return Path(value).expanduser().resolve() if value else default


COHORT = os.environ.get("PIPELINE_COHORT", "GvHD")   # "CRC" | "GvHD" | "IBD"
DATA_ROOT = _root("PIPELINE_DATA_ROOT", PIPELINE_DIR / "inputs")
GRAPH_ROOT = _root("PIPELINE_GRAPH_ROOT", DATA_ROOT)
RESULTS_DIR = _root("PIPELINE_RESULTS_DIR", PIPELINE_DIR / "outputs")

COHORT_DIR = DATA_ROOT / COHORT                                 # raw tables
GRAPH_DATA_FILE = GRAPH_ROOT / COHORT / "graph_data.npz"
DATASET_NAME = f"{COHORT}_outputs"


def run_dir(run_id: str, *sub: str) -> Path:
    """RESULTS_DIR/<DATASET_NAME>/<run_id>/<sub...>, created on access."""
    d = RESULTS_DIR.joinpath(DATASET_NAME, run_id, *sub)
    d.mkdir(parents=True, exist_ok=True)
    return d


# =============================================================================
# IO
# =============================================================================
def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=PIPELINE_DIR,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def make_run_id() -> str:
    """YYYYMMDD_HHMMSS_<git short hash>."""
    return f"{datetime.now():%Y%m%d_%H%M%S}_{_git('rev-parse', '--short', 'HEAD') or 'nogit'}"


# =============================================================================
# Provenance: which code, config, inputs and environment produced a run.
# Written to <run>/provenance/; does not affect any result.
# =============================================================================
def _sha256(path: Path) -> str | None:
    import hashlib
    if not Path(path).is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def record_stage(run_id: str, stage: str, config_path=None) -> None:
    """Append one line per executed stage (time, commit, config hash) and keep a copy of
    the config each stage actually used, so reruns into the same run folder stay traceable.
    The first call of a run also writes the Python / R environment."""
    import platform
    import shutil
    prov = run_dir(run_id, "provenance")
    cfg_hash = _sha256(config_path) if config_path else None
    if config_path and Path(config_path).is_file():
        shutil.copyfile(config_path, prov / f"config_{stage}.yaml")
    status = _git("status", "--porcelain")
    rec = {"time": datetime.now().isoformat(timespec="seconds"), "stage": stage,
           "cohort": COHORT, "git_commit": _git("rev-parse", "HEAD"),
           "git_dirty": bool(status) if status is not None else None,
           "config_sha256": cfg_hash, "graph_data_sha256": _sha256(GRAPH_DATA_FILE),
           "python": sys.version.split()[0], "platform": platform.platform(),
           "argv": sys.argv}
    with open(prov / "stages.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")

    freeze = prov / "pip_freeze.txt"
    if not freeze.exists():
        try:
            freeze.write_text(subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL).decode())
        except Exception as e:
            freeze.write_text(f"pip freeze failed: {e}\n")
    r_info = prov / "R_sessionInfo.txt"
    rscript = os.environ.get("RSCRIPT") or shutil.which("Rscript")
    if not r_info.exists() and rscript:
        pkgs = "c('ggplot2','dplyr','tidyr','readr','tibble','jsonlite','scales','patchwork')"
        expr = (f"for (p in {pkgs}) suppressWarnings(suppressMessages("
                f"require(p, character.only = TRUE, quietly = TRUE))); print(sessionInfo())")
        try:
            r_info.write_text(subprocess.check_output([rscript, "-e", expr],
                                                      stderr=subprocess.STDOUT).decode())
        except Exception as e:
            r_info.write_text(f"sessionInfo failed: {e}\n")


def load_config(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_npz(path) -> dict:
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def stage_args():
    """Parser with the --config / --run-id pair every stage takes."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=PIPELINE_DIR / "config.yaml")
    ap.add_argument("--run-id", required=True)
    return ap


# =============================================================================
# Seeds and splits
# =============================================================================
def set_global_seeds(seed: int) -> None:
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def fold_seed(base_seed: int, fold: int) -> int:
    """Model seed for one outer fold."""
    return int(base_seed * 10_000 + fold * 100)


def make_splits(y: np.ndarray, n_folds: int, seed: int) -> pd.DataFrame:
    """Stratified outer folds, one row per (fold, sample, role)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(np.arange(len(y)), y)):
        rows += [(fold, int(s), "train") for s in tr]
        rows += [(fold, int(s), "test") for s in te]
    return pd.DataFrame(rows, columns=["outer_fold", "sample_id", "role"])


def iter_folds(splits: pd.DataFrame):
    """Yield (fold, train_ids, test_ids)."""
    for fold in sorted(splits["outer_fold"].unique()):
        s = splits[splits["outer_fold"] == fold]
        yield (int(fold), s.loc[s["role"] == "train", "sample_id"].to_numpy(),
               s.loc[s["role"] == "test", "sample_id"].to_numpy())


# =============================================================================
# Metrics
# =============================================================================
def binary_metrics(y_true, prob) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    out = {"n": len(y_true), "n_pos": int(y_true.sum()), "n_neg": int((1 - y_true).sum())}
    for name, fn in (("auc", roc_auc_score), ("ap", average_precision_score)):
        try:
            out[name] = fn(y_true, prob)
        except ValueError:
            out[name] = np.nan
    return out


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {"n": len(y_true)}
    try:
        out["r2"] = r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan
    except ValueError:
        out["r2"] = np.nan
    return out


def cutoff_sweep(y_true, prob, grid) -> pd.DataFrame:
    """Confusion counts and derived rates at every cutoff."""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    rows = []
    for c in grid:
        pred = (prob >= c).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        f1 = (2 * prec * rec / (prec + rec)
              if not np.isnan(prec) and not np.isnan(rec) and prec + rec > 0 else np.nan)
        rows.append({"cutoff": float(c), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                     "precision": prec, "recall": rec, "f1": f1, "tpr": rec,
                     "fpr": fp / (fp + tn) if fp + tn else np.nan,
                     "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1)})
    return pd.DataFrame(rows)


# =============================================================================
# Run loading (shared by experiment.py and analyses.py)
# =============================================================================
def load_run(run_id: str, cfg: dict):
    """Pair features, per-task (config, labels, mask) and the fold table.

    X[pair] = Xb_filtered[bacterium] + Xv_filtered[virus].
    """
    pre = run_dir(run_id, "preprocessing")
    data = load_npz(pre / "data_filtered.npz")
    bi = np.asarray(data["bact_idx"]).ravel().astype(int)
    vi = np.asarray(data["virus_idx"]).ravel().astype(int)
    X = (np.asarray(data["Xb_filtered"], dtype=np.float32)[bi]
         + np.asarray(data["Xv_filtered"], dtype=np.float32)[vi])

    labels = pd.read_csv(pre / "labels.csv")
    tasks = {}
    for t in cfg["tasks"]:
        sub = labels[labels["task"] == t["name"]].sort_values("sample_id")
        dtype = int if t["task_type"] == "binary" else float
        tasks[t["name"]] = (t, sub["y"].to_numpy().astype(dtype),
                            sub["mask"].to_numpy().astype(int))
    splits = pd.read_csv(pre / "splits.csv")
    return data, X, tasks, splits
