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
# Paths. Every root can be overridden by an environment variable.
# =============================================================================
COHORT = os.environ.get("PIPELINE_COHORT", "GvHD")   # "CRC" | "GvHD" | "IBD"
DATA_ROOT = Path(os.environ.get(
    "PIPELINE_DATA_ROOT", r"C:\Sinan_Klein\LMU\lmu_thesis\datas_final\SINAN_datasets\data"))
SPLITS_ROOT = Path(os.environ.get(
    "PIPELINE_SPLITS_ROOT", r"C:\Sinan_Klein\LMU\lmu_thesis\multimodal_network\data"))
RESULTS_DIR = Path(os.environ.get(
    "PIPELINE_RESULTS_DIR", r"C:\Sinan_Klein\LMU\lmu_thesis\results_modularPipe"))

COHORT_DIR = DATA_ROOT / COHORT                                 # raw tables
GRAPH_DATA_FILE = SPLITS_ROOT / f"splits_unified_{COHORT}" / "graph_data.npz"
DATASET_NAME = f"{COHORT}_outputs"
PIPELINE_DIR = Path(__file__).resolve().parent


def run_dir(run_id: str, *sub: str) -> Path:
    """RESULTS_DIR/<DATASET_NAME>/<run_id>/<sub...>, created on access."""
    d = RESULTS_DIR.joinpath(DATASET_NAME, run_id, *sub)
    d.mkdir(parents=True, exist_ok=True)
    return d


# =============================================================================
# IO
# =============================================================================
def make_run_id() -> str:
    """YYYYMMDD_HHMMSS_<git short hash>."""
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                    stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        h = "nogit"
    return f"{datetime.now():%Y%m%d_%H%M%S}_{h}"


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
