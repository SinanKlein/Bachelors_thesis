"""
utils.py — pipeline utilities (merged from io / reducers / splits / metrics).

Sections, in order:
  1. IO            : config loading, run-id, npz/csv/json read & write
  2. Reducers      : feature reduction registry (mean_variance, identity, ...)
  3. Splits        : nested CV index generation
  4. Metrics       : classification (AUC/AP/Brier + cutoff sweep) and
                     regression (R2/MSE/RMSE/MAE)
"""
from __future__ import annotations
import json
import subprocess
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    r2_score, mean_squared_error, mean_absolute_error,
)


# =============================================================================
# 1. IO
# =============================================================================
def make_run_id() -> str:
    """YYYYMMDD_HHMMSS_<git_short_hash> — unique per run."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        h = "nogit"
    return f"{ts}_{h}"


def load_config(path: Path | str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def freeze_config(cfg: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def load_graph_data(npz_path: Path) -> dict:
    """Load graph_data.npz into a plain dict of numpy arrays."""
    if not npz_path.exists():
        raise FileNotFoundError(f"Could not find graph data at {npz_path}")
    raw = np.load(npz_path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def write_npz(arrays: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: Path) -> dict:
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


def set_global_seeds(seed: int) -> None:
    """
    Set seeds for pythons random, numpy, and PYTHONHASHSEED so that any
    library reading from these sources is deterministic. Library specific
    seeds (sklearn random_state, xgboost seed) are still passed per model.
    """
    import os
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def derive_fold_seed(base_seed: int, outer_fold: int, inner_fold: int) -> int:
    """
    Deterministic per fold seed derivation. inner_fold == -1 (outer-only) gets
    its own slot. Same (base_seed, outer, inner) -> same int every time.
    """
    return int(base_seed * 10_000 + outer_fold * 100 + (inner_fold + 1))


# =============================================================================
# 2. Reducers
# =============================================================================
class BaseReducer:
    name: str = "base"

    def fit(self, X: np.ndarray) -> "BaseReducer":
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def feature_stats(self) -> dict:
        return {}


class MeanVarianceReducer(BaseReducer):
    """Keep the top-`quantile` fraction of features ranked by raw variance.

    Important: this reducer no longer performs log1p or any transformation before
    ranking. The score is always var(X) on the raw input matrix. Downstream
    preprocessing may transform the kept columns afterwards, e.g. CLR.
    """
    name = "mean_variance"

    def __init__(self, quantile: float = 0.10, standardize: bool = False):
        if not 0 < quantile <= 1:
            raise ValueError("quantile must be in (0, 1]")
        self.quantile = quantile
        # Kept only for backward config compatibility. It is intentionally ignored.
        self.standardize = standardize

    def fit(self, X: np.ndarray) -> "MeanVarianceReducer":
        X = np.asarray(X, dtype=np.float64)
        self.means_ = X.mean(axis=0)
        self.vars_  = X.var(axis=0, ddof=1)
        self.score_ = self.vars_.copy()

        self.threshold_ = float(np.quantile(self.score_, 1.0 - self.quantile))
        self.selected_idx_ = np.where(self.score_ >= self.threshold_)[0]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_idx_]

    def feature_stats(self) -> dict:
        n = len(self.means_)
        selected = np.zeros(n, dtype=bool)
        selected[self.selected_idx_] = True
        return {
            "feature_id": np.arange(n),
            "mean":       self.means_,
            "var":        self.vars_,
            "score":      self.score_,
            "selected":   selected,
            "threshold":  self.threshold_,
        }


class IdentityReducer(BaseReducer):
    """No-op. Useful as a baseline."""
    name = "identity"

    def fit(self, X):
        self.means_ = X.mean(axis=0)
        self.vars_ = X.var(axis=0, ddof=1)
        self.score_ = self.vars_.copy()
        self.selected_idx_ = np.arange(X.shape[1])
        self.threshold_ = float(self.score_.min()) if X.shape[1] > 0 else 0.0
        return self

    def transform(self, X):
        return X

    def feature_stats(self):
        n = len(self.means_)
        return {
            "feature_id": np.arange(n),
            "mean":       self.means_,
            "var":        self.vars_,
            "score":      self.score_,
            "selected":   np.ones(n, dtype=bool),
            "threshold":  self.threshold_,
        }


REDUCERS = {
    "mean_variance": MeanVarianceReducer,
    "identity":      IdentityReducer,
    # add: "svd": SVDReducer, ...
}


def build_reducer(name: str, **params) -> BaseReducer:
    if name not in REDUCERS:
        raise KeyError(f"Unknown reducer '{name}'. Available: {list(REDUCERS)}")
    return REDUCERS[name](**params)


# =============================================================================
# 3. Splits — nested CV
# =============================================================================
def make_splits_dataframe(
    y: np.ndarray,
    n_outer: int = 10,
    n_inner: int = 0,
    stratify: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Create only the 10 outer train/test splits.

    The old nested-CV implementation created inner validation folds. For the
    thesis baseline this is intentionally simplified: each outer fold contains
    only role in {train, test} and inner_fold is always -1.
    """
    n = len(y)
    all_idx = np.arange(n)

    Outer = StratifiedKFold if stratify else KFold
    outer = Outer(n_splits=n_outer, shuffle=True, random_state=seed)
    outer_iter = outer.split(all_idx, y) if stratify else outer.split(all_idx)

    rows = []
    for outer_i, (train_idx, test_idx) in enumerate(outer_iter):
        for sid in train_idx:
            rows.append((outer_i, -1, int(sid), "train"))
        for sid in test_idx:
            rows.append((outer_i, -1, int(sid), "test"))

    return pd.DataFrame(rows, columns=["outer_fold", "inner_fold", "sample_id", "role"])


def clr_transform(X: np.ndarray, pseudocount: float = 0.5) -> np.ndarray:
    """Centered log-ratio transform rows after adding a pseudocount.

    For every sample/entity row x, returns log(x + pseudocount) minus its row
    mean log abundance. Input should be non-negative count-like data.
    """
    if pseudocount <= 0:
        raise ValueError("pseudocount must be positive for CLR transformation")
    X = np.asarray(X, dtype=np.float64)
    X_pc = np.clip(X, 0, None) + float(pseudocount)
    logX = np.log(X_pc)
    return (logX - logX.mean(axis=1, keepdims=True)).astype(np.float32)


# =============================================================================
# 4. Metrics
# =============================================================================
def basic_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict:
    """Binary classification: AUC, AP, Brier plus recall diagnostics."""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    out = {
        "n":     len(y_true),
        "n_pos": int(y_true.sum()),
        "n_neg": int((1 - y_true).sum()),
    }
    try:
        out["auc"] = roc_auc_score(y_true, prob)
    except ValueError:
        out["auc"] = np.nan
    try:
        out["ap"] = average_precision_score(y_true, prob)
    except ValueError:
        out["ap"] = np.nan
    try:
        out["brier"] = brier_score_loss(y_true, prob)
    except ValueError:
        out["brier"] = np.nan

    # Fixed-threshold recall/precision/F1 at 0.5. The existing cutoff_sweep.csv
    # still contains the full threshold curve; these are compact fold summaries.
    pred05 = (prob >= 0.5).astype(int)
    tp = int(((pred05 == 1) & (y_true == 1)).sum())
    fp = int(((pred05 == 1) & (y_true == 0)).sum())
    fn = int(((pred05 == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    out["precision_at_0.5"] = prec
    out["recall_at_0.5"] = rec
    out["f1_at_0.5"] = (2 * prec * rec / (prec + rec)) if (not np.isnan(prec) and not np.isnan(rec) and (prec + rec) > 0) else np.nan

    # Discovery-style recall among the top-scored pairs.
    n_pos = int(y_true.sum())
    for frac in (0.01, 0.05, 0.10):
        k = max(1, int(np.ceil(frac * len(y_true)))) if len(y_true) else 0
        if k > 0 and n_pos > 0:
            top = np.argsort(-prob)[:k]
            out[f"recall_top{int(frac*100)}pct"] = float(y_true[top].sum() / n_pos)
            out[f"precision_top{int(frac*100)}pct"] = float(y_true[top].sum() / k)
        else:
            out[f"recall_top{int(frac*100)}pct"] = np.nan
            out[f"precision_top{int(frac*100)}pct"] = np.nan
    return out


def cutoff_sweep(y_true: np.ndarray, prob: np.ndarray, grid: np.ndarray) -> pd.DataFrame:
    """One row per cutoff. Long format for ggplot."""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)

    rows = []
    for c in grid:
        pred = (prob >= c).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        rec  = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        if not np.isnan(prec) and not np.isnan(rec) and (prec + rec) > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        acc = (tp + tn) / max(tp + tn + fp + fn, 1)

        rows.append({
            "cutoff": float(c),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1,
            "tpr": rec, "fpr": fpr, "accuracy": acc,
        })
    return pd.DataFrame(rows)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Regression: R^2, MSE, RMSE, MAE plus basic target descriptives."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {
        "n":      len(y_true),
        "y_mean": float(np.mean(y_true)) if len(y_true) else np.nan,
        "y_std":  float(np.std(y_true, ddof=1)) if len(y_true) > 1 else np.nan,
    }
    if len(y_true) < 2:
        out.update({"r2": np.nan, "mse": np.nan, "rmse": np.nan, "mae": np.nan})
        return out
    try:
        out["r2"] = r2_score(y_true, y_pred)
    except ValueError:
        out["r2"] = np.nan
    try:
        mse = mean_squared_error(y_true, y_pred)
        out["mse"] = mse
        out["rmse"] = float(np.sqrt(mse))
    except ValueError:
        out["mse"] = np.nan
        out["rmse"] = np.nan
    try:
        out["mae"] = mean_absolute_error(y_true, y_pred)
    except ValueError:
        out["mae"] = np.nan
    return out
