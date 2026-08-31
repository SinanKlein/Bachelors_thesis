"""
analyses.py — the supplementary analysis that sits on top of the main experiment.

One block, toggled by its own `enabled` flag in the config and skipped when its
output already exists (use --force to redo):

  w_threshold  For 10 interior thresholds t, fit on 1[W > t] and score test AUC
               across the outer splits. Shows whether W becomes learnable at ANY
               cut, not just the one the task uses.

This was previously a block inside experiment.py. It is not part of the
experiment; it is an analysis OF it, so it lives here and experiment.py is left
doing one job.

Usage:
  python analyses.py --config default.yaml --run-id <run_id>
  python analyses.py --config default.yaml --run-id <run_id> --force
"""
from __future__ import annotations
import argparse
import functools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

print = functools.partial(print, flush=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import (preprocessing_dir, metrics_dir)
from utils import (load_config, load_npz, write_csv, write_json,
                   set_global_seeds)
from features import pair_matrix_from_filtered


def _p(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Block 1: W-threshold AUC sweep
# ---------------------------------------------------------------------------
# For each threshold t over the continuous glasso edge-probability W, build the
# binary label 1[W > t] and fit an L2 ("ridge-style") logistic regression on the
# pair feature matrix X, scoring test AUC across the 10 outer train/test splits.
# We only need AUC here, not feature selection, so L2 + a fast solver (lbfgs)
# replaces the old L1/liblinear model: ~50-100x faster per fit. The (threshold x
# split) fits are independent and run in parallel via joblib.
def run_w_threshold_sweep(cfg: dict, run_id: str, labels_df: pd.DataFrame,
                          splits_df: pd.DataFrame, X: np.ndarray,
                          base_seed: int) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from joblib import Parallel, delayed

    wcfg = cfg.get("w_threshold", {})
    n_cuts   = int(wcfg.get("n_cuts", 10))
    w_task   = wcfg.get("w_task", "w_reg")
    solver   = wcfg.get("solver", "lbfgs")
    C        = float(wcfg.get("C", 1.0))
    max_iter = int(wcfg.get("max_iter", 1000))
    n_jobs   = int(wcfg.get("n_jobs", -1))
    run_xgb    = bool(wcfg.get("run_xgboost", True))
    xgb_params = dict(wcfg.get("xgb_params", {}) or {})

    # 10 evenly spaced *interior* thresholds in (0, 1): excludes 0 and 1, which
    # would make 1[W > t] degenerate (all-positive / all-negative) for a clipped
    # edge probability. For n_cuts=10 -> 0.0909, 0.1818, ..., 0.9091.
    thresholds = np.linspace(0.0, 1.0, n_cuts + 2)[1:-1]

    sub = labels_df[labels_df["task"] == w_task].sort_values("sample_id")
    if sub.empty:
        print(f"[w_threshold] no task '{w_task}' in labels; skipping.")
        return
    w_all    = sub["y"].to_numpy(dtype=float)
    mask_all = sub["mask"].to_numpy(dtype=int)

    folds = sorted(splits_df.loc[splits_df["inner_fold"] == -1, "outer_fold"].unique())
    # Both models are swept over the identical (fold, threshold) grid so the plot
    # compares them on exactly the same 1[W > t] targets.
    model_names = ["l2_logistic"] + (["xgboost"] if run_xgb else [])
    jobs = [(int(f), float(t), m) for m in model_names
            for t in thresholds for f in folds]

    def _one(outer_fold, t, model_name):
        sf = splits_df[(splits_df["outer_fold"] == outer_fold) &
                       (splits_df["inner_fold"] == -1)]
        tr = sf.loc[sf["role"] == "train", "sample_id"].to_numpy()
        ev = sf.loc[sf["role"] == "test",  "sample_id"].to_numpy()
        tr = tr[mask_all[tr] == 1]
        ev = ev[mask_all[ev] == 1]
        ytr = (w_all[tr] > t).astype(int)
        yev = (w_all[ev] > t).astype(int)
        rec = dict(model=model_name, threshold=float(t), outer_fold=int(outer_fold),
                   n_pos_train=int(ytr.sum()), n_pos_test=int(yev.sum()),
                   n_train=int(len(tr)), n_test=int(len(ev)), auc=np.nan)
        if len(tr) == 0 or len(ev) == 0:
            return rec
        if len(np.unique(ytr)) < 2 or len(np.unique(yev)) < 2:
            return rec  # degenerate single-class fold -> AUC undefined
        if model_name == "l2_logistic":
            sc  = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=C, penalty="l2", solver=solver,
                                     max_iter=max_iter, class_weight="balanced")
            clf.fit(sc.transform(X[tr]), ytr)
            p = clf.predict_proba(sc.transform(X[ev]))[:, 1]
        else:  # xgboost: trees need no scaling; handle imbalance via scale_pos_weight
            from xgboost import XGBClassifier
            n_pos = int(ytr.sum()); n_neg = int(len(ytr) - n_pos)
            spw = (n_neg / n_pos) if n_pos > 0 else 1.0
            clf = XGBClassifier(eval_metric="logloss", n_jobs=1, verbosity=0,
                                scale_pos_weight=spw,
                                random_state=base_seed + outer_fold, **xgb_params)
            clf.fit(X[tr], ytr)
            p = clf.predict_proba(X[ev])[:, 1]
        rec["auc"] = float(roc_auc_score(yev, p))
        return rec

    t0 = time.time()
    _p(f"[w_threshold] {len(thresholds)} thresholds x {len(folds)} splits x "
       f"{len(model_names)} models = {len(jobs)} fits "
       f"(models={model_names}, n_jobs={n_jobs})")
    rows = Parallel(n_jobs=n_jobs)(delayed(_one)(f, t, m) for (f, t, m) in jobs)
    per_split = pd.DataFrame(rows).sort_values(["model", "threshold", "outer_fold"])

    agg = (per_split.groupby(["model", "threshold"])["auc"]
           .agg(auc_mean="mean", auc_sd="std", n_splits="count")
           .reset_index())

    met_dir = metrics_dir(run_id)
    write_csv(per_split, met_dir / "w_threshold_sweep.csv")
    write_csv(agg,       met_dir / "w_threshold_sweep_summary.csv")
    write_json({
        "run_id": run_id, "w_task": w_task, "n_cuts": n_cuts,
        "thresholds": [float(t) for t in thresholds],
        "models": model_names, "solver": solver, "C": C, "max_iter": max_iter,
        "xgb_params": xgb_params, "n_jobs": n_jobs,
        "elapsed_sec": round(time.time() - t0, 2),
    }, met_dir / "w_threshold_log.json")
    print(f"[w_threshold] DONE in {time.time() - t0:.1f}s -> "
          f"w_threshold_sweep.csv ({len(per_split)} rows), summary "
          f"({len(agg)} model x threshold rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
BLOCKS = ("w_threshold",)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--only", default=None,
                    help=f"Comma list of blocks to run. Default: all of {BLOCKS}.")
    ap.add_argument("--force", action="store_true",
                    help="Recompute a block even if its output is already saved.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    wanted = ([b.strip() for b in args.only.split(",") if b.strip()]
              if args.only else list(BLOCKS))
    unknown = [b for b in wanted if b not in BLOCKS]
    if unknown:
        raise SystemExit(f"[analyses] unknown block(s) {unknown}; choose from {BLOCKS}")

    base_seed = int(cfg.get("resampling", {}).get("seed", 42))
    set_global_seeds(base_seed)
    met_dir = metrics_dir(args.run_id)

    def saved(fname: str) -> bool:
        return (not args.force) and (met_dir / fname).exists()

    # The W-threshold block scores the SAME pair matrix the experiment used,
    # rebuilt from data_filtered.npz (a pure indexing step).
    need_X = "w_threshold" in wanted
    if need_X:
        pre_dir = preprocessing_dir(args.run_id)
        data = load_npz(pre_dir / "data_filtered.npz")
        labels_df = pd.read_csv(pre_dir / "labels.csv")
        splits_df = pd.read_csv(pre_dir / "splits.csv")
        combine = str(cfg.get("preprocess", {}).get("combine", "concat"))
        X = pair_matrix_from_filtered(data, combine=combine)
        print(f"[analyses] feature matrix: {X.shape}  (combine={combine})")

    if "w_threshold" in wanted:
        if not bool(cfg.get("w_threshold", {}).get("enabled", True)):
            print("\n[analyses] ### W-THRESHOLD SWEEP: SKIPPED (disabled) ###")
        elif saved("w_threshold_sweep.csv"):
            print("\n[analyses] ### W-THRESHOLD SWEEP: SKIPPED (already saved) ###")
        else:
            print("\n[analyses] ### W-THRESHOLD SWEEP ###")
            run_w_threshold_sweep(cfg, args.run_id, labels_df, splits_df, X, base_seed)


if __name__ == "__main__":
    main()
