"""
Stage 2: experiment.

Reads stage 1 outputs:
  preprocessing/data_filtered.npz
  preprocessing/labels.csv
  preprocessing/splits.csv

Runs four blocks in order, all in this single file:

  (1) MAIN NESTED-CV EXPERIMENT
      For each (outer fold, inner fold, task, model) where
      model.task_type == task.task_type: fit on train, predict on eval.
      Writes:
        predictions/predictions.csv     long format, one row per sample
        metrics/metrics_by_fold.csv     one row per (outer, inner, task, model)
        metrics/cutoff_sweep.csv        binary tasks only

  (2) W-THRESHOLD SWEEP                 
      Three panels (W>0, W+, W-), one binary model fit per (panel, tau, fold).
      Writes: metrics/w_threshold_sweep.csv

  (3) LASSO STABILITY SELECTION
      L1 logistic regression on Y for a log spaced lambda grid; selection
      frequency per (lambda, feature) across the 10 outer folds.
      Writes: metrics/lasso_selection.csv      (long, per nonzero coef)
              metrics/lasso_path_summary.csv   (per (lambda, feature))

  (4) XGBOOST HYPERPARAMETER GRID
      3x3 grid of (max_depth, learning_rate) for xgb_clf on Y. AUC per cell.
      Writes: metrics/xgb_grid.csv             (one row per cell x fold)
              metrics/xgb_grid_summary.csv     (mean/sd per cell)

Each block can be skipped via CLI flags:
  --skip-w-sweep / --skip-lasso / --skip-xgb-grid

Predictions schema (block 1):
  outer_fold, inner_fold, role, task, task_type, model, sample_id,
  y_true, prob, y_pred
    - prob is set for binary tasks (classifier output), NaN for regression
    - y_pred is set for regression (continuous prediction), NaN for binary

Usage:
  python experiment.py --config ../config/default.yaml --run-id <id_from_preprocess>
"""
from __future__ import annotations
import argparse
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import preprocessing_dir, predictions_dir, metrics_dir
from utils import (
    load_config, load_npz, write_csv, write_json,
    basic_metrics, cutoff_sweep, regression_metrics,
    set_global_seeds, derive_fold_seed,
)
from models import build_model

from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler

# Silence the sklearn 1.8 deprecation warnings 
warnings.filterwarnings(
    "ignore", category=FutureWarning,
    message=".*penalty.*was deprecated.*",
)
warnings.filterwarnings(
    "ignore", category=UserWarning,
    message=".*Inconsistent values: penalty.*",
)


# ---------------------------------------------------------------------------
# Progress / IO helpers
# ---------------------------------------------------------------------------
def _p(msg: str) -> None:
    """Print immediately (flush) so progress is visible during long runs,
    even when output is piped to a file (e.g. `python experiment.py ... | tee log.txt`)."""
    print(msg, flush=True)


def _write_or_append(new_df: pd.DataFrame, path, models_run, append: bool) -> None:
    """
    Write new_df to path. If append=True and path already exists, merge:
    drop any existing rows whose 'model' is in models_run (so re-running a
    model overwrites just its own rows) and keep all other models' rows.
    This is what lets you add a model later without re-running the baselines.
    """
    path = Path(path)
    if append and path.exists():
        old = pd.read_csv(path)
        if "model" in old.columns and models_run is not None:
            old = old[~old["model"].isin(list(models_run))]
        new_df = pd.concat([old, new_df], ignore_index=True)
        _p(f"[append] merged into existing {path.name} "
           f"(kept {len(old)} prior rows, added {len(new_df) - len(old)})")
    write_csv(new_df, path)


# ---------------------------------------------------------------------------
# Feature builder (used by every block)
# ---------------------------------------------------------------------------
def build_feature_matrix(data: dict) -> np.ndarray:
    """Build the per-sample feature matrix."""
    Xb = np.asarray(data["Xb_filtered"], dtype=np.float32)
    Xv = np.asarray(data["Xv_filtered"], dtype=np.float32)

    if "bact_idx" in data and "virus_idx" in data:
        bact_idx  = np.asarray(data["bact_idx"]).ravel().astype(int)
        virus_idx = np.asarray(data["virus_idx"]).ravel().astype(int)
        return np.concatenate([Xb[bact_idx], Xv[virus_idx]], axis=1)

    if Xb.shape[0] == Xv.shape[0]:
        return np.concatenate([Xb, Xv], axis=1)

    raise RuntimeError(
        f"Xb ({Xb.shape}) / Xv ({Xv.shape}) row counts differ."
    )


def build_feature_labels(data: dict) -> pd.DataFrame:
    """
    One row per column of build_feature_matrix(data), in the same column
    order. Columns: feature_id, feature_source ('Xb' or 'Xv'), feature_name.

    The feature matrix is concat([Xb_filtered, Xv_filtered], axis=1).
    Xb_filtered has shape (n_pairs, n_kept_bact_procs); the original
    bacterial ProC names live in bact_procs, indexed by Xb_selected_idx.
    """
    if "Xb_selected_idx" not in data or "Xv_selected_idx" not in data:
        raise KeyError("data_filtered.npz must contain Xb_selected_idx and "
                       "Xv_selected_idx (saved by preprocess.py).")

    bact_idx_kept  = np.asarray(data["Xb_selected_idx"]).ravel().astype(int)
    virus_idx_kept = np.asarray(data["Xv_selected_idx"]).ravel().astype(int)

    if "bact_procs" in data:
        bact_procs = np.asarray(data["bact_procs"]).ravel()
        bact_names = bact_procs[bact_idx_kept].astype(str)
    else:
        bact_names = np.array([f"bact_proc_{i}" for i in bact_idx_kept])

    if "virus_procs" in data:
        virus_procs = np.asarray(data["virus_procs"]).ravel()
        virus_names = virus_procs[virus_idx_kept].astype(str)
    else:
        virus_names = np.array([f"virus_proc_{i}" for i in virus_idx_kept])

    n_b = len(bact_names)
    n_v = len(virus_names)
    return pd.DataFrame({
        "feature_id":     np.arange(n_b + n_v),
        "feature_source": np.array(["Xb"] * n_b + ["Xv"] * n_v),
        "feature_name":   np.concatenate([bact_names, virus_names]),
    })


# ---------------------------------------------------------------------------
# Per-fold fit + predict, dispatched by task type  (used by block 1)
# ---------------------------------------------------------------------------
def _predict_one(model, ttype: str, X) -> np.ndarray:
    """Dispatch a single prediction call by task_type."""
    if ttype == "binary":
        return model.predict_proba(X)
    elif ttype == "regression":
        return model.predict(X)
    else:
        raise ValueError(f"Unknown task_type: {ttype}")


def fit_and_predict(model_cfg: dict, X_tr, y_tr, X_ev, random_state: int) -> np.ndarray:
    """Returns 1D array. Probabilities for binary, continuous values for regression.
    The random_state is injected into the model's params at construction time so
    every model that supports it becomes deterministic per fold.
    """
    params = dict(model_cfg.get("params", {}))
    params["random_state"] = random_state
    model = build_model(model_cfg["class"], **params)
    model.fit(X_tr, y_tr)
    return _predict_one(model, model_cfg["task_type"], X_ev)


def fit_predict_both(model_cfg: dict, X_tr, y_tr, X_ev, random_state: int,
                     also_train: bool = False):
    """Fit once, predict on eval and (optionally) on train. Returns:
        (pred_eval, pred_train_or_None)
    Used to compute generalization gap without paying for two fits.
    """
    params = dict(model_cfg.get("params", {}))
    params["random_state"] = random_state
    model = build_model(model_cfg["class"], **params)
    model.fit(X_tr, y_tr)
    ttype = model_cfg["task_type"]
    pred_ev = _predict_one(model, ttype, X_ev)
    pred_tr = _predict_one(model, ttype, X_tr) if also_train else None
    return pred_ev, pred_tr


# ---------------------------------------------------------------------------
# BLOCK 1: main nested-CV experiment
# ---------------------------------------------------------------------------
def run_main_experiment(cfg: dict, args, data, labels_df, splits_df, X, base_seed):
    pred_dir = predictions_dir(args.run_id)
    met_dir  = metrics_dir(args.run_id)
    verbosity = cfg.get("logging", {}).get("verbosity", 1)
    log_train = bool(cfg.get("evaluation", {}).get("log_train_metrics", True))
    append = bool(getattr(args, "append", False))
    models_run = [m["name"] for m in cfg["models"]]

    # task lookup: name  (task_dict, y array, mask array)
    task_lookup = {}
    for task in cfg["tasks"]:
        sub = labels_df[labels_df["task"] == task["name"]].sort_values("sample_id")
        ttype = task.get("task_type", "binary")
        if ttype == "binary":
            y = sub["y"].to_numpy().astype(int)
        else:
            y = sub["y"].to_numpy().astype(float)
        mask = sub["mask"].to_numpy().astype(int)
        task_lookup[task["name"]] = (task, y, mask)

    # group models by task_type for cleaner dispatch
    models_for = {
        "binary":     [m for m in cfg["models"] if m.get("task_type") == "binary"],
        "regression": [m for m in cfg["models"] if m.get("task_type") == "regression"],
    }

    cgrid = np.arange(
        cfg["evaluation"]["cutoff_grid_start"],
        cfg["evaluation"]["cutoff_grid_stop"] + 1e-9,
        cfg["evaluation"]["cutoff_grid_step"],
    )

    pred_rows = []
    met_rows  = []
    cut_rows  = []

    pairs = (splits_df[["outer_fold", "inner_fold"]]
             .drop_duplicates()
             .sort_values(["outer_fold", "inner_fold"])
             .to_records(index=False))

    t0 = time.time()
    fit_count = 0
    fit_time_sum = 0.0
    n_pairs = len(pairs)
    _p(f"[main] starting nested CV: {n_pairs} (outer,inner) folds, "
       f"models={models_run}")
    for (outer_fold, inner_fold) in pairs:
        sub = splits_df[
            (splits_df["outer_fold"] == outer_fold) &
            (splits_df["inner_fold"] == inner_fold)
        ]
        train_idx = sub.loc[sub["role"] == "train", "sample_id"].to_numpy()
        eval_role = "test" if inner_fold == -1 else "val"
        eval_idx = sub.loc[sub["role"] == eval_role, "sample_id"].to_numpy()

        if verbosity >= 1 and inner_fold == -1:
            print(f"[experiment] outer={outer_fold} (outer-only test fit)")
        if verbosity >= 2:
            print(f"[experiment]   outer={outer_fold} inner={inner_fold} "
                  f"({eval_role} n={len(eval_idx)})")

        for task in cfg["tasks"]:
            tcfg, y_all, mask_all = task_lookup[task["name"]]
            ttype = tcfg.get("task_type", "binary")

            # mask: keep only observed rows
            tr = train_idx[mask_all[train_idx] == 1]
            ev = eval_idx[mask_all[eval_idx] == 1]
            if len(tr) == 0 or len(ev) == 0:
                continue
            # binary: need both classes in train
            if ttype == "binary" and len(np.unique(y_all[tr])) < 2:
                continue
            # regression: need variance in train
            if ttype == "regression" and np.var(y_all[tr]) == 0:
                continue

            X_tr, X_ev = X[tr], X[ev]
            y_tr, y_ev = y_all[tr], y_all[ev]

            for m in models_for[ttype]:
                fold_seed = derive_fold_seed(base_seed, int(outer_fold), int(inner_fold))
                _fit_t0 = time.time()
                pred, pred_tr = fit_predict_both(
                    m, X_tr, y_tr, X_ev,
                    random_state=fold_seed,
                    also_train=log_train,
                )
                _dt = time.time() - _fit_t0
                fit_count += 1
                fit_time_sum += _dt
                _p(f"[main] fit {fit_count} | outer={outer_fold} inner={inner_fold} "
                   f"| {task['name']}/{m['name']} | n_tr={len(tr)} n_ev={len(ev)} "
                   f"| {_dt:.1f}s (avg {fit_time_sum / fit_count:.1f}s, "
                   f"elapsed {(time.time() - t0) / 60:.1f}m)")

                # per sample predictions: prob filled for binary, y_pred for regression
                pred_df = pd.DataFrame({
                    "outer_fold": int(outer_fold),
                    "inner_fold": int(inner_fold),
                    "role":       eval_role,
                    "task":       task["name"],
                    "task_type":  ttype,
                    "model":      m["name"],
                    "sample_id":  ev,
                    "y_true":     y_ev,
                    "prob":       pred if ttype == "binary"     else np.nan,
                    "y_pred":     pred if ttype == "regression" else np.nan,
                })
                pred_rows.append(pred_df)

                # aggregated metrics for this fold
                if ttype == "binary":
                    bm = basic_metrics(y_ev, pred)
                else:
                    bm = regression_metrics(y_ev, pred)

                met_rows.append({
                    "outer_fold": int(outer_fold),
                    "inner_fold": int(inner_fold),
                    "role":       eval_role,
                    "task":       task["name"],
                    "task_type":  ttype,
                    "model":      m["name"],
                    **bm,
                })

                # generalization gap support: log train set metrics from
                # the same fitted model (no refit cost) when enabled.
                if log_train and pred_tr is not None:
                    if ttype == "binary":
                        bm_tr = basic_metrics(y_tr, pred_tr)
                    else:
                        bm_tr = regression_metrics(y_tr, pred_tr)
                    met_rows.append({
                        "outer_fold": int(outer_fold),
                        "inner_fold": int(inner_fold),
                        "role":       "train",
                        "task":       task["name"],
                        "task_type":  ttype,
                        "model":      m["name"],
                        **bm_tr,
                    })

                # cutoff sweep  binary only
                if ttype == "binary":
                    cs = cutoff_sweep(y_ev, pred, cgrid)
                    cs.insert(0, "model",      m["name"])
                    cs.insert(0, "task",       task["name"])
                    cs.insert(0, "role",       eval_role)
                    cs.insert(0, "inner_fold", int(inner_fold))
                    cs.insert(0, "outer_fold", int(outer_fold))
                    cut_rows.append(cs)

    # ---- write outputs -----------------------------------------------------
    if not pred_rows:
        print("[experiment] No predictions produced. Check splits / labels / masks.")
        return

    preds = pd.concat(pred_rows, ignore_index=True)
    _write_or_append(preds, pred_dir / "predictions.csv", models_run, append)

    metrics = pd.DataFrame(met_rows)
    _write_or_append(metrics, met_dir / "metrics_by_fold.csv", models_run, append)

    if cut_rows:
        cutoffs = pd.concat(cut_rows, ignore_index=True)
        _write_or_append(cutoffs, met_dir / "cutoff_sweep.csv", models_run, append)
        n_cut = len(cutoffs)
    else:
        n_cut = 0

    write_json({
        "run_id":      args.run_id,
        "base_seed":   base_seed,
        "n_predictions": int(len(preds)),
        "n_metrics":     int(len(metrics)),
        "n_cutoff_rows": int(n_cut),
        "elapsed_sec":   round(time.time() - t0, 2),
        "tasks":  [t["name"] for t in cfg["tasks"]],
        "models": [m["name"] for m in cfg["models"]],
    }, met_dir / "predict_log.json")

    print(f"[experiment] DONE in {time.time() - t0:.1f}s")
    print(f"[experiment]   base_seed            : {base_seed}")
    print(f"[experiment]   predictions.csv      : {len(preds)} rows")
    print(f"[experiment]   metrics_by_fold.csv  : {len(metrics)} rows")
    print(f"[experiment]   cutoff_sweep.csv     : {n_cut} rows")


# ---------------------------------------------------------------------------
# BLOCK 2: W-threshold sweep  
# ---------------------------------------------------------------------------
def _w_panel_label(panel: str, W: np.ndarray, tau: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_full, keep_mask) for one panel at threshold tau.

      panel "W>0"  -- y = (W > tau).         full set of pairs.
      panel "W+"   -- y = (W > tau).         subset to W >= 0 only.
      panel "W-"   -- y = (W < -tau).        subset to W <= 0 only.
    """
    if panel == "W>0":
        y_full = (W > tau).astype(int)
        keep   = np.ones_like(y_full, dtype=bool)
    elif panel == "W+":
        keep   = W >= 0
        y_full = (W > tau).astype(int)
    elif panel == "W-":
        keep   = W <= 0
        y_full = (W < -tau).astype(int)
    else:
        raise ValueError(f"Unknown panel: {panel}")
    return y_full, keep


def run_w_threshold_sweep(cfg, args, data, splits_df, X, base_seed,
                          model_name: str = "sparse_logistic",
                          n_tau: int = 40):
    """Run the three-panel W-threshold sweep matching Daniele's slide 21."""
    print("\n" + "-" * 70)
    print("[w_sweep] W-threshold sweep (slide 21)")
    print("-" * 70)

    met_dir = metrics_dir(args.run_id)
    W = np.asarray(data["W_continuous"], dtype=float)
    print(f"[w_sweep] W: min={W.min():.3f}  max={W.max():.3f}  "
          f"n_pos={(W>0).sum()}  n_neg={(W<0).sum()}  n_zero={(W==0).sum()}")

    try:
        model_cfg = next(m for m in cfg["models"]
                         if m["name"] == model_name and m.get("task_type") == "binary")
    except StopIteration:
        print(f"[w_sweep] WARNING: no binary model named '{model_name}' "
              f"in config; skipping.")
        return
    print(f"[w_sweep] model = {model_name} ({model_cfg['class']})")

    # tau grids: match the slide. Top of each grid uses the 99th percentile.
    w_pos = W[W > 0]
    w_neg = W[W < 0]
    tau_pos_max = float(np.quantile(w_pos, 0.99))     if len(w_pos) else 1.0
    tau_neg_max = float(np.quantile(-w_neg, 0.99))    if len(w_neg) else 1.0
    eps = 1e-6
    grids = {
        "W>0": np.linspace(0.0, tau_pos_max, n_tau),
        "W+":  np.linspace(eps, tau_pos_max, n_tau),
        "W-":  np.linspace(eps, tau_neg_max, n_tau),
    }
    for panel, g in grids.items():
        print(f"[w_sweep]   {panel}: {len(g)} thresholds in "
              f"[{g[0]:.3g}, {g[-1]:.3g}]")

    outer_groups = (splits_df[splits_df["inner_fold"] == -1]
                    .groupby("outer_fold"))

    t0 = time.time()
    rows = []
    nfit = 0
    for panel, grid in grids.items():
        _p(f"[w_sweep] running panel {panel} ...")
        for outer_fold, sub in outer_groups:
            tr = sub.loc[sub["role"] == "train", "sample_id"].to_numpy()
            ev = sub.loc[sub["role"] == "test",  "sample_id"].to_numpy()

            for tau in grid:
                y_full, keep = _w_panel_label(panel, W, float(tau))

                tr_k = tr[keep[tr]]
                ev_k = ev[keep[ev]]
                if len(tr_k) == 0 or len(ev_k) == 0:
                    continue
                if len(np.unique(y_full[tr_k])) < 2:
                    continue
                if len(np.unique(y_full[ev_k])) < 2:
                    continue

                params = dict(model_cfg.get("params", {}))
                params["random_state"] = derive_fold_seed(base_seed,
                                                          int(outer_fold), -1)
                model = build_model(model_cfg["class"], **params)
                model.fit(X[tr_k], y_full[tr_k])
                prob = model.predict_proba(X[ev_k])
                nfit += 1
                if nfit % 25 == 0:
                    _p(f"[w_sweep]   {nfit} fits done "
                       f"(elapsed {(time.time() - t0) / 60:.1f}m)")

                bm = basic_metrics(y_full[ev_k], prob)
                rows.append({
                    "panel":      panel,
                    "threshold":  float(tau),
                    "outer_fold": int(outer_fold),
                    "n":          int(bm["n"]),
                    "n_pos":      int(bm["n_pos"]),
                    "auc":        float(bm["auc"]) if bm["auc"] == bm["auc"] else np.nan,
                    "ap":         float(bm["ap"])  if bm["ap"]  == bm["ap"]  else np.nan,
                })

    out_df = pd.DataFrame(rows)
    write_csv(out_df, met_dir / "w_threshold_sweep.csv")
    write_json({
        "model":       model_name,
        "n_tau":       int(n_tau),
        "tau_pos_max": tau_pos_max,
        "tau_neg_max": tau_neg_max,
        "n_rows":      int(len(out_df)),
        "elapsed_sec": round(time.time() - t0, 2),
    }, met_dir / "w_threshold_sweep_log.json")
    print(f"[w_sweep] DONE in {time.time()-t0:.1f}s  "
          f"({len(out_df)} rows -> w_threshold_sweep.csv)")


# ---------------------------------------------------------------------------
# BLOCK 3: Lasso stability selection
# ---------------------------------------------------------------------------
def _fit_lasso_binary(X_tr, y_tr, C: float, max_iter: int, random_state: int
                      ) -> np.ndarray:
    """L1 logistic regression with class_weight='balanced'."""
    m = LogisticRegression(
        penalty="l1", solver="liblinear", C=C,
        max_iter=max_iter, class_weight="balanced",
        random_state=random_state,
    )
    m.fit(X_tr, y_tr)
    return m.coef_.ravel()


def _fit_lasso_regression(X_tr, y_tr, alpha: float, max_iter: int,
                          random_state: int) -> np.ndarray:
    """Pure Lasso for continuous y."""
    m = Lasso(alpha=alpha, max_iter=max_iter, random_state=random_state)
    m.fit(X_tr, y_tr)
    return m.coef_.ravel()


def _summarize_lasso(long_df: pd.DataFrame, labels_df: pd.DataFrame,
                     n_folds: int) -> pd.DataFrame:
    """Per (lambda, feature_id): selection_freq, mean_coef, sign_consistency."""
    if long_df.empty:
        return pd.DataFrame(columns=[
            "log_lambda", "lambda", "feature_id", "feature_source",
            "feature_name", "n_selected", "selection_freq",
            "mean_coef", "sign_consistency",
        ])
    grp = long_df.groupby(["log_lambda", "lambda", "feature_id"])

    def _row(g):
        n = len(g)
        signs = np.sign(g["coef"].to_numpy())
        majority = 1 if (signs > 0).sum() >= (signs < 0).sum() else -1
        consistency = float((signs == majority).mean()) if n > 0 else np.nan
        return pd.Series({
            "n_selected":       n,
            "selection_freq":   n / n_folds,
            "mean_coef":        float(g["coef"].mean()),
            "sign_consistency": consistency,
        })

    out = grp.apply(_row).reset_index()
    out = out.merge(labels_df, on="feature_id", how="left")
    return out[[
        "log_lambda", "lambda", "feature_id", "feature_source",
        "feature_name", "n_selected", "selection_freq",
        "mean_coef", "sign_consistency",
    ]]


def run_lasso_selection(cfg, args, data, labels_df, splits_df, X, base_seed,
                        target: str = "Y",
                        log_lambda_min: float = -4.0,
                        log_lambda_max: float =  4.0,
                        n_lambda: int = 9,
                        max_iter: int = 5000):
    """Run the lasso stability-selection sweep across the lambda grid."""
    print("\n" + "-" * 70)
    print(f"[lasso] Lasso stability selection (target={target})")
    print("-" * 70)

    met_dir = metrics_dir(args.run_id)
    feat_labels = build_feature_labels(data)
    print(f"[lasso] X: {X.shape}  "
          f"({(feat_labels.feature_source=='Xb').sum()} Xb "
          f"+ {(feat_labels.feature_source=='Xv').sum()} Xv features)")

    # Choose target
    if target == "Y":
        sub = labels_df[labels_df["task_type"] == "binary"]
        if sub.empty:
            print("[lasso] No binary task in labels.csv; skipping.")
            return
        first = sub["task"].iloc[0]
        sub = sub[sub["task"] == first].sort_values("sample_id")
        y    = sub["y"].to_numpy().astype(int)
        mask = sub["mask"].to_numpy().astype(int)
        print(f"[lasso] target task: {first}  "
              f"n_pos={int(y[mask==1].sum())} / {int(mask.sum())}")
    elif target == "W":
        sub = labels_df[labels_df["task_type"] == "regression"]
        if sub.empty:
            print("[lasso] No regression task in labels.csv; skipping.")
            return
        first = sub["task"].iloc[0]
        sub = sub[sub["task"] == first].sort_values("sample_id")
        y    = sub["y"].to_numpy().astype(float)
        mask = sub["mask"].to_numpy().astype(int)
        print(f"[lasso] target task: {first}  n_observed={int(mask.sum())}")
    else:
        raise ValueError(target)

    log_lambda_grid = np.linspace(log_lambda_min, log_lambda_max, n_lambda)
    print(f"[lasso] grid: 10^[{log_lambda_min}, {log_lambda_max}], "
          f"{n_lambda} points -> {10**log_lambda_grid}")

    outer_groups = (splits_df[splits_df["inner_fold"] == -1]
                    .groupby("outer_fold"))

    t0 = time.time()
    rows = []
    for log_lam in log_lambda_grid:
        lam = float(10.0 ** log_lam)
        # sklearn:  LogisticRegression(C=1/lam),  Lasso(alpha=lam)
        C     = 1.0 / lam
        alpha = lam

        for outer_fold, sub in outer_groups:
            tr = sub.loc[sub["role"] == "train", "sample_id"].to_numpy()
            tr = tr[mask[tr] == 1]
            if len(tr) == 0:
                continue
            if target == "Y" and len(np.unique(y[tr])) < 2:
                continue
            if target == "W" and np.var(y[tr]) == 0:
                continue

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr])
            y_tr = y[tr]

            seed = derive_fold_seed(base_seed, int(outer_fold), -1)
            if target == "Y":
                coefs = _fit_lasso_binary(X_tr, y_tr, C=C,
                                          max_iter=max_iter, random_state=seed)
            else:
                coefs = _fit_lasso_regression(X_tr, y_tr, alpha=alpha,
                                              max_iter=max_iter, random_state=seed)

            nz = np.nonzero(coefs)[0]
            _p(f"[lasso]   lam={lam:.4g} fold={outer_fold} "
               f"nnz={len(nz)} (elapsed {(time.time() - t0) / 60:.1f}m)")
            for fid in nz:
                rows.append({
                    "log_lambda": float(log_lam),
                    "lambda":     lam,
                    "outer_fold": int(outer_fold),
                    "feature_id": int(fid),
                    "coef":       float(coefs[fid]),
                })
        _p(f"[lasso]   log10(lam)={log_lam:+.2f}  lam={lam:.4g}  done")

    long_df = pd.DataFrame(rows)
    if not long_df.empty:
        long_df = long_df.merge(feat_labels, on="feature_id", how="left")
    write_csv(long_df, met_dir / "lasso_selection.csv")

    n_folds = splits_df.loc[splits_df["inner_fold"] == -1, "outer_fold"].nunique()
    summary_df = _summarize_lasso(long_df, feat_labels, n_folds)
    write_csv(summary_df, met_dir / "lasso_path_summary.csv")

    write_json({
        "target":         target,
        "log_lambda_min": float(log_lambda_min),
        "log_lambda_max": float(log_lambda_max),
        "n_lambda":       int(n_lambda),
        "n_folds":        int(n_folds),
        "n_features":     int(len(feat_labels)),
        "n_features_Xb":  int((feat_labels.feature_source == "Xb").sum()),
        "n_features_Xv":  int((feat_labels.feature_source == "Xv").sum()),
        "max_iter":       int(max_iter),
        "n_rows":         int(len(long_df)),
        "elapsed_sec":    round(time.time() - t0, 2),
    }, met_dir / "lasso_log.json")
    print(f"[lasso] DONE in {time.time()-t0:.1f}s  "
          f"({len(long_df)} nonzero rows, {len(summary_df)} summary rows)")


# ---------------------------------------------------------------------------
# BLOCK 4: XGBoost (max_depth x learning_rate) grid
# ---------------------------------------------------------------------------
def _summarize_xgb_grid(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "max_depth", "learning_rate", "n_folds",
            "auc_mean", "auc_sd", "ap_mean", "ap_sd",
            "brier_mean", "brier_sd",
        ])
    g = df.groupby(["max_depth", "learning_rate"])
    return g.agg(
        n_folds   = ("auc",   "count"),
        auc_mean  = ("auc",   "mean"),
        auc_sd    = ("auc",   "std"),
        ap_mean   = ("ap",    "mean"),
        ap_sd     = ("ap",    "std"),
        brier_mean= ("brier", "mean"),
        brier_sd  = ("brier", "std"),
    ).reset_index()


def run_xgb_grid(cfg, args, data, labels_df, splits_df, X, base_seed,
                 max_depths=(3, 5, 8),
                 learning_rates=(0.01, 0.05, 0.2)):
    """3x3 (max_depth, learning_rate) grid for xgb_clf on the binary task."""
    print("\n" + "-" * 70)
    print("[xgb_grid] XGBoost hyperparameter grid")
    print("-" * 70)

    met_dir = metrics_dir(args.run_id)

    bin_tasks = labels_df[labels_df["task_type"] == "binary"]
    if bin_tasks.empty:
        print("[xgb_grid] No binary task in labels.csv; skipping.")
        return
    task_name = bin_tasks["task"].iloc[0]
    sub = bin_tasks[bin_tasks["task"] == task_name].sort_values("sample_id")
    y    = sub["y"].to_numpy().astype(int)
    mask = sub["mask"].to_numpy().astype(int)
    print(f"[xgb_grid] target: {task_name}  "
          f"n_pos={int(y[mask==1].sum())} / {int(mask.sum())}")

    # Base xgb_clf params from the config (n_estimators / subsample / colsample);
    # we override max_depth, learning_rate, random_state per cell/fold.
    try:
        xgb_cfg = next(m for m in cfg["models"]
                       if m["class"] == "xgb_clf" and m.get("task_type") == "binary")
        base_xgb_params = dict(xgb_cfg.get("params", {}))
    except StopIteration:
        print("[xgb_grid] WARNING: no xgb_clf entry in config; using wrapper defaults.")
        base_xgb_params = {}
    for k in ("max_depth", "learning_rate", "random_state"):
        base_xgb_params.pop(k, None)

    print(f"[xgb_grid] grid: max_depth={list(max_depths)}  "
          f"learning_rate={list(learning_rates)}  "
          f"({len(max_depths) * len(learning_rates)} cells)")
    print(f"[xgb_grid] fixed params: {base_xgb_params}")

    outer_groups = (splits_df[splits_df["inner_fold"] == -1]
                    .groupby("outer_fold"))

    t0 = time.time()
    rows = []
    n_cells = len(max_depths) * len(learning_rates)
    cell_idx = 0
    for max_depth in max_depths:
        for lr in learning_rates:
            cell_idx += 1
            for outer_fold, sub in outer_groups:
                tr = sub.loc[sub["role"] == "train", "sample_id"].to_numpy()
                ev = sub.loc[sub["role"] == "test",  "sample_id"].to_numpy()
                tr = tr[mask[tr] == 1]
                ev = ev[mask[ev] == 1]
                if len(tr) == 0 or len(ev) == 0:
                    continue
                if len(np.unique(y[tr])) < 2 or len(np.unique(y[ev])) < 2:
                    continue

                params = dict(base_xgb_params)
                params["max_depth"]     = int(max_depth)
                params["learning_rate"] = float(lr)
                params["random_state"]  = derive_fold_seed(
                    base_seed, int(outer_fold), -1
                )
                model = build_model("xgb_clf", **params)
                model.fit(X[tr], y[tr])
                prob = model.predict_proba(X[ev])
                _p(f"[xgb_grid]   cell {cell_idx}/{n_cells} "
                   f"(max_depth={max_depth}, lr={lr}) fold={outer_fold} done "
                   f"(elapsed {(time.time() - t0) / 60:.1f}m)")

                bm = basic_metrics(y[ev], prob)
                rows.append({
                    "max_depth":     int(max_depth),
                    "learning_rate": float(lr),
                    "outer_fold":    int(outer_fold),
                    "n":             int(bm["n"]),
                    "n_pos":         int(bm["n_pos"]),
                    "n_neg":         int(bm["n_neg"]),
                    "auc":           float(bm["auc"])   if bm["auc"]   == bm["auc"]   else np.nan,
                    "ap":            float(bm["ap"])    if bm["ap"]    == bm["ap"]    else np.nan,
                    "brier":         float(bm["brier"]) if bm["brier"] == bm["brier"] else np.nan,
                })
            _p(f"[xgb_grid]   cell {cell_idx}/{n_cells}: "
                  f"max_depth={max_depth} lr={lr} done")

    long_df = pd.DataFrame(rows)
    write_csv(long_df, met_dir / "xgb_grid.csv")
    summary_df = _summarize_xgb_grid(long_df)
    write_csv(summary_df, met_dir / "xgb_grid_summary.csv")

    if not summary_df.empty:
        best = summary_df.sort_values("auc_mean", ascending=False).iloc[0]
        print(f"[xgb_grid] best cell: max_depth={int(best['max_depth'])}  "
              f"learning_rate={best['learning_rate']}  "
              f"auc={best['auc_mean']:.4f} +/- {best['auc_sd']:.4f}")

    write_json({
        "target":          task_name,
        "max_depths":      [int(x) for x in max_depths],
        "learning_rates":  [float(x) for x in learning_rates],
        "n_cells":         int(n_cells),
        "n_rows":          int(len(long_df)),
        "base_xgb_params": base_xgb_params,
        "elapsed_sec":     round(time.time() - t0, 2),
    }, met_dir / "xgb_grid_log.json")
    print(f"[xgb_grid] DONE in {time.time()-t0:.1f}s "
          f"({len(long_df)} rows, {len(summary_df)} cells)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--skip-w-sweep",  action="store_true",
                    help="Skip the W-threshold sweep (block 2).")
    ap.add_argument("--skip-lasso",    action="store_true",
                    help="Skip the lasso stability selection (block 3).")
    ap.add_argument("--skip-xgb-grid", action="store_true",
                    help="Skip the XGBoost hyperparameter grid (block 4).")
    ap.add_argument("--only-models", default=None,
                    help="Comma-separated model names to run (affects block 1, "
                         "the main nested CV). Other models in the config are "
                         "ignored for this run. Use with --append to add a new "
                         "model without re-running the baselines.")
    ap.add_argument("--append", action="store_true",
                    help="Merge block-1 outputs into the existing CSVs for this "
                         "run-id instead of overwriting. Rows for the model(s) "
                         "being run are replaced; all other models are kept.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pre_dir = preprocessing_dir(args.run_id)

    # --only-models: restrict block 1 to just these model names (e.g. a new MLP).
    if args.only_models:
        wanted = [s.strip() for s in args.only_models.split(",") if s.strip()]
        available = {m["name"] for m in cfg["models"]}
        missing = [w for w in wanted if w not in available]
        if missing:
            print(f"[experiment] WARNING: --only-models names not in config: "
                  f"{missing}", flush=True)
        cfg["models"] = [m for m in cfg["models"] if m["name"] in wanted]
        if not cfg["models"]:
            raise SystemExit("[experiment] --only-models left no models to run.")
        print(f"[experiment] --only-models -> "
              f"{[m['name'] for m in cfg['models']]}", flush=True)
        if args.append:
            print("[experiment] --append: new rows will be merged into existing "
                  "block-1 CSVs (baselines kept).", flush=True)

    # Determinism: set global seeds at the start; per-fold seeds are derived
    # from the base seed and injected into each model.
    base_seed = int(cfg.get("resampling", {}).get("seed", 42))
    set_global_seeds(base_seed)

    # ---- load stage 1 outputs (shared by all blocks) ----------------------
    print(f"[experiment] loading preprocessing outputs from {pre_dir}")
    data      = load_npz(pre_dir / "data_filtered.npz")
    labels_df = pd.read_csv(pre_dir / "labels.csv")
    splits_df = pd.read_csv(pre_dir / "splits.csv")

    X = build_feature_matrix(data)
    print(f"[experiment] feature matrix: {X.shape}")

    # ---- (1) main nested-CV experiment ------------------------------------
    run_main_experiment(cfg, args, data, labels_df, splits_df, X, base_seed)

    # ---- (2) W-threshold sweep --------------------------------------------
    if not args.skip_w_sweep:
        run_w_threshold_sweep(cfg, args, data, splits_df, X, base_seed)

    # ---- (3) lasso stability selection ------------------------------------
    if not args.skip_lasso:
        run_lasso_selection(cfg, args, data, labels_df, splits_df, X, base_seed)

    # ---- (4) XGBoost hyperparameter grid ----------------------------------
    if not args.skip_xgb_grid:
        run_xgb_grid(cfg, args, data, labels_df, splits_df, X, base_seed)


if __name__ == "__main__":
    main()
