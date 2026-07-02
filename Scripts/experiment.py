"""
Stage 2: experiment.

Reads stage 1 outputs and runs the baseline models on the 10 outer train/test
splits only. The previous nested 10x5 validation CV, W-threshold sweep, and
lasso stability-selection blocks are intentionally disabled for the simplified
thesis baseline.

Writes:
  predictions/predictions.csv
  metrics/metrics_by_fold.csv
  metrics/cutoff_sweep.csv        binary tasks only

Usage:
  python experiment.py --config default.yaml --run-id <id_from_preprocess>
"""
from __future__ import annotations
import argparse
import functools
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

# Print everything immediately (flush) so progress is visible during long runs.
print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import preprocessing_dir, predictions_dir, metrics_dir
from utils import (
    load_config, load_npz, write_csv, write_json,
    basic_metrics, cutoff_sweep, regression_metrics,
    set_global_seeds, derive_fold_seed,
)
from models import build_model

# Optional legacy imports removed: no lasso stability-selection block is run.

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


# ---------------------------------------------------------------------------
# Target transforms (regression only). Fit in transformed space, score on the
# original scale. `logit` clips to [eps, 1-eps] first so 0/1 don't blow up.
# ---------------------------------------------------------------------------
def target_forward(y: np.ndarray, kind: str, eps: float) -> np.ndarray:
    """Map the raw target into the space the model is trained in."""
    if kind in (None, "none"):
        return np.asarray(y, dtype=float)
    if kind == "logit":
        yc = np.clip(np.asarray(y, dtype=float), eps, 1.0 - eps)
        return np.log(yc / (1.0 - yc))
    raise ValueError(f"Unknown target transform: {kind}")


def target_inverse(p: np.ndarray, kind: str) -> np.ndarray:
    """Map model output back to the original target scale for scoring/storage."""
    p = np.asarray(p, dtype=float)
    if kind in (None, "none"):
        return p
    if kind == "logit":
        return 1.0 / (1.0 + np.exp(-p))          # expit, back into (0,1)
    raise ValueError(f"Unknown target transform: {kind}")


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
    """Fit once, predict on eval and optionally on train.

    Returns (pred_eval, pred_train_or_None, info_trace). The third element is
    non-empty only for VIB-style models that expose get_info_trace().
    """
    params = dict(model_cfg.get("params", {}))
    params["random_state"] = random_state
    model = build_model(model_cfg["class"], **params)
    model.fit(X_tr, y_tr)
    ttype = model_cfg["task_type"]
    pred_ev = _predict_one(model, ttype, X_ev)
    pred_tr = _predict_one(model, ttype, X_tr) if also_train else None
    info_trace = model.get_info_trace() if hasattr(model, "get_info_trace") else []
    return pred_ev, pred_tr, info_trace


def fit_predict_joint_yw(model_cfg: dict, X_tr, y_tr, w_tr, X_ev,
                         random_state: int, also_train: bool = False):
    """Fit a joint Y+W latent model once and predict both tasks."""
    params = dict(model_cfg.get("params", {}))
    params["random_state"] = random_state
    model = build_model(model_cfg["class"], **params)
    model.fit(X_tr, y_tr, w_tr)
    pred_ev = model.predict(X_ev)
    pred_tr = model.predict(X_tr) if also_train else None
    info_trace = model.get_info_trace() if hasattr(model, "get_info_trace") else []
    return pred_ev, pred_tr, info_trace


# ---------------------------------------------------------------------------
# BLOCK 1: main 10-split train/test experiment
# ---------------------------------------------------------------------------
def run_main_experiment(cfg: dict, args, data, labels_df, splits_df, X, base_seed):
    pred_dir = predictions_dir(args.run_id)
    met_dir  = metrics_dir(args.run_id)
    verbosity = cfg.get("logging", {}).get("verbosity", 1)
    log_train = bool(cfg.get("evaluation", {}).get("log_train_metrics", True))
    logit_eps = float(cfg.get("evaluation", {}).get("logit_eps", 1e-3))
    append = bool(getattr(args, "append", False))
    models_run = [m["name"] for m in cfg["models"]]

    # Inject the bacterial feature block width for two-tower models. Baselines
    # still receive the exact same concatenated X = [Xb, Xv]; two-tower models
    # only use xb_dim to split this matrix internally.
    xb_dim = int(np.asarray(data["Xb_filtered"]).shape[1])
    for _m in cfg["models"]:
        if str(_m.get("class", "")).startswith("twotower_"):
            _m.setdefault("params", {})["xb_dim"] = xb_dim

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
    joint_yw_models = [m for m in cfg["models"] if m.get("task_type") == "joint_y_w"]

    cgrid = np.arange(
        cfg["evaluation"]["cutoff_grid_start"],
        cfg["evaluation"]["cutoff_grid_stop"] + 1e-9,
        cfg["evaluation"]["cutoff_grid_step"],
    )

    pred_rows = []
    met_rows  = []
    cut_rows  = []
    info_rows = []

    pairs = (splits_df[["outer_fold", "inner_fold"]]
             .drop_duplicates()
             .sort_values(["outer_fold", "inner_fold"])
             .to_records(index=False))

    t0 = time.time()
    fit_count = 0
    fit_time_sum = 0.0
    n_pairs = len(pairs)
    _p(f"[main] starting 10-split train/test evaluation: {n_pairs} split(s), "
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

            # target transform (regression only): fit on transformed y, then
            # invert predictions back to the original scale before scoring.
            ttransform = tcfg.get("transform", "none") if ttype == "regression" else "none"
            y_tr_fit = target_forward(y_tr, ttransform, logit_eps)

            for m in models_for[ttype]:
                fold_seed = derive_fold_seed(base_seed, int(outer_fold), int(inner_fold))
                _fit_t0 = time.time()
                pred, pred_tr, trace = fit_predict_both(
                    m, X_tr, y_tr_fit, X_ev,
                    random_state=fold_seed,
                    also_train=log_train,
                )
                if trace:
                    for row in trace:
                        rr = dict(row)
                        rr.update({
                            "outer_fold": int(outer_fold),
                            "inner_fold": int(inner_fold),
                            "role": "train_trace",
                            "task": task["name"],
                            "model": m["name"],
                        })
                        info_rows.append(rr)
                # back to original scale (no-op unless a transform is set)
                pred = target_inverse(pred, ttransform)
                if pred_tr is not None:
                    pred_tr = target_inverse(pred_tr, ttransform)
                _dt = time.time() - _fit_t0
                fit_count += 1
                fit_time_sum += _dt
                _p(f"[main] fit {fit_count} | outer={outer_fold} inner={inner_fold} "
                   f"| {task['name']}/{m['name']}"
                   f"{' [logit]' if ttransform == 'logit' else ''} "
                   f"| n_tr={len(tr)} n_ev={len(ev)} "
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


        # Joint latent-space models: fit once on rows where both Y and W are observed,
        # then emit predictions/metrics for both tasks. The W side is now a
        # classification target (w_class = 1[W > 0.5]); the joint model's W head
        # is trained with BCE and predicts a probability. This is only used for
        # models with task_type == "joint_y_w".
        if joint_yw_models and "y" in task_lookup and "w_class" in task_lookup:
            y_task, y_all, y_mask = task_lookup["y"]
            w_task, w_all, w_mask = task_lookup["w_class"]
            tr = train_idx[(y_mask[train_idx] == 1) & (w_mask[train_idx] == 1)]
            ev = eval_idx[(y_mask[eval_idx] == 1) & (w_mask[eval_idx] == 1)]
            if (len(tr) > 0 and len(ev) > 0
                    and len(np.unique(y_all[tr])) >= 2
                    and len(np.unique(w_all[tr])) >= 2):
                X_tr, X_ev = X[tr], X[ev]
                y_tr, y_ev = y_all[tr].astype(int), y_all[ev].astype(int)
                w_tr, w_ev = w_all[tr].astype(int), w_all[ev].astype(int)

                for m in joint_yw_models:
                    fold_seed = derive_fold_seed(base_seed, int(outer_fold), int(inner_fold))
                    _fit_t0 = time.time()
                    pred_ev, pred_tr, trace = fit_predict_joint_yw(
                        m, X_tr, y_tr, w_tr.astype(float), X_ev,
                        random_state=fold_seed,
                        also_train=log_train,
                    )
                    y_prob = pred_ev["y_prob"]
                    w_prob = pred_ev["w_pred"]   # joint W head now returns a probability
                    if pred_tr is not None:
                        y_prob_tr = pred_tr["y_prob"]
                        w_prob_tr = pred_tr["w_pred"]
                    else:
                        y_prob_tr = None
                        w_prob_tr = None
                    _dt = time.time() - _fit_t0
                    fit_count += 1
                    fit_time_sum += _dt
                    _p(f"[main] fit {fit_count} | outer={outer_fold} inner={inner_fold} "
                       f"| joint(Y,Wclass)/{m['name']} "
                       f"| n_tr={len(tr)} n_ev={len(ev)} "
                       f"| {_dt:.1f}s (avg {fit_time_sum / fit_count:.1f}s, "
                       f"elapsed {(time.time() - t0) / 60:.1f}m)")

                    if trace:
                        for row in trace:
                            rr = dict(row)
                            rr.update({
                                "outer_fold": int(outer_fold),
                                "inner_fold": int(inner_fold),
                                "role": "train_trace",
                                "task": "y+w_class",
                                "model": m["name"],
                            })
                            info_rows.append(rr)

                    pred_rows.append(pd.DataFrame({
                        "outer_fold": int(outer_fold), "inner_fold": int(inner_fold),
                        "role": eval_role, "task": "y", "task_type": "binary",
                        "model": m["name"], "sample_id": ev, "y_true": y_ev,
                        "prob": y_prob, "y_pred": np.nan,
                    }))
                    pred_rows.append(pd.DataFrame({
                        "outer_fold": int(outer_fold), "inner_fold": int(inner_fold),
                        "role": eval_role, "task": "w_class", "task_type": "binary",
                        "model": m["name"], "sample_id": ev, "y_true": w_ev,
                        "prob": w_prob, "y_pred": np.nan,
                    }))

                    met_rows.append({
                        "outer_fold": int(outer_fold), "inner_fold": int(inner_fold),
                        "role": eval_role, "task": "y", "task_type": "binary",
                        "model": m["name"], **basic_metrics(y_ev, y_prob),
                    })
                    met_rows.append({
                        "outer_fold": int(outer_fold), "inner_fold": int(inner_fold),
                        "role": eval_role, "task": "w_class", "task_type": "binary",
                        "model": m["name"], **basic_metrics(w_ev, w_prob),
                    })

                    if log_train and pred_tr is not None:
                        met_rows.append({
                            "outer_fold": int(outer_fold), "inner_fold": int(inner_fold),
                            "role": "train", "task": "y", "task_type": "binary",
                            "model": m["name"], **basic_metrics(y_tr, y_prob_tr),
                        })
                        met_rows.append({
                            "outer_fold": int(outer_fold), "inner_fold": int(inner_fold),
                            "role": "train", "task": "w_class", "task_type": "binary",
                            "model": m["name"], **basic_metrics(w_tr, w_prob_tr),
                        })

                    for _tname, _yt, _yp in (("y", y_ev, y_prob), ("w_class", w_ev, w_prob)):
                        cs = cutoff_sweep(_yt, _yp, cgrid)
                        cs.insert(0, "model",      m["name"])
                        cs.insert(0, "task",       _tname)
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

    if info_rows:
        info = pd.DataFrame(info_rows)
        _write_or_append(info, met_dir / "info_theory_trace.csv", models_run, append)

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
        "n_info_rows":   int(len(info_rows)),
        "elapsed_sec":   round(time.time() - t0, 2),
        "tasks":  [t["name"] for t in cfg["tasks"]],
        "models": [m["name"] for m in cfg["models"]],
    }, met_dir / "predict_log.json")

    print(f"[experiment] DONE in {time.time() - t0:.1f}s")
    print(f"[experiment]   base_seed            : {base_seed}")
    print(f"[experiment]   predictions.csv      : {len(preds)} rows")
    print(f"[experiment]   metrics_by_fold.csv  : {len(metrics)} rows")
    print(f"[experiment]   cutoff_sweep.csv     : {n_cut} rows")
    if info_rows:
        print(f"[experiment]   info_theory_trace.csv: {len(info_rows)} rows")


# ---------------------------------------------------------------------------
# Block 2: W-threshold AUC sweep  (slide 21)
# ---------------------------------------------------------------------------
# For each threshold t over the continuous glasso edge-probability W, build the
# binary label 1[W > t] and fit an L2 ("ridge-style") logistic regression on the
# pair feature matrix X, scoring test AUC across the 10 outer train/test splits.
# We only need AUC here, not feature selection, so L2 + a fast solver (lbfgs)
# replaces the old L1/liblinear model: ~50-100x faster per fit. The (threshold x
# split) fits are independent and run in parallel via joblib.
def run_w_threshold_sweep(cfg, args, labels_df, splits_df, X, base_seed):
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
    jobs  = [(int(f), float(t)) for t in thresholds for f in folds]

    def _one(outer_fold, t):
        sf = splits_df[(splits_df["outer_fold"] == outer_fold) &
                       (splits_df["inner_fold"] == -1)]
        tr = sf.loc[sf["role"] == "train", "sample_id"].to_numpy()
        ev = sf.loc[sf["role"] == "test",  "sample_id"].to_numpy()
        tr = tr[mask_all[tr] == 1]
        ev = ev[mask_all[ev] == 1]
        ytr = (w_all[tr] > t).astype(int)
        yev = (w_all[ev] > t).astype(int)
        rec = dict(threshold=float(t), outer_fold=int(outer_fold),
                   n_pos_train=int(ytr.sum()), n_pos_test=int(yev.sum()),
                   n_train=int(len(tr)), n_test=int(len(ev)), auc=np.nan)
        if len(tr) == 0 or len(ev) == 0:
            return rec
        if len(np.unique(ytr)) < 2 or len(np.unique(yev)) < 2:
            return rec  # degenerate single-class fold -> AUC undefined
        sc  = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, penalty="l2", solver=solver,
                                 max_iter=max_iter, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), ytr)
        p = clf.predict_proba(sc.transform(X[ev]))[:, 1]
        rec["auc"] = float(roc_auc_score(yev, p))
        return rec

    t0 = time.time()
    _p(f"[w_threshold] {len(thresholds)} thresholds x {len(folds)} splits "
       f"= {len(jobs)} L2-logistic fits (n_jobs={n_jobs}, solver={solver})")
    rows = Parallel(n_jobs=n_jobs)(delayed(_one)(f, t) for (f, t) in jobs)
    per_split = pd.DataFrame(rows).sort_values(["threshold", "outer_fold"])

    agg = (per_split.groupby("threshold")["auc"]
           .agg(auc_mean="mean", auc_sd="std", n_splits="count")
           .reset_index())

    met_dir = metrics_dir(args.run_id)
    write_csv(per_split, met_dir / "w_threshold_sweep.csv")
    write_csv(agg,       met_dir / "w_threshold_sweep_summary.csv")
    write_json({
        "run_id": args.run_id, "w_task": w_task, "n_cuts": n_cuts,
        "thresholds": [float(t) for t in thresholds],
        "model": "l2_logistic", "solver": solver, "C": C, "max_iter": max_iter,
        "n_jobs": n_jobs, "elapsed_sec": round(time.time() - t0, 2),
    }, met_dir / "w_threshold_log.json")
    print(f"[w_threshold] DONE in {time.time() - t0:.1f}s -> "
          f"w_threshold_sweep.csv ({len(per_split)} rows), summary "
          f"({len(agg)} thresholds)")


# ---------------------------------------------------------------------------
# Block 3: Lasso stability selection
# ---------------------------------------------------------------------------
# Meinshausen-Buhlmann stability selection: draw B subsamples of size
# floor(frac*n) WITHOUT replacement, fit an L1 (lasso) logistic on each, and
# record the fraction of subsamples in which each feature has a non-zero
# coefficient (its selection frequency). Features with frequency >= pi are
# "stably selected". Ridge cannot be used here: it never zeros coefficients, so
# there is nothing to select. Speed comes from running the B fits in parallel
# (joblib); the lasso solver itself is unchanged.
def run_stability_selection(cfg, args, data, labels_df, X, base_seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from joblib import Parallel, delayed

    scfg = cfg.get("stability", {})
    task_name = scfg.get("task", "y")
    B         = int(scfg.get("n_bootstrap", 100))
    frac      = float(scfg.get("subsample_frac", 0.5))
    C         = float(scfg.get("C", 0.1))          # smaller C => stronger L1 => sparser
    solver    = scfg.get("solver", "liblinear")
    max_iter  = int(scfg.get("max_iter", 2000))
    pi        = float(scfg.get("pi_threshold", 0.6))
    n_jobs    = int(scfg.get("n_jobs", -1))

    sub = labels_df[labels_df["task"] == task_name].sort_values("sample_id")
    if sub.empty:
        print(f"[stability] no task '{task_name}' in labels; skipping.")
        return
    y_all    = sub["y"].to_numpy(dtype=int)
    mask_all = sub["mask"].to_numpy(dtype=int)
    obs = np.where(mask_all == 1)[0]
    Xo, yo = X[obs], y_all[obs]
    n, p = Xo.shape
    sub_n = max(2, int(round(frac * n)))

    rng   = np.random.default_rng(base_seed)
    seeds = rng.integers(0, 2**31 - 1, size=B)

    def _one(seed):
        r = np.random.default_rng(int(seed))
        idx = r.choice(n, size=sub_n, replace=False)
        Xs, ys = Xo[idx], yo[idx]
        if len(np.unique(ys)) < 2:
            return np.zeros(p, dtype=bool)
        sc  = StandardScaler().fit(Xs)
        clf = LogisticRegression(C=C, penalty="l1", solver=solver,
                                 max_iter=max_iter, class_weight="balanced")
        clf.fit(sc.transform(Xs), ys)
        return np.abs(clf.coef_).ravel() > 0

    t0 = time.time()
    _p(f"[stability] task='{task_name}' B={B} subsamples of n={sub_n}/{n} "
       f"(p={p}, C={C}, n_jobs={n_jobs})")
    sel = np.vstack(Parallel(n_jobs=n_jobs)(delayed(_one)(int(s)) for s in seeds))
    freq = sel.mean(axis=0)

    feat = build_feature_labels(data).copy()
    feat["selection_freq"] = freq
    feat["selected"] = freq >= pi
    feat = feat.sort_values("selection_freq", ascending=False)

    met_dir = metrics_dir(args.run_id)
    write_csv(feat, met_dir / "stability_selection.csv")
    write_json({
        "run_id": args.run_id, "task": task_name, "n_bootstrap": B,
        "subsample_frac": frac, "subsample_n": sub_n, "n_obs": int(n),
        "n_features": int(p), "C": C, "penalty": "l1", "solver": solver,
        "pi_threshold": pi, "n_selected": int(feat["selected"].sum()),
        "n_jobs": n_jobs, "elapsed_sec": round(time.time() - t0, 2),
    }, met_dir / "stability_log.json")
    print(f"[stability] DONE in {time.time() - t0:.1f}s -> "
          f"stability_selection.csv ({len(feat)} features, "
          f"{int(feat['selected'].sum())} selected at pi>={pi})")


# Legacy W-threshold sweep, lasso stability-selection, and XGBoost-grid
# blocks were removed for the simplified baseline.

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--skip-xgb-grid", action="store_true",
                    help="Legacy no-op: the XGBoost grid block is disabled in the simplified baseline.")
    ap.add_argument("--only-models", default=None,
                    help="Comma-separated model names to run (affects the main "
                         "10-split train/test experiment). Other models in the config are "
                         "ignored for this run. Use with --append to add a new "
                         "model without re-running the baselines.")
    ap.add_argument("--append", action="store_true",
                    help="Merge block-1 outputs into the existing CSVs for this "
                         "run-id instead of overwriting. Rows for the model(s) "
                         "being run are replaced; all other models are kept.")
    ap.add_argument("--force", action="store_true",
                    help="Recompute a block even if its output is already saved.")
    ap.add_argument("--skip-w-threshold", action="store_true",
                    help="Skip the W-threshold AUC sweep block.")
    ap.add_argument("--skip-stability", action="store_true",
                    help="Skip the lasso stability-selection block.")
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

    # skip-if-already-saved: if a block's output CSV exists, skip it
    # (use --force to recompute). Scoped to this run_id's metrics folder.
    _mdir = metrics_dir(args.run_id)
    def _saved(fname):
        return (not args.force) and (_mdir / fname).exists()

    # ---- main 10-split train/test experiment ------------------------------
    # an append / --only-models run must always run the main block.
    if (args.append or args.only_models) or not _saved("metrics_by_fold.csv"):
        print("\n[experiment] ### MAIN: 10-split train/test experiment ###")
        run_main_experiment(cfg, args, data, labels_df, splits_df, X, base_seed)
    else:
        print("\n[experiment] ### MAIN: SKIPPED (already saved) ###")

    # ---- block 2: W-threshold AUC sweep -----------------------------------
    w_on = bool(cfg.get("w_threshold", {}).get("enabled", True))
    if args.skip_w_threshold or not w_on:
        print("\n[experiment] ### W-THRESHOLD SWEEP: SKIPPED ###")
    elif _saved("w_threshold_sweep.csv"):
        print("\n[experiment] ### W-THRESHOLD SWEEP: SKIPPED (already saved) ###")
    else:
        print("\n[experiment] ### W-THRESHOLD SWEEP ###")
        run_w_threshold_sweep(cfg, args, labels_df, splits_df, X, base_seed)

    # ---- block 3: lasso stability selection -------------------------------
    s_on = bool(cfg.get("stability", {}).get("enabled", True))
    if args.skip_stability or not s_on:
        print("\n[experiment] ### STABILITY SELECTION: SKIPPED ###")
    elif _saved("stability_selection.csv"):
        print("\n[experiment] ### STABILITY SELECTION: SKIPPED (already saved) ###")
    else:
        print("\n[experiment] ### STABILITY SELECTION ###")
        run_stability_selection(cfg, args, data, labels_df, X, base_seed)


if __name__ == "__main__":
    main()
