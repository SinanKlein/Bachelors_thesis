"""
Stage 2: experiment.

Reads stage 1 outputs and runs the baseline + latent models on the 10 outer
train/test splits. One job only: the W-threshold sweep lives in analyses.py,
and the representation ablation in ablation.py.

Models may set `target_task` in the config to be trained/evaluated on a single
task only; models without it run on every task matching their task_type.

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
from dataclasses import dataclass
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
from features import pair_matrix_from_filtered
from models import build_model


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
# Per-fold fit + predict, dispatched by task type
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



def fit_predict_both(model_cfg: dict, X_tr, y_tr, X_ev, random_state: int):
    """Fit once and predict on the eval split.

    Returns (pred_eval, info_trace). The second element is non-empty only for
    models that expose get_info_trace(); the current models do not, so it is
    normally empty.

    Training-split predictions are deliberately not produced: they existed only
    to feed the generalization-gap figures, which have been removed.
    """
    params = dict(model_cfg.get("params", {}))
    params["random_state"] = random_state
    model = build_model(model_cfg["class"], **params)
    model.fit(X_tr, y_tr)
    ttype = model_cfg["task_type"]
    pred_ev = _predict_one(model, ttype, X_ev)
    info_trace = model.get_info_trace() if hasattr(model, "get_info_trace") else []
    return pred_ev, info_trace


def fit_predict_joint_yw(model_cfg: dict, X_tr, y_tr, w_tr, X_ev,
                         random_state: int):
    """Fit a joint Y+W latent model once and predict both tasks."""
    params = dict(model_cfg.get("params", {}))
    params["random_state"] = random_state
    model = build_model(model_cfg["class"], **params)
    model.fit(X_tr, y_tr, w_tr)
    pred_ev = model.predict(X_ev)
    info_trace = model.get_info_trace() if hasattr(model, "get_info_trace") else []
    return pred_ev, info_trace


# ---------------------------------------------------------------------------
# The 10-split train/test experiment
# ---------------------------------------------------------------------------
# run_main_experiment() is deliberately thin: it assembles the per-run context,
# walks the folds, and delegates. Each helper below owns exactly one concern and
# appends to the shared `acc` accumulator, so no state is hidden in a closure.
@dataclass
class ExperimentContext:
    """Everything the per-fold helpers need, assembled once per run."""
    task_lookup: dict                 # task name -> (config, labels, mask)
    models_for: dict                  # task_type -> [model config]
    joint_models: list                # models with task_type == joint_y_w
    X: np.ndarray                     # pair feature matrix
    cutoff_grid: np.ndarray
    logit_eps: float                  # clip for the regression logit transform
    base_seed: int
    t0: float                         # run start, for elapsed-time logging


def _new_accumulator() -> dict:
    """Rows collected across folds, plus fit counters for the progress line."""
    return {"pred": [], "met": [], "cut": [], "info": [],
            "n_fits": 0, "fit_seconds": 0.0}


def build_task_lookup(cfg: dict, labels_df: pd.DataFrame) -> dict:
    """task name -> (task config, label vector, observation mask), id-ordered."""
    lookup = {}
    for task in cfg["tasks"]:
        sub = labels_df[labels_df["task"] == task["name"]].sort_values("sample_id")
        ttype = task.get("task_type", "binary")
        y = (sub["y"].to_numpy().astype(int) if ttype == "binary"
             else sub["y"].to_numpy().astype(float))
        lookup[task["name"]] = (task, y, sub["mask"].to_numpy().astype(int))
    return lookup


def _log_fit(acc: dict, ctx: ExperimentContext, seconds: float, what: str,
             outer_fold, inner_fold, n_tr: int, n_ev: int) -> None:
    """One progress line per fit, with a running average and elapsed total."""
    acc["n_fits"] += 1
    acc["fit_seconds"] += seconds
    _p(f"[main] fit {acc['n_fits']} | outer={outer_fold} inner={inner_fold} "
       f"| {what} | n_tr={n_tr} n_ev={n_ev} "
       f"| {seconds:.1f}s (avg {acc['fit_seconds'] / acc['n_fits']:.1f}s, "
       f"elapsed {(time.time() - ctx.t0) / 60:.1f}m)")


def _add_trace(acc: dict, trace, outer_fold, inner_fold, task: str, model: str) -> None:
    """Optional per-fit diagnostics; empty for every current model."""
    for row in trace or []:
        acc["info"].append({**dict(row),
                            "outer_fold": int(outer_fold), "inner_fold": int(inner_fold),
                            "role": "train_trace", "task": task, "model": model})


def _add_cutoff_sweep(acc: dict, ctx: ExperimentContext, y_true, prob,
                      outer_fold, inner_fold, role: str, task: str, model: str) -> None:
    """Full threshold curve for one binary fit (binary tasks only)."""
    cs = cutoff_sweep(y_true, prob, ctx.cutoff_grid)
    cs.insert(0, "model", model)
    cs.insert(0, "task", task)
    cs.insert(0, "role", role)
    cs.insert(0, "inner_fold", int(inner_fold))
    cs.insert(0, "outer_fold", int(outer_fold))
    acc["cut"].append(cs)


def _tag(outer_fold, inner_fold, role: str, task: str, ttype: str, model: str) -> dict:
    """Identifying columns, in the exact order every downstream CSV expects."""
    return {"outer_fold": int(outer_fold), "inner_fold": int(inner_fold),
            "role": role, "task": task, "task_type": ttype, "model": model}


def _score(y_true, pred, ttype: str) -> dict:
    """Metric set for one fit, chosen by task type."""
    return basic_metrics(y_true, pred) if ttype == "binary" else regression_metrics(y_true, pred)


def _eligible(y_train, ttype: str) -> bool:
    """A fold is skipped unless the target can actually be fit on it."""
    if ttype == "binary":
        return len(np.unique(y_train)) >= 2
    return np.var(y_train) != 0


def _run_single_task_models(cfg, ctx, acc, fold) -> None:
    """Every single-task model, on every task it targets, for one fold."""
    outer_fold, inner_fold, train_idx, eval_idx, eval_role = fold

    for task in cfg["tasks"]:
        name = task["name"]
        tcfg, y_all, mask_all = ctx.task_lookup[name]
        ttype = tcfg.get("task_type", "binary")

        tr = train_idx[mask_all[train_idx] == 1]
        ev = eval_idx[mask_all[eval_idx] == 1]
        if len(tr) == 0 or len(ev) == 0 or not _eligible(y_all[tr], ttype):
            continue

        X_tr, X_ev = ctx.X[tr], ctx.X[ev]
        y_tr, y_ev = y_all[tr], y_all[ev]

        # Regression only: fit in transformed space, score on the original scale.
        ttransform = tcfg.get("transform", "none") if ttype == "regression" else "none"
        y_tr_fit = target_forward(y_tr, ttransform, ctx.logit_eps)

        for m in ctx.models_for[ttype]:
            # target_task pins a model to one task; without it a model runs on
            # every task of its type (the baseline sweep).
            if m.get("target_task") and m["target_task"] != name:
                continue

            seed = derive_fold_seed(ctx.base_seed, int(outer_fold), int(inner_fold))
            tick = time.time()
            pred, trace = fit_predict_both(
                m, X_tr, y_tr_fit, X_ev, random_state=seed)
            _add_trace(acc, trace, outer_fold, inner_fold, name, m["name"])

            pred = target_inverse(pred, ttransform)
            _log_fit(acc, ctx, time.time() - tick,
                     f"{name}/{m['name']}{' [logit]' if ttransform == 'logit' else ''} ",
                     outer_fold, inner_fold, len(tr), len(ev))

            acc["pred"].append(pd.DataFrame({
                **_tag(outer_fold, inner_fold, eval_role, name, ttype, m["name"]),
                "sample_id": ev, "y_true": y_ev,
                "prob":   pred if ttype == "binary" else np.nan,
                "y_pred": pred if ttype == "regression" else np.nan,
            }))
            acc["met"].append({**_tag(outer_fold, inner_fold, eval_role, name, ttype, m["name"]),
                               **_score(y_ev, pred, ttype)})
            if ttype == "binary":
                _add_cutoff_sweep(acc, ctx, y_ev, pred, outer_fold, inner_fold,
                                  eval_role, name, m["name"])


def _run_joint_models(cfg, ctx, acc, fold) -> None:
    """Joint Y+W models for one fold: fit once, emit metrics for both tasks.

    Trained on the rows where BOTH targets are observed. The W head is a
    classifier (w_class), trained with BCE, so it returns a probability.
    """
    outer_fold, inner_fold, train_idx, eval_idx, eval_role = fold
    if not (ctx.joint_models and "y" in ctx.task_lookup and "w_class" in ctx.task_lookup):
        return

    _, y_all, y_mask = ctx.task_lookup["y"]
    _, w_all, w_mask = ctx.task_lookup["w_class"]
    tr = train_idx[(y_mask[train_idx] == 1) & (w_mask[train_idx] == 1)]
    ev = eval_idx[(y_mask[eval_idx] == 1) & (w_mask[eval_idx] == 1)]
    if not (len(tr) and len(ev)
            and len(np.unique(y_all[tr])) >= 2 and len(np.unique(w_all[tr])) >= 2):
        return

    X_tr, X_ev = ctx.X[tr], ctx.X[ev]
    y_tr, y_ev = y_all[tr].astype(int), y_all[ev].astype(int)
    w_tr, w_ev = w_all[tr].astype(int), w_all[ev].astype(int)

    for m in ctx.joint_models:
        seed = derive_fold_seed(ctx.base_seed, int(outer_fold), int(inner_fold))
        tick = time.time()
        pred_ev, trace = fit_predict_joint_yw(
            m, X_tr, y_tr, w_tr.astype(float), X_ev, random_state=seed)
        _log_fit(acc, ctx, time.time() - tick, f"joint(Y,Wclass)/{m['name']} ",
                 outer_fold, inner_fold, len(tr), len(ev))
        _add_trace(acc, trace, outer_fold, inner_fold, "y+w_class", m["name"])

        # (task name, eval truth, eval prob)
        heads = (
            ("y",       y_ev, pred_ev["y_prob"]),
            ("w_class", w_ev, pred_ev["w_pred"]),
        )
        for name, y_true, prob in heads:
            acc["pred"].append(pd.DataFrame({
                **_tag(outer_fold, inner_fold, eval_role, name, "binary", m["name"]),
                "sample_id": ev, "y_true": y_true, "prob": prob, "y_pred": np.nan}))
            acc["met"].append({**_tag(outer_fold, inner_fold, eval_role, name, "binary", m["name"]),
                               **basic_metrics(y_true, prob)})
            _add_cutoff_sweep(acc, ctx, y_true, prob, outer_fold, inner_fold,
                              eval_role, name, m["name"])


def _write_outputs(cfg, args, acc, base_seed, pred_dir, met_dir, t0) -> None:
    """Collate the accumulated rows and write every artefact."""
    models_run = [m["name"] for m in cfg["models"]]
    append = bool(getattr(args, "append", False))

    preds = pd.concat(acc["pred"], ignore_index=True)
    _write_or_append(preds, pred_dir / "predictions.csv", models_run, append)

    metrics = pd.DataFrame(acc["met"])
    _write_or_append(metrics, met_dir / "metrics_by_fold.csv", models_run, append)

    if acc["info"]:
        _write_or_append(pd.DataFrame(acc["info"]),
                         met_dir / "info_theory_trace.csv", models_run, append)

    n_cut = 0
    if acc["cut"]:
        cutoffs = pd.concat(acc["cut"], ignore_index=True)
        _write_or_append(cutoffs, met_dir / "cutoff_sweep.csv", models_run, append)
        n_cut = len(cutoffs)

    write_json({
        "run_id":        args.run_id,
        "base_seed":     base_seed,
        "n_predictions": int(len(preds)),
        "n_metrics":     int(len(metrics)),
        "n_cutoff_rows": int(n_cut),
        "n_info_rows":   int(len(acc["info"])),
        "elapsed_sec":   round(time.time() - t0, 2),
        "tasks":  [t["name"] for t in cfg["tasks"]],
        "models": models_run,
    }, met_dir / "predict_log.json")

    print(f"[experiment] DONE in {time.time() - t0:.1f}s")
    print(f"[experiment]   base_seed            : {base_seed}")
    print(f"[experiment]   predictions.csv      : {len(preds)} rows")
    print(f"[experiment]   metrics_by_fold.csv  : {len(metrics)} rows")
    print(f"[experiment]   cutoff_sweep.csv     : {n_cut} rows")
    if acc["info"]:
        print(f"[experiment]   info_theory_trace.csv: {len(acc['info'])} rows")


def _iter_folds(splits_df: pd.DataFrame, verbosity: int):
    """Yield (outer, inner, train_idx, eval_idx, eval_role) per split."""
    pairs = (splits_df[["outer_fold", "inner_fold"]]
             .drop_duplicates()
             .sort_values(["outer_fold", "inner_fold"])
             .to_records(index=False))
    for outer_fold, inner_fold in pairs:
        sub = splits_df[(splits_df["outer_fold"] == outer_fold) &
                        (splits_df["inner_fold"] == inner_fold)]
        train_idx = sub.loc[sub["role"] == "train", "sample_id"].to_numpy()
        eval_role = "test" if inner_fold == -1 else "val"
        eval_idx = sub.loc[sub["role"] == eval_role, "sample_id"].to_numpy()

        if verbosity >= 1 and inner_fold == -1:
            print(f"[experiment] outer={outer_fold} (outer-only test fit)")
        if verbosity >= 2:
            print(f"[experiment]   outer={outer_fold} inner={inner_fold} "
                  f"({eval_role} n={len(eval_idx)})")
        yield outer_fold, inner_fold, train_idx, eval_idx, eval_role


def run_main_experiment(cfg: dict, args, data, labels_df, splits_df, X, base_seed):
    """Fit every configured model on every fold and write the results."""
    ev_cfg = cfg.get("evaluation", {})
    ctx = ExperimentContext(
        task_lookup=build_task_lookup(cfg, labels_df),
        models_for={
            "binary":     [m for m in cfg["models"] if m.get("task_type") == "binary"],
            "regression": [m for m in cfg["models"] if m.get("task_type") == "regression"],
        },
        joint_models=[m for m in cfg["models"] if m.get("task_type") == "joint_y_w"],
        X=X,
        cutoff_grid=np.arange(cfg["evaluation"]["cutoff_grid_start"],
                              cfg["evaluation"]["cutoff_grid_stop"] + 1e-9,
                              cfg["evaluation"]["cutoff_grid_step"]),
        logit_eps=float(ev_cfg.get("logit_eps", 1e-3)),
        base_seed=base_seed,
        t0=time.time(),
    )
    acc = _new_accumulator()
    verbosity = cfg.get("logging", {}).get("verbosity", 1)

    n_splits = splits_df[["outer_fold", "inner_fold"]].drop_duplicates().shape[0]
    _p(f"[main] starting 10-split train/test evaluation: {n_splits} split(s), "
       f"models={[m['name'] for m in cfg['models']]}")

    for fold in _iter_folds(splits_df, verbosity):
        _run_single_task_models(cfg, ctx, acc, fold)
        _run_joint_models(cfg, ctx, acc, fold)

    if not acc["pred"]:
        print("[experiment] No predictions produced. Check splits / labels / masks.")
        return
    _write_outputs(cfg, args, acc, base_seed,
                   predictions_dir(args.run_id), metrics_dir(args.run_id), ctx.t0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--run-id", required=True)
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

    combine = str(cfg.get("preprocess", {}).get("combine", "concat"))
    X = pair_matrix_from_filtered(data, combine=combine)
    print(f"[experiment] feature matrix: {X.shape}  (combine={combine})")

    # skip-if-already-saved: if a block's output CSV exists, skip it
    # (use --force to recompute). Scoped to this run_id's metrics folder.
    _mdir = metrics_dir(args.run_id)
    def _saved(fname):
        return (not args.force) and (_mdir / fname).exists()

    # an append / --only-models run must always re-run.
    if (args.append or args.only_models) or not _saved("metrics_by_fold.csv"):
        print("\n[experiment] ### 10-split train/test experiment ###")
        run_main_experiment(cfg, args, data, labels_df, splits_df, X, base_seed)
    else:
        print("\n[experiment] ### SKIPPED (already saved) ###")



if __name__ == "__main__":
    main()
