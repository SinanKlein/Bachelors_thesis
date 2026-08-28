"""
latent.py — export the MLP latent interaction spaces.

Each latent MLP is refit per outer fold and the post-ReLU hidden activation of
its trunk is taken as the pair's latent vector. Latents are produced
OUT-OF-FOLD — every pair is encoded by a model that never saw it in training.
The forward path used for prediction is untouched; this only taps it.

Three spaces, each pinned to its own task:
  mlp_latent_y        trained on Y
  mlp_latent_w_class  trained on w_class
  mlp_latent_yw       joint (Y + w_class), one shared trunk with two heads

Writes under <run>/latent/<model_name>/: latent_space.csv.gz and the export log.

Usage:
  python latent.py --config default.yaml --run-id <run_id>
  python latent.py --config default.yaml --run-id <run_id> --model mlp_latent_yw
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

from paths import preprocessing_dir, run_dir
from utils import (load_config, load_npz, write_json,
                   set_global_seeds, derive_fold_seed)
from features import pair_matrix_from_filtered
from experiment import target_forward, target_inverse
from models import build_model


LATENT_MODELS = ["mlp_latent_y", "mlp_latent_w_class", "mlp_latent_yw"]


def _p(msg: str) -> None:
    print(msg, flush=True)


def _task_arrays(labels_df: pd.DataFrame, task_name: str, task_type: str):
    sub = labels_df[labels_df["task"] == task_name].sort_values("sample_id")
    if sub.empty:
        raise KeyError(f"Task '{task_name}' not found in labels.csv")
    y = sub["y"].to_numpy().astype(int if task_type == "binary" else float)
    mask = sub["mask"].to_numpy().astype(int)
    sample_id = sub["sample_id"].to_numpy().astype(int)
    if not np.array_equal(sample_id, np.arange(len(sample_id))):
        raise ValueError("labels.csv sample_id is not 0..n-1 after sorting; cannot align safely.")
    return y, mask


def _find_model_cfg(cfg: dict, model_name: str) -> dict:
    for m in cfg.get("models", []):
        if m.get("name") == model_name:
            return m
    raise KeyError(f"Model '{model_name}' not found in config models.")


def _encode_latent(model, X: np.ndarray, batch_size: int = 4096):
    """Return the latent vectors for a model via its encode_latent() method.

    The second return value is a zero array kept only for signature compatibility
    with the old plotting code (there is no stochastic logvar in these MLPs).
    """
    if not hasattr(model, "encode_latent"):
        raise TypeError(f"{type(model).__name__} does not expose encode_latent().")
    z = model.encode_latent(X, batch_size=batch_size)
    z = np.asarray(z, dtype=float)
    return z, np.zeros_like(z, dtype=float)


def _model_targets(model_cfg: dict) -> list[str]:
    """Tasks a latent model is trained on and plotted for.

    joint_y_w -> both y and w_class; a binary/regression model is pinned to its
    `target_task` (falling back to y for binary, w_class for regression).
    """
    if model_cfg.get("task_type") == "joint_y_w":
        return ["y", "w_class"]
    tt = model_cfg.get("target_task")
    if tt:
        return [tt]
    return ["y"] if model_cfg.get("task_type") == "binary" else ["w_class"]


def _predict_for_export(model, model_cfg: dict, X: np.ndarray, w_transform: str, logit_eps: float):
    """Route a model's prediction into y_prob and/or w_pred by its target task."""
    y_prob = None
    w_pred = None
    model_type = model_cfg.get("task_type")
    targets = _model_targets(model_cfg)
    if model_type == "joint_y_w":
        pred = model.predict(X)
        y_prob = np.asarray(pred["y_prob"], dtype=float)
        w_pred = np.asarray(pred["w_pred"], dtype=float)  # W head is a classifier -> probability
    elif model_type == "binary":
        prob = np.asarray(model.predict_proba(X), dtype=float)
        if "w_class" in targets:
            w_pred = prob            # this MLP predicts the W class, not Y
        else:
            y_prob = prob
    elif model_type == "regression":
        w_pred = target_inverse(np.asarray(model.predict(X), dtype=float), w_transform)
    return y_prob, w_pred


def _export_index(model_cfg, y_mask, w_mask):
    """Eligible pairs for a latent model, matching the main experiment's masks.

    Each model is encoded over its task's mask (from labels.csv). With `y` masked
    to Mask_observed, y-, w_class-, and joint models all share the same W-observed
    intersection; if `y`'s mask_source is reverted to null, the y latent again
    spans the full grid.
    """
    targets = _model_targets(model_cfg)
    if model_cfg.get("task_type") == "joint_y_w":
        return np.where((y_mask == 1) & (w_mask == 1))[0]
    if "w_class" in targets:
        return np.where(w_mask == 1)[0]
    return np.where(y_mask == 1)[0]


def _fit_model_on_train(model_cfg, X, tr_rows, y_all, w_all, w_cls_all, w_transform, logit_eps, seed):
    """Fit the latent model on an explicit training-row set (leakage-safe).

    Used per outer fold so the latent for any pair is produced by a model that
    never trained on that pair's label.
    """
    model_type = model_cfg.get("task_type")
    targets = _model_targets(model_cfg)
    params = dict(model_cfg.get("params", {}))
    params["random_state"] = seed
    model = build_model(model_cfg["class"], **params)
    if model_type == "joint_y_w":
        y_fit = y_all[tr_rows].astype(int)
        w_fit = w_cls_all[tr_rows].astype(float)   # joint W head is a classifier (w_class)
        model.fit(X[tr_rows], y_fit, w_fit)
    elif model_type == "binary":
        # pinned to y or to w_class depending on target_task
        tgt = w_cls_all if "w_class" in targets else y_all
        model.fit(X[tr_rows], tgt[tr_rows].astype(int))
    elif model_type == "regression":
        w_fit = target_forward(w_all[tr_rows].astype(float), w_transform, logit_eps)
        model.fit(X[tr_rows], w_fit)
    else:
        raise ValueError(f"Model '{model_cfg.get('name')}' has unsupported task_type '{model_type}'.")
    return model


def _export_one(cfg, run_id, model_name, out_dir, data, labels_df, X, splits, base_seed, force):
    t0 = time.time()
    model_cfg = _find_model_cfg(cfg, model_name)
    model_type = model_cfg.get("task_type")

    model_out = out_dir / model_name
    model_out.mkdir(parents=True, exist_ok=True)
    if (model_out / "latent_space.csv.gz").exists() and not force:
        _p(f"[latent] {model_name}: already exported; use --force to recompute.")
        return

    y_all, y_mask = _task_arrays(labels_df, "y", "binary")
    w_all, w_mask = _task_arrays(labels_df, "w_reg", "regression")
    w_cls_all, w_cls_mask = _task_arrays(labels_df, "w_class", "binary")
    logit_eps = float(cfg.get("evaluation", {}).get("logit_eps", 1e-3))
    w_task_cfg = next((t for t in cfg.get("tasks", []) if t.get("name") == "w_reg"), {})
    w_transform = w_task_cfg.get("transform", "none")

    _p(f"[latent] fitting {model_name}")
    model_type = model_cfg.get("task_type")
    idx = _export_index(model_cfg, y_mask, w_mask)
    _p(f"[latent] {model_name}: out-of-fold encoding over {len(idx)} rows")

    sid_to_pos = {int(sid): i for i, sid in enumerate(idx)}
    batch_size = int(dict(model_cfg.get("params", {})).get("batch_size", 512)) * 8

    z = None
    logvar = None
    y_prob = None
    w_pred = None
    filled = np.zeros(len(idx), dtype=bool)

    for outer_fold in sorted(splits["outer_fold"].unique()):
        fold = splits[splits["outer_fold"] == outer_fold]
        tr_ids = fold[fold["role"] == "train"]["sample_id"].to_numpy().astype(int)
        te_ids = fold[fold["role"] == "test"]["sample_id"].to_numpy().astype(int)
        tr_rows = np.array([s for s in tr_ids if int(s) in sid_to_pos], dtype=int)
        te_rows = np.array([s for s in te_ids if int(s) in sid_to_pos], dtype=int)
        if len(tr_rows) == 0 or len(te_rows) == 0:
            continue
        # Same seed derivation as experiment.py, so the refit here reproduces
        # the network whose metrics are reported in metrics_by_fold.csv.
        model = _fit_model_on_train(model_cfg, X, tr_rows, y_all, w_all, w_cls_all,
                                    w_transform, logit_eps,
                                    derive_fold_seed(base_seed, int(outer_fold), -1))
        z_te, _lv_te = _encode_latent(model, X[te_rows], batch_size=batch_size)
        yp_te, wp_te = _predict_for_export(model, model_cfg, X[te_rows], w_transform, logit_eps)
        if z is None:
            z = np.zeros((len(idx), z_te.shape[1]), dtype=float)
            if yp_te is not None:
                y_prob = np.full(len(idx), np.nan, dtype=float)
            if wp_te is not None:
                w_pred = np.full(len(idx), np.nan, dtype=float)
        pos = np.array([sid_to_pos[int(s)] for s in te_rows], dtype=int)
        z[pos] = z_te
        if y_prob is not None and yp_te is not None:
            y_prob[pos] = yp_te
        if w_pred is not None and wp_te is not None:
            w_pred[pos] = wp_te
        filled[pos] = True

    if z is None or not filled.all():
        n_missing = 0 if z is None else int((~filled).sum())
        raise RuntimeError(
            f"[latent] {model_name}: out-of-fold encoding incomplete "
            f"({n_missing} of {len(idx)} rows never appeared in a test fold). "
            f"Splits must partition the samples (StratifiedKFold/KFold)."
        )
    logvar = np.zeros_like(z)

    latent_df = pd.DataFrame({"sample_id": idx.astype(int), "latent_model": model_name})
    if "bact_idx" in data:
        bact_idx = np.asarray(data["bact_idx"]).ravel().astype(int)[idx]
        latent_df["bact_idx"] = bact_idx
        if "bact_ids" in data:
            latent_df["bact_id"] = np.asarray(data["bact_ids"]).astype(str)[bact_idx]
    if "virus_idx" in data:
        virus_idx = np.asarray(data["virus_idx"]).ravel().astype(int)[idx]
        latent_df["virus_idx"] = virus_idx
        if "virus_ids" in data:
            latent_df["virus_id"] = np.asarray(data["virus_ids"]).astype(str)[virus_idx]

    latent_df["y"] = y_all[idx].astype(int)
    latent_df["w"] = w_cls_all[idx].astype(int)         # W is now a class label (1[W > 0])
    latent_df["w_reg"] = w_all[idx].astype(float)       # raw continuous W, kept for reference
    latent_df["w_mask"] = w_mask[idx].astype(int)
    if y_prob is not None:
        latent_df["y_prob"] = y_prob
    if w_pred is not None:
        latent_df["w_pred"] = w_pred
    for j in range(z.shape[1]):
        latent_df[f"mu_{j+1}"] = z[:, j]
        latent_df[f"logvar_{j+1}"] = logvar[:, j]

    high_w = latent_df["w"] == 1
    y_pos = latent_df["y"] == 1
    latent_df["agreement_class"] = np.select(
        [y_pos & high_w, y_pos & ~high_w, ~y_pos & high_w],
        ["Y=1 & W=1", "Y=1 only", "W=1 only"],
        default="background",
    )
    latent_df.to_csv(model_out / "latent_space.csv.gz", index=False, compression="gzip")

    write_json({
        "run_id": run_id,
        "model": model_name,
        "model_type": model_type,
        "target_task": model_cfg.get("target_task"),
        "plot_targets": _model_targets(model_cfg),   # tasks this latent is plotted for
        "n_rows_encoded": int(len(idx)),
        "n_features": int(X.shape[1]),
        "latent_dim": int(z.shape[1]),
        "encoding": "out_of_fold",
        "elapsed_sec": round(time.time() - t0, 2),
        "note": "Latents are out-of-fold: each pair is encoded by a model trained only on that pair's outer-fold training set. mu_* are the post-ReLU hidden activation of the MLP; logvar_* is zero-filled only for compatibility with the old plotting code. plot_targets lists the task(s) this latent space was trained for. Hidden units are NOT aligned across folds (each fold has its own initialisation), so any analysis that pools rows from different folds mixes coordinate systems.",
    }, model_out / "latent_export_log.json")
    _p(f"[latent] {model_name}: wrote {model_out} in {time.time() - t0:.1f}s")


def run_export(cfg: dict, run_id: str, model_names: list[str],
               force: bool = False) -> None:
    """Export the out-of-fold latent space for each requested model."""
    base_seed = int(cfg.get("resampling", {}).get("seed", 42))
    set_global_seeds(base_seed)

    pre_dir = preprocessing_dir(run_id)
    _p(f"[latent] loading preprocessing outputs from {pre_dir}")
    data = load_npz(pre_dir / "data_filtered.npz")
    labels_df = pd.read_csv(pre_dir / "labels.csv")
    splits = pd.read_csv(pre_dir / "splits.csv")
    combine = str(cfg.get("preprocess", {}).get("combine", "concat"))
    X = pair_matrix_from_filtered(data, combine=combine)
    _p(f"[latent] feature matrix: {X.shape}  (combine={combine})")

    out_dir = run_dir(run_id) / "latent"
    out_dir.mkdir(parents=True, exist_ok=True)
    for model_name in model_names:
        _export_one(cfg, run_id, model_name, out_dir, data, labels_df, X,
                    splits, base_seed, force)
    _p("[latent] export DONE")

# Main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", default=None,
                    help="Export a single latent model instead of all three.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    models = [args.model] if args.model else list(LATENT_MODELS)

    print(f"[latent] === EXPORT === {models}")
    run_export(cfg, args.run_id, models, args.force)


if __name__ == "__main__":
    main()
