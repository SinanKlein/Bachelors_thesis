"""
Export learned deterministic two-tower latent interaction spaces for thesis
interpretation.

The main experiment keeps all classical baselines on the same concatenated input
X = [Xb, Xv].  The two-tower models use the same X but split it internally into
bacterial and viral ProC blocks.  The exported pair-level latent space is the
Hadamard product z_ij = h_b(i) * h_v(j), whose sum is the dot-product
compatibility score.

Writes under RESULTS_DIR / DATASET_NAME / <run_id> / latent / <model_name>:
  latent_space.csv.gz              one row per encoded edge/pair
  feature_pc_correlations.csv      ProC feature correlations with latent PCs
  pca_loadings.csv                 latent-dimension loadings for the PCs
  latent_export_log.json           run summary

Usage:
  python export_latent_space.py --config default.yaml --run-id <run_id>
  python export_latent_space.py --config default.yaml --run-id <run_id> --all-three
  python export_latent_space.py --config default.yaml --run-id <run_id> --model twotower_yw_joint
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
from utils import load_config, load_npz, write_csv, write_json, set_global_seeds
from experiment import build_feature_matrix, build_feature_labels, target_forward, target_inverse
from models import build_model


LATENT_MODELS = ["twotower_y_only", "twotower_w_only", "twotower_yw_joint", "mlp_yw_joint"]


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
    """Return pair latent vectors for two-tower models; retain VIB compatibility."""
    if hasattr(model, "encode_latent"):
        z = model.encode_latent(X, batch_size=batch_size)
        return np.asarray(z, dtype=float), np.zeros_like(z, dtype=float)

    # Backward compatibility for old VIB wrappers if someone calls the script
    # with a vib_* model manually.
    import torch
    if not hasattr(model, "_model") or not hasattr(model._model, "encode"):
        raise TypeError("The fitted model does not expose encode_latent() or _model.encode().")
    Xs = model._scaler.transform(X).astype(np.float32)
    device = model._device
    mus = []
    logvars = []
    model._model.eval()
    with torch.no_grad():
        for start in range(0, Xs.shape[0], batch_size):
            xb = torch.as_tensor(Xs[start:start + batch_size], dtype=torch.float32, device=device)
            mu, logvar = model._model.encode(xb)
            mus.append(mu.cpu().numpy())
            logvars.append(logvar.cpu().numpy())
    return np.vstack(mus), np.vstack(logvars)


def _predict_for_export(model, model_type: str, X: np.ndarray, w_transform: str, logit_eps: float):
    y_prob = None
    w_pred = None
    if model_type == "joint_y_w":
        pred = model.predict(X)
        y_prob = np.asarray(pred["y_prob"], dtype=float)
        w_pred = np.asarray(pred["w_pred"], dtype=float)  # W head is a classifier -> probability
    elif model_type == "binary":
        y_prob = np.asarray(model.predict_proba(X), dtype=float)
    elif model_type == "regression":
        w_pred = target_inverse(np.asarray(model.predict(X), dtype=float), w_transform)
    return y_prob, w_pred


def _safe_corr_with_columns(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    y_center = y - np.nanmean(y)
    y_sd = np.sqrt(np.nansum(y_center ** 2))
    X_mean = np.nanmean(X, axis=0)
    X_center = X - X_mean
    X_sd = np.sqrt(np.nansum(X_center ** 2, axis=0))
    denom = X_sd * y_sd
    num = np.nansum(X_center * y_center[:, None], axis=0)
    out = np.full(X.shape[1], np.nan, dtype=float)
    ok = denom > 0
    out[ok] = num[ok] / denom[ok]
    return out


def _export_index(model_type, y_mask, w_mask):
    """Eligible pairs for a latent model, matching the main experiment's masks."""
    if model_type == "joint_y_w":
        return np.where((y_mask == 1) & (w_mask == 1))[0]
    if model_type == "binary":
        return np.where(y_mask == 1)[0]
    if model_type == "regression":
        return np.where(w_mask == 1)[0]
    raise ValueError(f"Unsupported task_type '{model_type}'.")


def _fit_model_on_train(model_cfg, X, tr_rows, y_all, w_all, w_cls_all, w_transform, logit_eps, seed):
    """Fit the latent model on an explicit training-row set (leakage-safe).

    Used per outer fold so the latent for any pair is produced by a model that
    never trained on that pair's label.
    """
    model_type = model_cfg.get("task_type")
    params = dict(model_cfg.get("params", {}))
    params["random_state"] = seed
    model = build_model(model_cfg["class"], **params)
    if model_type == "joint_y_w":
        y_fit = y_all[tr_rows].astype(int)
        w_fit = w_cls_all[tr_rows].astype(float)   # joint W head is a classifier (w_class)
        model.fit(X[tr_rows], y_fit, w_fit)
    elif model_type == "binary":
        model.fit(X[tr_rows], y_all[tr_rows].astype(int))
    elif model_type == "regression":
        w_fit = target_forward(w_all[tr_rows].astype(float), w_transform, logit_eps)
        model.fit(X[tr_rows], w_fit)
    else:
        raise ValueError(f"Model '{model_cfg.get('name')}' has unsupported task_type '{model_type}'.")
    return model


def _export_one(cfg, run_id, model_name, out_dir, data, labels_df, X, feature_labels, splits, base_seed, n_components, force):
    t0 = time.time()
    model_cfg = _find_model_cfg(cfg, model_name)
    model_type = model_cfg.get("task_type")
    params = dict(model_cfg.get("params", {}))

    # Inject Xb width for two-tower models. Baselines remain concatenated; this
    # only tells the two-tower where to split [Xb, Xv] internally.
    if str(model_cfg.get("class", "")).startswith("twotower_"):
        params["xb_dim"] = int(np.asarray(data["Xb_filtered"]).shape[1])
        model_cfg = dict(model_cfg)
        model_cfg["params"] = params

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
    idx = _export_index(model_type, y_mask, w_mask)
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
        model = _fit_model_on_train(model_cfg, X, tr_rows, y_all, w_all, w_cls_all,
                                    w_transform, logit_eps, base_seed + int(outer_fold))
        z_te, _lv_te = _encode_latent(model, X[te_rows], batch_size=batch_size)
        yp_te, wp_te = _predict_for_export(model, model_type, X[te_rows], w_transform, logit_eps)
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

    from sklearn.decomposition import PCA
    n_comp = int(min(n_components, z.shape[1]))
    pca = PCA(n_components=n_comp, random_state=base_seed)
    pcs = pca.fit_transform(z)

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
    latent_df["w"] = w_cls_all[idx].astype(int)         # W is now a class label (1[W > 0.5])
    latent_df["w_reg"] = w_all[idx].astype(float)       # raw continuous W, kept for reference
    latent_df["w_mask"] = w_mask[idx].astype(int)
    if y_prob is not None:
        latent_df["y_prob"] = y_prob
    if w_pred is not None:
        latent_df["w_pred"] = w_pred
    for j in range(z.shape[1]):
        latent_df[f"mu_{j+1}"] = z[:, j]
        latent_df[f"logvar_{j+1}"] = logvar[:, j]
    for j in range(n_comp):
        latent_df[f"PC{j+1}"] = pcs[:, j]

    high_w = latent_df["w"] == 1
    y_pos = latent_df["y"] == 1
    latent_df["agreement_class"] = np.select(
        [y_pos & high_w, y_pos & ~high_w, ~y_pos & high_w],
        ["Y=1 & W=1", "Y=1 only", "W=1 only"],
        default="background",
    )
    latent_df.to_csv(model_out / "latent_space.csv.gz", index=False, compression="gzip")

    loadings = []
    for pc_i in range(n_comp):
        for dim_j in range(z.shape[1]):
            loadings.append({"pc": f"PC{pc_i+1}", "latent_dim": f"mu_{dim_j+1}", "loading": float(pca.components_[pc_i, dim_j])})
    write_csv(pd.DataFrame(loadings), model_out / "pca_loadings.csv")

    X_export = X[idx]
    feat_df = feature_labels.copy()
    for pc_i in range(n_comp):
        corr = _safe_corr_with_columns(X_export, pcs[:, pc_i])
        feat_df[f"cor_PC{pc_i+1}"] = corr
        feat_df[f"abs_cor_PC{pc_i+1}"] = np.abs(corr)
    write_csv(feat_df, model_out / "feature_pc_correlations.csv")

    write_json({
        "run_id": run_id,
        "model": model_name,
        "model_type": model_type,
        "n_rows_encoded": int(len(idx)),
        "n_features": int(X.shape[1]),
        "xb_dim": int(np.asarray(data["Xb_filtered"]).shape[1]),
        "latent_dim": int(z.shape[1]),
        "n_pca_components": int(n_comp),
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "encoding": "out_of_fold",
        "elapsed_sec": round(time.time() - t0, 2),
        "note": "Latents are out-of-fold: each pair is encoded by a model trained only on that pair's outer-fold training set, so probe decodability is leakage-free. mu_* are deterministic latent coordinates (z=h_b*h_v for two-tower; post-ReLU hidden activation for mlp_*); logvar_* is zero-filled only for compatibility with the old plotting code.",
    }, model_out / "latent_export_log.json")
    _p(f"[latent] {model_name}: wrote {model_out} in {time.time() - t0:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", default="twotower_yw_joint", help="Latent model name from default.yaml.")
    ap.add_argument("--all-three", action="store_true", help="Export twotower_y_only, twotower_w_only, and twotower_yw_joint.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing latent exports.")
    ap.add_argument("--n-components", type=int, default=2, help="Number of latent PCs to export. Default: 2")
    args = ap.parse_args()

    cfg = load_config(args.config)
    base_seed = int(cfg.get("resampling", {}).get("seed", 42))
    set_global_seeds(base_seed)

    pre_dir = preprocessing_dir(args.run_id)
    _p(f"[latent] loading preprocessing outputs from {pre_dir}")
    data = load_npz(pre_dir / "data_filtered.npz")
    labels_df = pd.read_csv(pre_dir / "labels.csv")
    splits = pd.read_csv(pre_dir / "splits.csv")
    X = build_feature_matrix(data)
    feature_labels = build_feature_labels(data)
    _p(f"[latent] feature matrix: {X.shape}")

    out_dir = run_dir(args.run_id) / "latent"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_names = LATENT_MODELS if args.all_three else [args.model]
    for model_name in model_names:
        _export_one(cfg, args.run_id, model_name, out_dir, data, labels_df, X, feature_labels,
                    splits, base_seed, args.n_components, args.force)
    _p("[latent] DONE")


if __name__ == "__main__":
    main()
