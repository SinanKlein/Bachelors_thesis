"""10-fold train/test experiment and out-of-fold latent export.

  python experiment.py --config config.yaml --run-id <run_id>

Writes:
  predictions/predictions.csv
  metrics/metrics_by_fold.csv, cutoff_sweep.csv, lambda_cv.csv, lambda_cv_curve.csv
  latent/<model>/latent_space.csv.gz    models with export_latent: true
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from common import (binary_metrics, cutoff_sweep, fold_seed, iter_folds, load_config,
                    load_run, regression_metrics, run_dir, set_global_seeds, stage_args,
                    write_csv, write_json)
from models import build_model


def logit(y, eps):
    y = np.clip(np.asarray(y, dtype=float), eps, 1.0 - eps)
    return np.log(y / (1.0 - y))


def expit(p):
    return 1.0 / (1.0 + np.exp(-np.asarray(p, dtype=float)))


class Collector:
    """Rows for every output table, plus the latent vectors per model."""

    def __init__(self, grid):
        self.grid = grid
        self.pred, self.met, self.cut, self.lam, self.curve = [], [], [], [], []
        self.latent = {}

    def add(self, fold, task, ttype, model, ids, y_true, pred):
        tag = {"outer_fold": fold, "task": task, "task_type": ttype, "model": model}
        binary = ttype == "binary"
        self.pred.append(pd.DataFrame({**tag, "sample_id": ids, "y_true": y_true,
                                       "prob": pred if binary else np.nan,
                                       "y_pred": np.nan if binary else pred}))
        self.met.append({**tag, **(binary_metrics(y_true, pred) if binary
                                   else regression_metrics(y_true, pred))})
        if binary:
            cs = cutoff_sweep(y_true, pred, self.grid)
            for col, val in (("model", model), ("task", task), ("outer_fold", fold)):
                cs.insert(0, col, val)
            self.cut.append(cs)

    def add_lambda(self, model, fold, task, name):
        if model.cv_record:
            tag = {"outer_fold": fold, "task": task, "model": name}
            self.lam.append({**model.cv_record, **tag})
            self.curve += [{**row, **tag} for row in model.cv_curve]

    def add_latent(self, name, ids, z, **preds):
        self.latent.setdefault(name, []).append(
            pd.DataFrame({"sample_id": ids, **preds, "_z": list(z)}))


def run(cfg: dict, run_id: str) -> None:
    t0 = time.time()
    seed = int(cfg["resampling"]["seed"])
    set_global_seeds(seed)
    data, X, tasks, splits = load_run(run_id, cfg)
    ev_cfg = cfg["evaluation"]
    eps = float(ev_cfg.get("logit_eps", 1e-3))
    col = Collector(np.arange(ev_cfg["cutoff_grid_start"], ev_cfg["cutoff_grid_stop"] + 1e-9,
                              ev_cfg["cutoff_grid_step"]))
    single = [m for m in cfg["models"] if m["task_type"] != "joint_y_w"]
    joint = [m for m in cfg["models"] if m["task_type"] == "joint_y_w"]
    print(f"[experiment] X {X.shape} | models {[m['name'] for m in cfg['models']]}")

    def fit(m, fold, X_tr, *targets):
        tick = time.time()
        model = build_model(m["class"], **{**m.get("params", {}),
                                           "random_state": fold_seed(seed, fold)})
        model.fit(X_tr, *targets)
        print(f"[experiment] fold {fold} | {m['name']} | n_tr={len(X_tr)} | {time.time() - tick:.1f}s")
        return model

    for fold, train_ids, test_ids in iter_folds(splits):
        # Single-task models, on every task of their type (or only target_task).
        for task_name, (t, y, mask) in tasks.items():
            ttype = t["task_type"]
            tr, ev = train_ids[mask[train_ids] == 1], test_ids[mask[test_ids] == 1]
            if not len(tr) or not len(ev) or (
                    len(np.unique(y[tr])) < 2 if ttype == "binary" else np.var(y[tr]) == 0):
                continue
            to_logit = ttype == "regression" and t.get("transform") == "logit"
            y_fit = logit(y[tr], eps) if to_logit else y[tr]
            for m in single:
                if m["task_type"] != ttype or m.get("target_task", task_name) != task_name:
                    continue
                model = fit(m, fold, X[tr], y_fit)
                pred = model.predict_proba(X[ev]) if ttype == "binary" else model.predict(X[ev])
                pred = expit(pred) if to_logit else np.asarray(pred, dtype=float)
                col.add(fold, task_name, ttype, m["name"], ev, y[ev], pred)
                if hasattr(model, "cv_record"):
                    col.add_lambda(model, fold, task_name, m["name"])
                if m.get("export_latent"):
                    key = "w_pred" if task_name == "w_class" else "y_prob"
                    col.add_latent(m["name"], ev, model.encode_latent(X[ev]), **{key: pred})

        # Joint y + w_class models, on pairs where both are observed.
        if joint and "y" in tasks and "w_class" in tasks:
            _, y, y_mask = tasks["y"]
            _, w, w_mask = tasks["w_class"]
            both = (y_mask == 1) & (w_mask == 1)
            tr, ev = train_ids[both[train_ids]], test_ids[both[test_ids]]
            if len(tr) and len(ev) and len(np.unique(y[tr])) >= 2 and len(np.unique(w[tr])) >= 2:
                for m in joint:
                    model = fit(m, fold, X[tr], y[tr].astype(int), w[tr].astype(float))
                    out = model.predict(X[ev])
                    col.add(fold, "y", "binary", m["name"], ev, y[ev].astype(int), out["y_prob"])
                    col.add(fold, "w_class", "binary", m["name"], ev, w[ev].astype(int), out["w_pred"])
                    if m.get("export_latent"):
                        col.add_latent(m["name"], ev, model.encode_latent(X[ev]),
                                       y_prob=np.asarray(out["y_prob"], dtype=float),
                                       w_pred=np.asarray(out["w_pred"], dtype=float))

    met_dir = run_dir(run_id, "metrics")
    write_csv(pd.concat(col.pred, ignore_index=True), run_dir(run_id, "predictions") / "predictions.csv")
    write_csv(pd.DataFrame(col.met), met_dir / "metrics_by_fold.csv")
    if col.cut:
        write_csv(pd.concat(col.cut, ignore_index=True), met_dir / "cutoff_sweep.csv")
    if col.lam:
        write_csv(pd.DataFrame(col.lam), met_dir / "lambda_cv.csv")
        write_csv(pd.DataFrame(col.curve), met_dir / "lambda_cv_curve.csv")
    for name, parts in col.latent.items():
        export_latent(run_id, name, pd.concat(parts), data, tasks)
    write_json({"run_id": run_id, "base_seed": seed, "elapsed_sec": round(time.time() - t0, 2),
                "tasks": list(tasks), "models": [m["name"] for m in cfg["models"]]},
               met_dir / "predict_log.json")
    print(f"[experiment] done in {time.time() - t0:.1f}s")


def export_latent(run_id, name, df, data, tasks) -> None:
    """One row per test pair: ids, labels, predictions, mu_* and agreement class.

    Each fold trains its own network, so hidden units are not aligned across folds.
    """
    df = df.sort_values("sample_id").reset_index(drop=True)
    ids = df["sample_id"].to_numpy()
    bi = np.asarray(data["bact_idx"]).ravel().astype(int)[ids]
    vi = np.asarray(data["virus_idx"]).ravel().astype(int)[ids]
    out = pd.DataFrame({
        "sample_id": ids, "latent_model": name,
        "bact_idx": bi, "bact_id": np.asarray(data["bact_ids"]).astype(str)[bi],
        "virus_idx": vi, "virus_id": np.asarray(data["virus_ids"]).astype(str)[vi],
        "y": tasks["y"][1][ids], "w": tasks["w_class"][1][ids],
        "w_reg": tasks["w_reg"][1][ids], "w_mask": tasks["w_reg"][2][ids],
    })
    for key in ("y_prob", "w_pred"):
        if key in df:
            out[key] = df[key].to_numpy()
    Z = np.vstack(df["_z"].to_numpy()).astype(float)
    out = pd.concat([out, pd.DataFrame(Z, columns=[f"mu_{j + 1}" for j in range(Z.shape[1])])], axis=1)
    y_pos, w_pos = out["y"] == 1, out["w"] == 1
    out["agreement_class"] = np.select([y_pos & w_pos, y_pos & ~w_pos, ~y_pos & w_pos],
                                       ["Y=1 & W=1", "Y=1 only", "W=1 only"], default="background")
    d = run_dir(run_id, "latent", name)
    out.to_csv(d / "latent_space.csv.gz", index=False, compression="gzip")
    print(f"[experiment] latent {name}: {len(out)} rows x {Z.shape[1]} dims -> {d}")


if __name__ == "__main__":
    args = stage_args().parse_args()
    run(load_config(args.config), args.run_id)
