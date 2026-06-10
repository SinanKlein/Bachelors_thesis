"""
Stage 1: preprocessing.

  1. Load raw graph_data.npz   (must contain Xb, Xv, W_adjacency, Y_adjacency,
                                and W_mask_adjacency)
  2. Apply mean-variance filter globally to Xb and Xv (top quantile by variance)
  3. Generate nested 10x5 CV splits (stratified on the first BINARY task)
  4. Save filtered features + descriptive logs + splits + labels

The binary task targets Y (CRISPR labels). The regression task targets W
(continuous co-abundance probability) on the subset where W is observed.

Usage:
    python preprocess.py --config ../config/default.yaml
    python preprocess.py --config ../config/default.yaml --run-id myrun123
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import GRAPH_DATA_FILE, preprocessing_dir, run_dir
from utils import (
    load_config, load_graph_data, write_csv, write_json, write_npz,
    freeze_config, make_run_id,
    build_reducer,
    make_splits_dataframe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_label(data: dict, source: str, mask_source: str | None,
                  binarize: bool, task_type: str
                  ) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Pull a label vector (and optional mask) from the loaded data dict.

    For binary tasks, returns int labels (0/1).
    For regression tasks, returns float labels unchanged.
    """
    if source not in data:
        raise KeyError(f"Label '{source}' not found in graph data. "
                       f"Available keys: {list(data.keys())}")
    y_raw = np.asarray(data[source]).ravel()

    if task_type == "binary":
        if binarize:
            y = (y_raw > 0).astype(int)
        else:
            y = y_raw.astype(int)
    elif task_type == "regression":
        y = y_raw.astype(float)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    mask = None
    if mask_source is not None:
        if mask_source not in data:
            raise KeyError(f"Mask '{mask_source}' not found in graph data.")
        mask = np.asarray(data[mask_source]).ravel().astype(int)
    return y, mask


def stats_to_long(stats: dict, source: str) -> pd.DataFrame:
    df = pd.DataFrame({
        "source":     source,
        "feature_id": stats["feature_id"],
        "mean":       stats["mean"],
        "var":        stats["var"],
        "score":      stats["score"],
        "selected":   stats["selected"],
    })
    df["threshold"] = stats["threshold"]
    return df


def find_first_binary_task(tasks: list[dict]) -> dict:
    """Splits are stratified on the first binary task."""
    for t in tasks:
        if t.get("task_type", "binary") == "binary":
            return t
    raise ValueError("Cannot stratify splits: no binary task in config.")


def flatten_pair_labels(data: dict) -> dict:
    """
    Convert the (n_bact, n_virus) adjacency matrices into per-pair arrays.

    Reads from `data`:
        W_adjacency       (n_bact, n_virus) float    co-abundance (continuous)
        Y_adjacency       (n_bact, n_virus) int      CRISPR labels (0/1)
        W_mask_adjacency  (n_bact, n_virus) int      1 if W observed, else 0

    Writes to `data` (in place) and returns it:
        bact_idx        (n_pairs,)  int    row index per pair
        virus_idx       (n_pairs,)  int    col index per pair
        Y_binary        (n_pairs,)  int    CRISPR label (target for binary task)
        W_continuous    (n_pairs,)  float  raw W value (target for regression)
        Mask_observed   (n_pairs,)  int    1 where W is observed
                                           (used to mask the regression task)

    Pair ordering: row-major. Pair i has bact_idx[i] = i // n_virus,
    virus_idx[i] = i % n_virus, so it matches W_adjacency.flatten().
    """
    for required in ("W_adjacency", "Y_adjacency", "W_mask_adjacency"):
        if required not in data:
            raise KeyError(
                f"'{required}' not found in graph_data.npz. "
                f"Re-run create_unified_splits.py to regenerate it."
            )

    W = np.asarray(data["W_adjacency"], dtype=np.float64)
    Y = np.asarray(data["Y_adjacency"], dtype=np.int64)
    Wmask = np.asarray(data["W_mask_adjacency"], dtype=np.int64)
    if not (W.shape == Y.shape == Wmask.shape) or W.ndim != 2:
        raise ValueError(
            f"W ({W.shape}), Y ({Y.shape}), W_mask ({Wmask.shape}) "
            f"must all be the same 2D shape."
        )
    n_bact, n_virus = W.shape

    # row-major flattening: pair i = (i // n_virus, i % n_virus)
    bact_idx  = np.repeat(np.arange(n_bact), n_virus).astype(np.int64)
    virus_idx = np.tile(np.arange(n_virus), n_bact).astype(np.int64)

    data["bact_idx"]      = bact_idx
    data["virus_idx"]     = virus_idx
    data["Y_binary"]      = Y.flatten()              # CRISPR (binary target)
    data["W_continuous"]  = W.flatten()              # raw W (regression target)
    data["Mask_observed"] = Wmask.flatten()          # 1 where W is observed
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--run-id", default=None,
                    help="Optional. Auto-generated if omitted.")
    args = ap.parse_args()

    run_id = args.run_id or make_run_id()
    print(f"[preprocess] run_id = {run_id}")

    cfg = load_config(args.config)
    out = preprocessing_dir(run_id)
    freeze_config(cfg, run_dir(run_id) / "config.yaml")

    # ---- 1. Load raw -------------------------------------------------------
    print(f"[preprocess] loading {GRAPH_DATA_FILE}")
    data = load_graph_data(GRAPH_DATA_FILE)
    print(f"[preprocess] keys in graph_data.npz: {list(data.keys())}")

    data = flatten_pair_labels(data)
    n_pairs = len(data["Y_binary"])
    n_pos_Y = int(data["Y_binary"].sum())
    n_obs_W = int(data["Mask_observed"].sum())
    print(f"[preprocess] {n_pairs} pairs")
    print(f"[preprocess]   Y=1 (CRISPR positive): {n_pos_Y} ({100*n_pos_Y/n_pairs:.2f}%)")
    print(f"[preprocess]   W observed:            {n_obs_W} ({100*n_obs_W/n_pairs:.2f}%)")

    # ---- 2. Apply reducer to each feature matrix ---------------------------
    pcfg = cfg["preprocess"]
    feature_stats_long = []
    filtered_arrays = {}

    for source in pcfg["apply_to"]:
        if source not in data:
            print(f"[preprocess] WARNING: '{source}' not in data, skipping")
            continue
        X = np.asarray(data[source], dtype=np.float32)
        print(f"[preprocess] {source}: input shape {X.shape}")

        reducer = build_reducer(
            pcfg["reducer"],
            quantile=pcfg["quantile"],
            standardize=pcfg.get("standardize", True),
        )
        X_filt = reducer.fit_transform(X)
        print(f"[preprocess] {source}: kept {X_filt.shape[1]}/{X.shape[1]} features "
              f"(threshold = {reducer.feature_stats()['threshold']:.4g})")

        filtered_arrays[f"{source}_filtered"] = X_filt
        filtered_arrays[f"{source}_selected_idx"] = reducer.selected_idx_
        feature_stats_long.append(stats_to_long(reducer.feature_stats(), source))

    feature_stats_df = pd.concat(feature_stats_long, ignore_index=True)
    write_csv(feature_stats_df, out / "feature_stats.csv")
    print(f"[preprocess] wrote feature_stats.csv ({len(feature_stats_df)} rows)")

    # ---- 3. Save filtered features (npz) -----------------------------------
    passthrough = {k: v for k, v in data.items() if k not in pcfg["apply_to"]}
    write_npz({**filtered_arrays, **passthrough}, out / "data_filtered.npz")
    print(f"[preprocess] wrote data_filtered.npz")

    # ---- 4. Generate nested CV splits --------------------------------------
    rcfg = cfg["resampling"]
    strat_task = find_first_binary_task(cfg["tasks"])
    y_for_split, _ = extract_label(
        data, strat_task["label_source"], mask_source=None,
        binarize=strat_task.get("binarize", False),
        task_type="binary",
    )
    print(f"[preprocess] generating splits stratified on '{strat_task['name']}' "
          f"(n={len(y_for_split)}, n_pos={int(y_for_split.sum())})")

    splits_df = make_splits_dataframe(
        y_for_split,
        n_outer=rcfg["n_outer"],
        n_inner=rcfg["n_inner"],
        stratify=rcfg.get("stratify", True),
        seed=rcfg.get("seed", 42),
    )
    write_csv(splits_df, out / "splits.csv")
    print(f"[preprocess] wrote splits.csv ({len(splits_df)} rows)")

    # ---- 5. Label table (one row per sample x task) ------------------------
    label_rows = []
    for task in cfg["tasks"]:
        ttype = task.get("task_type", "binary")
        y_t, mask_t = extract_label(
            data, task["label_source"], task.get("mask_source"),
            binarize=task.get("binarize", False),
            task_type=ttype,
        )
        df = pd.DataFrame({
            "sample_id": np.arange(len(y_t)),
            "task":      task["name"],
            "task_type": ttype,
            "y":         y_t,
            "mask":      mask_t if mask_t is not None else 1,
        })
        label_rows.append(df)
    labels_long = pd.concat(label_rows, ignore_index=True)
    write_csv(labels_long, out / "labels.csv")
    print(f"[preprocess] wrote labels.csv ({len(labels_long)} rows)")

    # ---- 6. Preprocessing summary log --------------------------------------
    summary = {
        "run_id":      run_id,
        "config_used": str(args.config),
        "reducer":     pcfg["reducer"],
        "quantile":    pcfg["quantile"],
        "standardize": pcfg.get("standardize", True),
        "n_pairs":     int(n_pairs),
        "n_Y_pos":     int(n_pos_Y),
        "n_W_observed": int(n_obs_W),
        "feature_summary": {
            source: {
                "n_input":    int((feature_stats_df["source"] == source).sum()),
                "n_selected": int(((feature_stats_df["source"] == source) &
                                    feature_stats_df["selected"]).sum()),
                "threshold":  float(feature_stats_df.loc[
                                     feature_stats_df["source"] == source, "threshold"
                                 ].iloc[0]),
            }
            for source in pcfg["apply_to"] if source in data
        },
        "splits": {
            "n_outer": rcfg["n_outer"],
            "n_inner": rcfg["n_inner"],
            "stratify": rcfg.get("stratify", True),
            "stratify_on": strat_task["name"],
            "seed": rcfg.get("seed", 42),
            "n_samples": len(y_for_split),
        },
        "tasks": [
            {
                "name": t["name"],
                "task_type": t.get("task_type", "binary"),
                "label_source": t["label_source"],
                "mask_source": t.get("mask_source"),
            }
            for t in cfg["tasks"]
        ],
    }
    write_json(summary, out / "preprocess_log.json")
    print(f"[preprocess] wrote preprocess_log.json")
    print(f"[preprocess] DONE — outputs in {out}")
    print(f"[preprocess] run_id (use this for next step): {run_id}")


if __name__ == "__main__":
    main()
