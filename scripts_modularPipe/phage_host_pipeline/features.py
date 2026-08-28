"""
features.py — the single source of truth for how features are built.

Three switches define a representation, and they are the SAME three the
ablation varies. Every consumer calls the functions here, so the main pipeline
and the ablation cannot drift apart:

  selection : 'quantile'   per-source top fraction by RAW variance
              'equal_dim'  all Xv procs + top-|Xv| Xb procs, so both pair-halves
                           have identical width (required before a `sum`)
  transform : 'clr' | 'presence_absence', applied to the KEPT columns only.
              Variance is ALWAYS scored on raw counts, never on a transform, so
              the transform never changes which features get selected.
  combine   : 'concat' | 'sum'. The sum is POSITIONAL (bacterial proc j + viral
              proc j); bacteria and viruses use separate proc vocabularies, so
              it is an architectural choice, not "is cluster X in either
              partner".

Consumers:
  preprocess.py  select_and_transform  -> writes data_filtered.npz
  experiment.py  pair_matrix (reads that npz)
  latent.py      same as experiment.py
  ablation.py    select_and_transform + pair_matrix, per arm, in memory
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import build_reducer, apply_feature_transform

# Config helper
def resolve_quantile(quantile_cfg, source: str) -> float:
    """`quantile` may be a single number for every source, or a per-source map."""
    if isinstance(quantile_cfg, dict):
        if source in quantile_cfg:
            return float(quantile_cfg[source])
        if "default" in quantile_cfg:
            return float(quantile_cfg["default"])
        raise KeyError(
            f"No quantile configured for source '{source}' and no 'default' key. "
            f"Have: {list(quantile_cfg.keys())}"
        )
    return float(quantile_cfg)

# Selection
def fit_selectors(data: dict, pcfg: dict, selection: str, log: str = "") -> dict:
    """Fit one reducer per feature source. Returns {source: fitted reducer}.

    'equal_dim' deliberately fits Xv FIRST (keeping all of it) and then takes the
    top-|Xv| of Xb. If Xb has fewer procs than Xv, the viral side is trimmed to
    match, so a `sum` combine stays dimensionally valid either way.
    """
    sources = list(pcfg.get("apply_to", ["Xb", "Xv"]))
    reducers: dict = {}

    if selection == "equal_dim":
        if not {"Xb", "Xv"} <= set(sources):
            raise ValueError(
                "selection='equal_dim' needs both Xb and Xv in preprocess.apply_to")
        Xb_raw = np.asarray(data["Xb"], dtype=np.float32)
        Xv_raw = np.asarray(data["Xv"], dtype=np.float32)
        k = Xv_raw.shape[1]
        reducers["Xv"] = build_reducer("top_k_variance", k=k).fit(Xv_raw)
        reducers["Xb"] = build_reducer("top_k_variance", k=k).fit(Xb_raw)
        n_b = len(reducers["Xb"].selected_idx_)
        if n_b != k:
            print(f"{log}NOTE: equal_dim wanted k={k} bacterial procs, only {n_b} "
                  f"exist; trimming Xv to {n_b} so combine='sum' stays valid.")
            reducers["Xv"] = build_reducer("top_k_variance", k=n_b).fit(Xv_raw)
        return reducers

    if selection == "quantile":
        for source in sources:
            if source not in data:
                print(f"{log}WARNING: '{source}' not in data, skipping")
                continue
            X_raw = np.asarray(data[source], dtype=np.float32)
            q = resolve_quantile(pcfg["quantile"], source)
            print(f"{log}{source}: keep top {100 * q:.1f}% by RAW variance")
            reducers[source] = build_reducer(
                pcfg.get("reducer", "mean_variance"), quantile=q,
                standardize=False).fit(X_raw)
        return reducers

    raise ValueError(f"Unknown selection {selection!r} (use 'quantile' or 'equal_dim')")

# Selection + transform
def select_and_transform(data: dict, pcfg: dict, selection: str, transform: str,
                         log: str = "") -> tuple[dict, dict]:
    """Select features by raw variance, then transform the kept columns.

    Returns (arrays, reducers) where `arrays` is ready to be written into
    data_filtered.npz:
        <source>_filtered      selected + transformed
        <source>_raw_filtered  selected, untransformed
        <source>_selected_idx  indices into the original feature axis
    """
    pseudocount = float(pcfg.get("clr_pseudocount", 0.5))
    reducers = fit_selectors(data, pcfg, selection, log=log)
    arrays: dict = {}

    for source, reducer in reducers.items():
        X_raw = np.asarray(data[source], dtype=np.float32)
        X_raw_filt = reducer.transform(X_raw)
        X_t = apply_feature_transform(X_raw_filt, transform, pseudocount=pseudocount)
        print(f"{log}{source}: input {X_raw.shape} -> kept {X_t.shape[1]}"
              f"/{X_raw.shape[1]} features, transform={transform}"
              + (f" (pseudocount={pseudocount:g})" if transform == "clr" else ""))
        arrays[f"{source}_filtered"] = X_t.astype(np.float32)
        arrays[f"{source}_raw_filtered"] = X_raw_filt.astype(np.float32)
        arrays[f"{source}_selected_idx"] = reducer.selected_idx_

    return arrays, reducers

# Pair assembly
def pair_matrix(Xb: np.ndarray, Xv: np.ndarray, bact_idx, virus_idx,
                combine: str = "concat") -> np.ndarray:
    """Expand per-entity features to one row per bacterium x virus pair."""
    Xb = np.asarray(Xb, dtype=np.float32)
    Xv = np.asarray(Xv, dtype=np.float32)

    if bact_idx is not None and virus_idx is not None:
        Xb_pair = Xb[np.asarray(bact_idx).ravel().astype(int)]
        Xv_pair = Xv[np.asarray(virus_idx).ravel().astype(int)]
    elif Xb.shape[0] == Xv.shape[0]:
        Xb_pair, Xv_pair = Xb, Xv
    else:
        raise RuntimeError(f"Xb ({Xb.shape}) / Xv ({Xv.shape}) row counts differ.")

    if combine == "concat":
        return np.concatenate([Xb_pair, Xv_pair], axis=1)
    if combine == "sum":
        if Xb_pair.shape[1] != Xv_pair.shape[1]:
            raise RuntimeError(
                f"combine='sum' needs equal feature widths, got Xb={Xb_pair.shape[1]} "
                f"vs Xv={Xv_pair.shape[1]}. Set preprocess.selection: equal_dim.")
        return Xb_pair + Xv_pair
    raise ValueError(f"Unknown combine mode: {combine!r} (use 'concat' or 'sum')")


def pair_matrix_from_filtered(data: dict, combine: str = "concat") -> np.ndarray:
    """pair_matrix() for a loaded data_filtered.npz."""
    return pair_matrix(data["Xb_filtered"], data["Xv_filtered"],
                       data.get("bact_idx"), data.get("virus_idx"), combine)


def feature_stats_long(reducer, source: str, feature_names=None,
                       n_rows: int | None = None) -> pd.DataFrame:
    """Per-feature selection diagnostics for one source, in long form."""
    stats = reducer.feature_stats()
    df = pd.DataFrame({
        "source":     source,
        "feature_id": stats["feature_id"],
        "mean":       stats["mean"],
        "var":        stats["var"],
        "score":      stats["score"],
        "selected":   stats["selected"],
    })
    if feature_names is not None and len(feature_names) == len(df):
        df["feature_name"] = np.asarray(feature_names).astype(str)
    else:
        df["feature_name"] = [f"{source}_feature_{i}" for i in df["feature_id"]]
    # Raw total abundance across entity rows; used by the descriptive ProC plots.
    df["total_abundance"] = df["mean"] * (int(n_rows) if n_rows is not None else 1)
    df["threshold"] = stats["threshold"]
    return df
