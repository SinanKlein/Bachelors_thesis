"""
descriptives.py — emit every descriptive number the Data chapter reports.

Reads graph_data.npz and nothing else, so it does not depend on a model having
been fit and is unaffected by the representation the pipeline is configured for.
It exists because those numbers were previously recoverable only by reading text
baked into a plot title, which meant the thesis quoted figures no script
produced. Every number in the Data chapter should come out of here.

Writes into <run>/metrics/ :
  descriptives.json   nested, grouped by scope
  descriptives.csv    tidy long format: scope, metric, value

Scopes
------
  grid      the full genus x vOTU Cartesian product
  mask      the W-observed block, i.e. the pairs every model actually sees
  degree_*  CRISPR degree statistics, reported on both scopes because they
            differ substantially (the mask block truncates the high-degree tail)
  features  protein-cluster profiles, over the grid-aligned entities

Note on the mask: a cell is observed exactly when its row and its column were
both matched into W, so the observed set is the outer product of two indicator
vectors. n_bact_matched * n_virus_matched == n_observed is asserted below; if
that assertion ever fails, the mask is no longer a block and the Data chapter's
description of it is wrong.

Usage:
  python descriptives.py --run-id <run_id>
"""
from __future__ import annotations
import argparse
import functools
import json
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import GRAPH_DATA_FILE, metrics_dir


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_graph(graph_path: Path):
    d = np.load(graph_path, allow_pickle=True)
    Y = np.asarray(d["Y_adjacency"]).astype(int)
    W = np.asarray(d["W_adjacency"]).astype(float)
    M = (np.asarray(d["W_mask_adjacency"]).astype(bool)
         if "W_mask_adjacency" in d.files else np.ones_like(W, dtype=bool))
    Xb = np.asarray(d["Xb"]).astype(float) if "Xb" in d.files else None
    Xv = np.asarray(d["Xv"]).astype(float) if "Xv" in d.files else None
    return Y, W, M, Xb, Xv


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------
def _degree_stats(A: np.ndarray, top_k: int = 10) -> dict:
    """CRISPR degree summary for one binary matrix."""
    A = (A > 0).astype(int)
    db, dv = A.sum(axis=1), A.sum(axis=0)
    total = int(db.sum())
    top = np.sort(db)[::-1][:top_k].sum()
    return {
        "n_edges": total,
        "genus_degree_median": float(np.median(db)),
        "genus_degree_mean": float(db.mean()),
        "genus_degree_max": int(db.max()) if db.size else 0,
        "genus_degree_zero": int((db == 0).sum()),
        "votu_degree_median": float(np.median(dv)),
        "votu_degree_mean": float(dv.mean()),
        "votu_degree_max": int(dv.max()) if dv.size else 0,
        "votu_degree_zero": int((dv == 0).sum()),
        f"top{top_k}_genera_share_of_edges": float(top / total) if total else 0.0,
    }


def _feature_stats(X: np.ndarray) -> dict:
    """Sparsity summary for one protein-cluster profile matrix."""
    nz = (X != 0)
    per_entity = nz.sum(axis=1)
    return {
        "n_entities": int(X.shape[0]),
        "n_clusters": int(X.shape[1]),
        "density": float(nz.mean()),
        "clusters_per_entity_median": float(np.median(per_entity)),
        "clusters_per_entity_min": int(per_entity.min()),
        "clusters_per_entity_max": int(per_entity.max()),
    }


def compute(Y: np.ndarray, W: np.ndarray, M: np.ndarray,
            Xb: np.ndarray | None, Xv: np.ndarray | None) -> dict:
    n_bact, n_virus = Y.shape
    n_pairs = n_bact * n_virus

    rows, cols = M.any(axis=1), M.any(axis=0)
    n_b_matched, n_v_matched = int(rows.sum()), int(cols.sum())
    n_obs = int(M.sum())
    assert n_b_matched * n_v_matched == n_obs, (
        "The observation mask is not a block: "
        f"{n_b_matched} x {n_v_matched} != {n_obs}. The Data chapter describes "
        "it as the outer product of two indicator vectors — recheck "
        "align_W_to_grid in build_graph_data.py before quoting these numbers.")

    y_m, w_m = Y[M], W[M]
    n_y = int((y_m > 0).sum())
    n_w = int((w_m > 0).sum())
    n_both = int(((y_m > 0) & (w_m > 0)).sum())
    expected_both = (n_y * n_w / n_obs) if n_obs else 0.0

    out = {
        "grid": {
            "n_bacteria": n_bact,
            "n_viruses": n_virus,
            "n_pairs": n_pairs,
            "n_crispr_positive": int((Y > 0).sum()),
            "crispr_prevalence": float((Y > 0).mean()),
        },
        "mask": {
            "n_bacteria_matched": n_b_matched,
            "n_viruses_matched": n_v_matched,
            "n_observed": n_obs,
            "coverage_of_grid": float(n_obs / n_pairs),
            "n_crispr_positive": n_y,
            "crispr_prevalence": float(n_y / n_obs) if n_obs else 0.0,
            "n_glasso_edge": n_w,
            "glasso_prevalence": float(n_w / n_obs) if n_obs else 0.0,
            "w_mean": float(w_m.mean()) if n_obs else 0.0,
            "w_sd": float(w_m.std()) if n_obs else 0.0,
            "w_median": float(np.median(w_m)) if n_obs else 0.0,
            "w_frac_exactly_zero": float((w_m == 0).mean()) if n_obs else 0.0,
            "w_frac_above_half": float((w_m > 0.5).mean()) if n_obs else 0.0,
            "n_both": n_both,
            "n_both_expected_if_independent": float(expected_both),
            "both_observed_over_expected": (float(n_both / expected_both)
                                            if expected_both else 0.0),
        },
        "degree_grid": _degree_stats(Y),
        "degree_mask": _degree_stats(Y[np.ix_(rows, cols)]),
    }

    if Xb is not None:
        out["features_bacteria"] = _feature_stats(Xb)
    if Xv is not None:
        out["features_viruses"] = _feature_stats(Xv)
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _write(run_id: str, stats: dict) -> Path:
    out = metrics_dir(run_id)

    with open(out / "descriptives.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    with open(out / "descriptives.csv", "w", encoding="utf-8") as fh:
        fh.write("scope,metric,value\n")
        for scope, block in stats.items():
            for metric, value in block.items():
                fh.write(f"{scope},{metric},{value!r}\n"
                         if isinstance(value, str) else
                         f"{scope},{metric},{value}\n")
    return out


def _report(stats: dict) -> None:
    g, m = stats["grid"], stats["mask"]
    print(f"[descriptives] grid      {g['n_bacteria']} x {g['n_viruses']} "
          f"= {g['n_pairs']} pairs | CRISPR+ {g['n_crispr_positive']} "
          f"({g['crispr_prevalence']:.2%})")
    print(f"[descriptives] mask      {m['n_bacteria_matched']} x "
          f"{m['n_viruses_matched']} = {m['n_observed']} observed "
          f"({m['coverage_of_grid']:.1%} of grid)")
    print(f"[descriptives] on mask   CRISPR+ {m['n_crispr_positive']} "
          f"({m['crispr_prevalence']:.2%}) | W>0 {m['n_glasso_edge']} "
          f"({m['glasso_prevalence']:.1%}) | mean W {m['w_mean']:.3f}")
    print(f"[descriptives] overlap   {m['n_both']} observed vs "
          f"{m['n_both_expected_if_independent']:.0f} expected under "
          f"independence (ratio {m['both_observed_over_expected']:.2f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None,
                    help="Unused; accepted so the runner can pass it uniformly.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--graph-data", type=Path, default=GRAPH_DATA_FILE)
    args = ap.parse_args()

    graph_path = Path(args.graph_data) if args.graph_data else GRAPH_DATA_FILE
    if not graph_path.exists():
        raise FileNotFoundError(f"graph_data.npz not found at {graph_path}")

    Y, W, M, Xb, Xv = _load_graph(graph_path)
    stats = compute(Y, W, M, Xb, Xv)
    out = _write(args.run_id, stats)
    _report(stats)
    print(f"[descriptives] wrote descriptives.json + descriptives.csv to {out}")


if __name__ == "__main__":
    main()
