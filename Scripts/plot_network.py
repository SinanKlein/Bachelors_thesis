"""
Bipartite bacteria-virus interaction network plot (multimodal: CRISPR + glasso).

Reads the raw interaction adjacency matrices from graph_data.npz and draws a
bipartite network with bacteria on the left and viruses on the right. Edges are
colored by modality:
  - "both"        : CRISPR link (Y=1) AND high glasso edge probability (W > thr)
  - "CRISPR only" : Y = 1 but W <= thr (or W unobserved)
  - "glasso only" : W > thr but Y = 0

Pure matplotlib (no networkx / igraph), so it needs no extra packages.

Usage:
  python plot_network.py --config default.yaml --run-id <run_id>
  python plot_network.py --run-id <run_id> --w-threshold 0.5 --max-edges 1500
"""
from __future__ import annotations
import argparse
import functools
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from paths import GRAPH_DATA_FILE, run_dir

EDGE_STYLE = {
    "both":        dict(color="#6A3D9A", alpha=0.85, lw=0.9, z=3),
    "CRISPR only": dict(color="#E31A1C", alpha=0.65, lw=0.7, z=2),
    "glasso only": dict(color="#1F78B4", alpha=0.45, lw=0.5, z=1),
}


def _load_graph(graph_path: Path):
    d = np.load(graph_path, allow_pickle=True)
    Y = np.asarray(d["Y_adjacency"]).astype(float)        # (n_bact, n_virus)
    W = np.asarray(d["W_adjacency"]).astype(float)
    Wm = (np.asarray(d["W_mask_adjacency"]).astype(int)
          if "W_mask_adjacency" in d.files else np.ones_like(W, dtype=int))
    bact_ids = (np.asarray(d["bact_ids"]).astype(str)
                if "bact_ids" in d.files else np.array([f"b{i}" for i in range(Y.shape[0])]))
    virus_ids = (np.asarray(d["virus_ids"]).astype(str)
                 if "virus_ids" in d.files else np.array([f"v{j}" for j in range(Y.shape[1])]))
    return Y, W, Wm, bact_ids, virus_ids


def build_edges(Y, W, Wm, w_threshold):
    """Return list of (b, v, w_value, edge_type)."""
    crispr = Y >= 1
    glasso = (Wm == 1) & (W > w_threshold)
    bb, vv = np.where(crispr | glasso)
    edges = []
    for b, v in zip(bb.tolist(), vv.tolist()):
        is_c, is_g = bool(crispr[b, v]), bool(glasso[b, v])
        etype = "both" if (is_c and is_g) else ("CRISPR only" if is_c else "glasso only")
        edges.append((b, v, float(W[b, v]), etype))
    return edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Unused; accepted for runner consistency.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--graph-data", default=None, help="Path to graph_data.npz (default: paths.GRAPH_DATA_FILE).")
    ap.add_argument("--w-threshold", type=float, default=0.5, help="W edge-probability cut for a glasso edge.")
    ap.add_argument("--max-edges", type=int, default=1500,
                    help="Cap on drawn edges (prioritizes 'both', then CRISPR, then strongest W).")
    ap.add_argument("--label-max", type=int, default=40,
                    help="Annotate node names only when a side has <= this many nodes.")
    args = ap.parse_args()

    graph_path = Path(args.graph_data) if args.graph_data else GRAPH_DATA_FILE
    if not graph_path.exists():
        raise FileNotFoundError(f"graph_data.npz not found at {graph_path}")
    Y, W, Wm, bact_ids, virus_ids = _load_graph(graph_path)
    print(f"[network] adjacency {Y.shape[0]} bacteria x {Y.shape[1]} viruses")

    edges = build_edges(Y, W, Wm, args.w_threshold)
    n_all = len(edges)
    if n_all == 0:
        print("[network] no edges found; nothing to plot.")
        return

    # cap edges: keep all 'both', then CRISPR-only, then strongest glasso-only
    rank = {"both": 0, "CRISPR only": 1, "glasso only": 2}
    edges.sort(key=lambda e: (rank[e[3]], -e[2]))
    capped = edges[: args.max_edges]

    # nodes that participate in the drawn edges
    b_used = sorted({e[0] for e in capped})
    v_used = sorted({e[1] for e in capped})
    # order nodes by degree for a tidier layout
    from collections import Counter
    bdeg = Counter(e[0] for e in capped)
    vdeg = Counter(e[1] for e in capped)
    b_used.sort(key=lambda b: -bdeg[b])
    v_used.sort(key=lambda v: -vdeg[v])
    b_y = {b: pos for b, pos in zip(b_used, np.linspace(0, 1, max(len(b_used), 1)))}
    v_y = {v: pos for v, pos in zip(v_used, np.linspace(0, 1, max(len(v_used), 1)))}
    x_b, x_v = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(9, 11))
    for etype in ("glasso only", "CRISPR only", "both"):  # draw weak first
        st = EDGE_STYLE[etype]
        for b, v, _w, et in capped:
            if et != etype:
                continue
            ax.plot([x_b, x_v], [b_y[b], v_y[v]], color=st["color"],
                    alpha=st["alpha"], lw=st["lw"], zorder=st["z"], solid_capstyle="round")

    ax.scatter([x_b] * len(b_used), [b_y[b] for b in b_used],
               s=[18 + 12 * bdeg[b] for b in b_used], color="#33A02C",
               edgecolor="white", linewidth=0.4, zorder=4, label="Bacteria")
    ax.scatter([x_v] * len(v_used), [v_y[v] for v in v_used],
               s=[18 + 12 * vdeg[v] for v in v_used], color="#FF7F00",
               edgecolor="white", linewidth=0.4, zorder=4, label="Virus")

    if len(b_used) <= args.label_max:
        for b in b_used:
            ax.text(x_b - 0.02, b_y[b], str(bact_ids[b]), ha="right", va="center", fontsize=6, color="#1b5e20")
    if len(v_used) <= args.label_max:
        for v in v_used:
            ax.text(x_v + 0.02, v_y[v], str(virus_ids[v]), ha="left", va="center", fontsize=6, color="#b35900")

    n_c = sum(1 for e in edges if e[3] in ("both", "CRISPR only"))
    n_g = sum(1 for e in edges if e[3] in ("both", "glasso only"))
    n_both = sum(1 for e in edges if e[3] == "both")
    ax.set_title("Bacteria–virus interaction network", fontsize=14, fontweight="bold")
    cap = (f"{len(b_used)} bacteria, {len(v_used)} viruses | "
           f"CRISPR edges={n_c}, glasso(W>{args.w_threshold:g}) edges={n_g}, overlap={n_both}")
    if len(capped) < n_all:
        cap += f" | showing strongest {len(capped)}/{n_all} edges"
    ax.text(0.5, -0.02, cap, transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#555555")

    handles = [Line2D([0], [0], color=EDGE_STYLE[t]["color"], lw=2, label=t) for t in EDGE_STYLE]
    handles += [Line2D([0], [0], marker="o", color="w", markerfacecolor="#33A02C", markersize=8, label="Bacteria"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF7F00", markersize=8, label="Virus")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
              ncol=5, frameon=False, fontsize=8)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.06, 1.06)
    ax.axis("off")

    out = run_dir(args.run_id) / "plots" / "network"
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "interaction_network.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # edge list for downstream use / thesis tables
    import csv
    with open(out / "interaction_edges.csv", "w", newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["bact_id", "virus_id", "W", "edge_type"])
        for b, v, wval, et in edges:
            wcsv.writerow([bact_ids[b], virus_ids[v], f"{wval:.6g}", et])
    print(f"[network] wrote {out / 'interaction_network.png'} "
          f"({len(b_used)}x{len(v_used)} nodes, {n_all} edges, {len(capped)} drawn)")


if __name__ == "__main__":
    main()
