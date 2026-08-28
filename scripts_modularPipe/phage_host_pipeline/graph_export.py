"""
graph_export.py — export the raw bacteria x virus adjacency matrices to CSV.

Reads graph_data.npz 

Writes into <run>/graph_matrices/ :
  Y_adjacency.csv        (n_bact x n_virus) 0/1   CRISPR linkage
  W_adjacency.csv        (n_bact x n_virus) float glasso edge probability
  W_mask_adjacency.csv   (n_bact x n_virus) 0/1   1 where W is observed
  bact_ids.csv           one bacteria id per row
  virus_ids.csv          one virus id per row

Usage:
  python graph_export.py --run-id <run_id>
"""
from __future__ import annotations
import argparse
import functools
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))


from paths import GRAPH_DATA_FILE, run_dir



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



def export_matrices(run_id: str, graph_path: Path) -> None:

    graph_path = Path(graph_path) if graph_path else GRAPH_DATA_FILE
    if not graph_path.exists():
        raise FileNotFoundError(f"graph_data.npz not found at {graph_path}")

    Y, W, Wm, bact_ids, virus_ids = _load_graph(graph_path)
    print(f"[export_matrices] adjacency {Y.shape[0]} bacteria x {Y.shape[1]} viruses")

    out = run_dir(run_id) / "graph_matrices"
    out.mkdir(parents=True, exist_ok=True)

    np.savetxt(out / "Y_adjacency.csv",      Y,  delimiter=",", fmt="%d")
    np.savetxt(out / "W_adjacency.csv",      W,  delimiter=",", fmt="%.6g")
    np.savetxt(out / "W_mask_adjacency.csv", Wm, delimiter=",", fmt="%d")
    with open(out / "bact_ids.csv", "w", encoding="utf-8") as fh:
        fh.write("bact_id\n")
        for b in bact_ids:
            fh.write(f"{b}\n")
    with open(out / "virus_ids.csv", "w", encoding="utf-8") as fh:
        fh.write("virus_id\n")
        for v in virus_ids:
            fh.write(f"{v}\n")

    n_crispr = int((Y >= 1).sum())
    n_obs    = int((Wm == 1).sum())
    print(f"[export_matrices] wrote 5 files to {out} | "
          f"CRISPR edges={n_crispr}, W-observed cells={n_obs}")

# Main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None,
                    help="Unused; accepted so the runner can pass it uniformly.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--graph-data", type=Path, default=GRAPH_DATA_FILE)
    args = ap.parse_args()

    export_matrices(args.run_id, args.graph_data)


if __name__ == "__main__":
    main()
