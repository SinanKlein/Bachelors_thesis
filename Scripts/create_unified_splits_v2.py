# create_unified_splits_v2.py
#
# Modular rewrite of create_unified_splits.py.
# ----------------------------
#   * Edge scope = all pairs in Y.
#   * Variance based feature selection.
#   * 10 stratified 80/10/10 splits (StratifiedKFold on the binary Y).
#   * W aligned onto the Y grid, fill 0, with a W observation mask.
#
# Everything is driven by the CONFIG block. Swap ACTIVE_Y / ACTIVE_W / the
# X bundle bases to retarget; nothing below CONFIG hardcodes a dataset.

import os
import re
import gzip
import functools
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

# Print everything immediately (flush) so progress is visible live.
print = functools.partial(print, flush=True)


# ===========================================================================
# CONFIG 
# ===========================================================================
BASE = Path(r"C:\Sinan_Klein\LMU\lmu_thesis\datas_final\SINAN_datasets\data\GvHD")

#  X : node-feature bundles (folder + base name per entity type) 
X_DIR = BASE / "abundance_derived" / "procs"
X_BUNDLES = {
    # role        base filename (without __rows/__cols or matrix extension)
    "bacteria": "bacteria_genus_procs_counts_min_genera_gt2",
    "virus":    "virus_votu_lev0_procs_counts_min_votus_gt2",
}

#  Y : CRISPR linkage tables (CRC stores them as plain .csv.gz; name once) 
Y_FILES = [
    BASE / "abundance_derived" / "network" / "crispr_genus_by_votu_binary.csv.gz",
    BASE / "abundance_derived" / "network" / "crispr_genus_by_votu_counts.csv.gz",
    BASE / "abundance_derived" / "network" / "crispr_host_taxid_by_votu_binary.csv.gz",
    BASE / "abundance_derived" / "network" / "crispr_host_taxid_by_votu_counts.csv.gz",
]
ACTIVE_Y = 0          # 0 = genus_binary (the only feature alignable target)
BINARIZE_Y = True     # (Y > 0) -> 1; harmless if Y is already binary

#  W : glasso edge probability tables 
W_FILES = [
    BASE / "Networks" / "Bipartite_edge_probability" / "glasso"
         / "GvHD_glasso_edge_probability_bacteria_virus_stars01.csv.gz",
    BASE / "Networks" / "Bipartite_edge_probability" / "glasso"
         / "GvHD_glasso_edge_probability_bacteria_virus_stars02.csv.gz",
]
ACTIVE_W = 1         # 0 = stars01

#  preprocessing knobs  
VARIANCE_QUANTILE = 0.0   # 0.0 = keep all nonzero variance features
N_SPLITS = 10
SEED = 42

# Name matching for W (its bacteria/virus IDs are formatted differently).
NORMALIZE_W_NAMES = True

BUILD_FLAT_EDGE_SPLITS = False

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RAW_DATA_DIR as OUT_DIR   # noqa: E402
# ===========================================================================


os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------
def resolve_path(p: Path) -> Path:
    """Accept either a file or a folder-that-contains-a-same-named csv."""
    if p.is_file():
        return p
    if p.is_dir():
        same = p / p.name
        if same.is_file():
            return same
        cands = sorted(p.glob("*.csv")) + sorted(p.glob("*.csv.gz"))
        if cands:
            return cands[0]
    return p


def read_table(p: Path, index_col=0) -> pd.DataFrame:
    p = resolve_path(p)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    sep = "\t" if str(p).lower().rstrip(".gz").endswith(".tsv") else ","
    return pd.read_csv(p, sep=sep, index_col=index_col)


def read_labels(p: Path) -> list[str]:
    return [str(x) for x in read_table(p).index]


def normalize_name(s: str) -> str:
    """Lowercase + drop all non-alphanumerics. Edit here to tune W matching."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# ---------------------------------------------------------------------------
# X bundle loading  (matrix + row/col label sidecars)
# ---------------------------------------------------------------------------
_MATRIX_EXTS = (".mtx.gz", ".mtx", ".npz", ".npy", ".csv.gz", ".csv", ".tsv", ".tsv.gz")
_SIDECARS = ("__rows.csv", "__cols.csv", "__rows.tsv", "__cols.tsv")


def _find_matrix_file(x_dir: Path, base: str) -> Path:
    cands = []
    for p in sorted(x_dir.iterdir()):
        if not p.is_file() or not p.name.startswith(base):
            continue
        low = p.name.lower()
        if low.endswith(_SIDECARS):
            continue
        if low.endswith(_MATRIX_EXTS):
            cands.append(p)
    if not cands:
        listing = "\n  ".join(q.name for q in sorted(x_dir.iterdir()))
        raise FileNotFoundError(
            f"No matrix file found for bundle '{base}' in {x_dir}.\n"
            f"Expected something like '{base}.mtx' alongside the __rows/__cols "
            f"label files. Folder contains:\n  {listing}"
        )
    cands.sort(key=lambda q: 0 if q.suffix.lower() in (".mtx", ".gz") else 1)
    return cands[0]


def _load_matrix_values(p: Path) -> np.ndarray:
    low = p.name.lower()
    if low.endswith((".mtx", ".mtx.gz")):
        from scipy.io import mmread
        if low.endswith(".gz"):
            with gzip.open(p, "rb") as fh:
                m = mmread(fh)
        else:
            m = mmread(str(p))
        import scipy.sparse as sp
        return m.toarray() if sp.issparse(m) else np.asarray(m)
    if low.endswith(".npz"):
        try:
            from scipy.sparse import load_npz
            return load_npz(p).toarray()
        except Exception:
            d = np.load(p, allow_pickle=True)
            return np.asarray(d[d.files[0]])
    if low.endswith(".npy"):
        return np.asarray(np.load(p, allow_pickle=True))
    return read_table(p).to_numpy()


def load_x_bundle(x_dir: Path, base: str):
    """Return (mat float32 [n_ent x n_proc], entities list, procs list)."""
    rows_p = x_dir / f"{base}__rows.csv"
    cols_p = x_dir / f"{base}__cols.csv"
    entities = read_labels(rows_p)
    procs = read_labels(cols_p)

    mp = _find_matrix_file(x_dir, base)
    mat = np.asarray(_load_matrix_values(mp), dtype=np.float64)

    n_ent, n_proc = len(entities), len(procs)
    if mat.shape == (n_ent, n_proc):
        pass
    elif mat.shape == (n_proc, n_ent):
        mat = mat.T  # stored procs x entities
    else:
        raise ValueError(
            f"[{base}] matrix {mp.name} shape {mat.shape} matches neither "
            f"(entities={n_ent} x procs={n_proc}) nor its transpose."
        )
    print(f"  [{base}] matrix={mp.name}  shape={mat.shape}  "
          f"density={(mat != 0).mean():.4%}")
    return mat.astype(np.float32), entities, procs


# ---------------------------------------------------------------------------
# W alignment onto the Y grid (with optional name normalization)
# ---------------------------------------------------------------------------
def align_W_to_grid(W: pd.DataFrame, bact_list, virus_list, normalize: bool):
    """
    Reindex W onto the (bact_list x virus_list) grid.

    Returns:
        W_grid  (n_bact, n_virus) float64  — W value where matched, else 0
        W_mask  (n_bact, n_virus) float32  — 1 where BOTH names matched in W
    """
    if normalize:
        Wn = W.copy()
        Wn.index = [normalize_name(x) for x in W.index]
        Wn.columns = [normalize_name(x) for x in W.columns]
        # collapse any collisions from normalization (rare) by max
        Wn = Wn.groupby(level=0).max()
        Wn = Wn.T.groupby(level=0).max().T
        grid_b = [normalize_name(b) for b in bact_list]
        grid_v = [normalize_name(v) for v in virus_list]
    else:
        Wn = W
        grid_b = list(bact_list)
        grid_v = list(virus_list)

    in_b = np.array([b in Wn.index for b in grid_b], dtype=np.float32)
    in_v = np.array([v in Wn.columns for v in grid_v], dtype=np.float32)
    W_mask = np.outer(in_b, in_v).astype(np.float32)

    W_grid = (Wn.reindex(index=grid_b, columns=grid_v, fill_value=0.0)
                .to_numpy(dtype=np.float64))
    return W_grid, W_mask


# ===========================================================================
# 1. Load inputs
# ===========================================================================
print("Loading X bundles...")
Xb_mat, bact_entities, bact_procs = load_x_bundle(X_DIR, X_BUNDLES["bacteria"])
Xv_mat, virus_entities, virus_procs = load_x_bundle(X_DIR, X_BUNDLES["virus"])
bact_pos = {b: i for i, b in enumerate(bact_entities)}
virus_pos = {v: i for i, v in enumerate(virus_entities)}

print("Loading Y...")
Y_df = read_table(Y_FILES[ACTIVE_Y])
print(f"  Y = {Y_FILES[ACTIVE_Y].stem}  shape={Y_df.shape}  "
      f"(rows={Y_df.index.name}, cols->vOTU)")

print("Loading W...")
W_df = read_table(W_FILES[ACTIVE_W])
print(f"  W = {W_FILES[ACTIVE_W].stem}  shape={W_df.shape}")


# ===========================================================================
# 2. Align entities  (edge scope = Y pairs that also have features)
# ===========================================================================
bact_list  = [b for b in map(str, Y_df.index)   if b in bact_pos]
virus_list = [v for v in map(str, Y_df.columns) if v in virus_pos]
n_bact, n_virus = len(bact_list), len(virus_list)
n_edges = n_bact * n_virus

dropped_b = len(Y_df.index) - n_bact
dropped_v = len(Y_df.columns) - n_virus
print(f"\nAlignment (Y ∩ X-features):")
print(f"  bacteria: {n_bact}/{len(Y_df.index)} kept "
      f"({dropped_b} Y rows have no features)")
print(f"  viruses:  {n_virus}/{len(Y_df.columns)} kept "
      f"({dropped_v} Y cols have no features)")
if n_bact < 2 or n_virus < 2:
    raise SystemExit(
        "Almost nothing aligned. If you picked a host_taxid Y, note its rows "
        "are taxids and cannot match the genus-named features — use ACTIVE_Y=0."
    )
print(f"  edge scope: {n_bact} x {n_virus} = {n_edges} edges")


# ===========================================================================
# 3. Build aligned arrays
# ===========================================================================
# node features ordered to match bact_list / virus_list
Xb_np = Xb_mat[[bact_pos[b]  for b in bact_list]]    # (n_bact,  p_b)
Xv_np = Xv_mat[[virus_pos[v] for v in virus_list]]   # (n_virus, p_v)

# Y on the grid (binary target)
Y_sub = Y_df.loc[bact_list, virus_list]
Y_2d = (Y_sub.to_numpy() > 0).astype(np.int64) if BINARIZE_Y else \
       Y_sub.to_numpy().astype(np.int64)
Y_flat = Y_2d.flatten()

# W on the grid + observation mask (names normalized to improve matching)
W_2d, W_mask_2d = align_W_to_grid(W_df, bact_list, virus_list, NORMALIZE_W_NAMES)
W_flat = W_2d.flatten().astype(np.float64)
W_mask = W_mask_2d.flatten().astype(np.float32)

# edge to node index maps (row-major: edge i = (i // n_virus, i % n_virus))
bact_idx_all  = np.repeat(np.arange(n_bact), n_virus).astype(np.int64)
virus_idx_all = np.tile(np.arange(n_virus), n_bact).astype(np.int64)

print(f"\n  Y+ (CRISPR positive): {Y_flat.sum()} / {n_edges} "
      f"({Y_flat.mean():.4f})")
print(f"  W observed cells:     {int(W_mask.sum())} / {n_edges} "
      f"({W_mask.mean():.4f})   <-- W's coverage of the Y grid")


# ===========================================================================
# 4. Variance feature selection  (same mechanism as before)
# ===========================================================================
# Edge feature variance == node feature variance under uniform edge repetition

feat_var = np.concatenate([Xb_np.var(axis=0), Xv_np.var(axis=0)])
if VARIANCE_QUANTILE > 0:
    var_threshold = float(np.quantile(feat_var[feat_var > 0], VARIANCE_QUANTILE))
else:
    var_threshold = 0.0
feat_mask = feat_var > var_threshold
p_total = Xb_np.shape[1] + Xv_np.shape[1]
print(f"\n  Variance filter (q={VARIANCE_QUANTILE}): "
      f"{int(feat_mask.sum())}/{p_total} edge-features kept "
      f"(threshold={var_threshold:.4g})")


# ===========================================================================
# 5. 10 stratified splits  (80/10/10)
# ===========================================================================
print("\nCreating stratified splits...")
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

X_flat_filtered = None
if BUILD_FLAT_EDGE_SPLITS:
    est_gb = n_edges * int(feat_mask.sum()) * 4 / 1e9
    print(f"  [flat splits] materializing X_flat (~{est_gb:.1f} GB float32)...")
    X_flat = np.hstack([Xb_np[bact_idx_all], Xv_np[virus_idx_all]]).astype(np.float32)
    X_flat_filtered = X_flat[:, feat_mask]

for k, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(n_edges), Y_flat)):
    y_trainval = Y_flat[trainval_idx]
    skf_inner = StratifiedKFold(n_splits=9, shuffle=True, random_state=SEED + k)
    for train_sub, val_sub in skf_inner.split(np.zeros(len(trainval_idx)), y_trainval):
        train_idx = trainval_idx[train_sub]
        val_idx   = trainval_idx[val_sub]
        break

    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0
    assert len(train_idx) + len(val_idx) + len(test_idx) == n_edges

    payload = dict(
        Y_train=Y_flat[train_idx], Y_val=Y_flat[val_idx], Y_test=Y_flat[test_idx],
        W_train=W_flat[train_idx], W_val=W_flat[val_idx], W_test=W_flat[test_idx],
        W_mask_train=W_mask[train_idx], W_mask_val=W_mask[val_idx],
        W_mask_test=W_mask[test_idx],
        bact_idx_train=bact_idx_all[train_idx], bact_idx_val=bact_idx_all[val_idx],
        bact_idx_test=bact_idx_all[test_idx],
        virus_idx_train=virus_idx_all[train_idx], virus_idx_val=virus_idx_all[val_idx],
        virus_idx_test=virus_idx_all[test_idx],
        train_indices=train_idx, val_indices=val_idx, test_indices=test_idx,
        feature_mask=feat_mask,
    )
    if X_flat_filtered is not None:
        payload.update(
            X_train=X_flat_filtered[train_idx],
            X_val=X_flat_filtered[val_idx],
            X_test=X_flat_filtered[test_idx],
        )

    np.savez(os.path.join(OUT_DIR, f"split_k{k}.npz"), **payload)
    print(f"  split {k}: train={len(train_idx)} (Y+={Y_flat[train_idx].sum()}) | "
          f"val={len(val_idx)} (Y+={Y_flat[val_idx].sum()}) | "
          f"test={len(test_idx)} (Y+={Y_flat[test_idx].sum()})")


# ===========================================================================
# 6. graph_data.npz  (identical keys to the original)
# ===========================================================================
print("\nSaving graph_data.npz...")
np.savez(
    os.path.join(OUT_DIR, "graph_data.npz"),
    Xb=Xb_np,                                   # (n_bact,  p_b)
    Xv=Xv_np,                                   # (n_virus, p_v)
    W_adjacency=W_2d.astype(np.float64),        # (n_bact, n_virus)
    Y_adjacency=Y_2d.astype(np.int64),          # (n_bact, n_virus) CRISPR
    W_mask_adjacency=W_mask_2d.astype(np.int64),# (n_bact, n_virus) 1 if W observed
    bact_ids=np.array(bact_list, dtype=object),
    virus_ids=np.array(virus_list, dtype=object),
    bact_procs=np.array(bact_procs, dtype=object),
    virus_procs=np.array(virus_procs, dtype=object),
    n_bact=n_bact,
    n_virus=n_virus,
)


# ===========================================================================
# 7. Summary
# ===========================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Y target:     {Y_FILES[ACTIVE_Y].stem}")
print(f"W target:     {W_FILES[ACTIVE_W].stem}")
print(f"Edge scope:   {n_bact} bacteria x {n_virus} viruses = {n_edges} edges")
print(f"Y prevalence: {Y_flat.sum()} / {n_edges} = {Y_flat.mean():.4f}")
print(f"W coverage:   {int(W_mask.sum())} / {n_edges} = {W_mask.mean():.4f}")
print(f"Feature dim:  p_b={Xb_np.shape[1]}, p_v={Xv_np.shape[1]} "
      f"(kept {int(feat_mask.sum())} after variance filter)")
print(f"Splits:       {N_SPLITS} -> {OUT_DIR}")
print(f"Flat splits:  {'built' if BUILD_FLAT_EDGE_SPLITS else 'skipped (legacy)'}")
print()
print(f"graph_data.npz written to: {OUT_DIR}  (this is paths.RAW_DATA_DIR)")
print("W is a task -> regression is computed only where W_mask==1 (its subset).")
