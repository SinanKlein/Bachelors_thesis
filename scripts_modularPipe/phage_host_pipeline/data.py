"""Data stages.

  python data.py build                                  raw tables -> graph_data.npz
  python data.py preprocess --config C --run-id R       features, labels, splits
  python data.py describe   --config C --run-id R       descriptive numbers + matrix CSVs
"""
from __future__ import annotations

import gzip
import re
import sys

import numpy as np
import pandas as pd
import yaml

from common import (COHORT, COHORT_DIR, GRAPH_DATA_FILE, load_config, load_npz,
                    make_splits, run_dir, stage_args, write_csv, write_json)

# =============================================================================
# 1. build: raw X / Y / W tables -> graph_data.npz
# =============================================================================
X_DIR = COHORT_DIR / "abundance_derived" / "procs"
X_BUNDLES = {"bacteria": "bacteria_genus_procs_counts_min_genera_gt2",
             "virus": "virus_votu_lev0_procs_counts_min_votus_gt2"}
Y_FILE = COHORT_DIR / "abundance_derived" / "network" / "crispr_genus_by_votu_binary.csv.gz"
W_FILE = (COHORT_DIR / "Networks" / "Bipartite_edge_probability" / "glasso"
          / f"{COHORT}_glasso_edge_probability_bacteria_virus_stars02.csv.gz")

MATRIX_EXTS = (".mtx.gz", ".mtx", ".npz", ".npy", ".csv.gz", ".csv", ".tsv", ".tsv.gz")
SIDECARS = ("__rows.csv", "__cols.csv", "__rows.tsv", "__cols.tsv")


def read_table(p, index_col=0) -> pd.DataFrame:
    """CSV/TSV (optionally gzipped). A folder resolves to the table inside it."""
    if p.is_dir():
        same = p / p.name
        cands = [same] if same.is_file() else sorted(p.glob("*.csv")) + sorted(p.glob("*.csv.gz"))
        p = cands[0] if cands else p
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    sep = "\t" if str(p).lower().rstrip(".gz").endswith(".tsv") else ","
    return pd.read_csv(p, sep=sep, index_col=index_col)


def load_matrix(p) -> np.ndarray:
    low = p.name.lower()
    if low.endswith((".mtx", ".mtx.gz")):
        import scipy.sparse as sp
        from scipy.io import mmread
        if low.endswith(".gz"):
            with gzip.open(p, "rb") as fh:
                m = mmread(fh)
        else:
            m = mmread(str(p))
        return m.toarray() if sp.issparse(m) else np.asarray(m)
    if low.endswith(".npz"):
        try:
            from scipy.sparse import load_npz as load_sparse
            return load_sparse(p).toarray()
        except Exception:
            d = np.load(p, allow_pickle=True)
            return np.asarray(d[d.files[0]])
    if low.endswith(".npy"):
        return np.asarray(np.load(p, allow_pickle=True))
    return read_table(p).to_numpy()


def load_x_bundle(base: str):
    """(entities x procs float32 matrix, entity names, proc names)."""
    entities = [str(x) for x in read_table(X_DIR / f"{base}__rows.csv").index]
    procs = [str(x) for x in read_table(X_DIR / f"{base}__cols.csv").index]
    cands = [p for p in sorted(X_DIR.iterdir())
             if p.is_file() and p.name.startswith(base)
             and not p.name.lower().endswith(SIDECARS) and p.name.lower().endswith(MATRIX_EXTS)]
    if not cands:
        raise FileNotFoundError(f"No matrix file for bundle '{base}' in {X_DIR}")
    cands.sort(key=lambda q: 0 if q.suffix.lower() in (".mtx", ".gz") else 1)
    mat = np.asarray(load_matrix(cands[0]), dtype=np.float64)
    if mat.shape == (len(procs), len(entities)):
        mat = mat.T
    elif mat.shape != (len(entities), len(procs)):
        raise ValueError(f"[{base}] shape {mat.shape} does not match "
                         f"{len(entities)} entities x {len(procs)} procs")
    print(f"  [{base}] {cands[0].name} shape={mat.shape} density={(mat != 0).mean():.4%}")
    return mat.astype(np.float32), entities, procs


def normalize_name(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def align_w(W: pd.DataFrame, bact, virus):
    """W on the (bact x virus) grid (0 where unmatched) and its 0/1 match mask."""
    Wn = W.copy()
    Wn.index = [normalize_name(x) for x in W.index]
    Wn.columns = [normalize_name(x) for x in W.columns]
    Wn = Wn.groupby(level=0).max().T.groupby(level=0).max().T   # collapse name collisions
    gb = [normalize_name(b) for b in bact]
    gv = [normalize_name(v) for v in virus]
    mask = np.outer([b in Wn.index for b in gb], [v in Wn.columns for v in gv])
    grid = Wn.reindex(index=gb, columns=gv, fill_value=0.0).to_numpy(dtype=np.float64)
    return grid, mask.astype(np.int64)


def build() -> None:
    GRAPH_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    Xb, bact_ents, bact_procs = load_x_bundle(X_BUNDLES["bacteria"])
    Xv, virus_ents, virus_procs = load_x_bundle(X_BUNDLES["virus"])
    Y_df = read_table(Y_FILE)
    W_df = read_table(W_FILE)
    print(f"  Y {Y_df.shape}  W {W_df.shape}")

    # Grid = Y rows/cols that have features.
    b_pos = {b: i for i, b in enumerate(bact_ents)}
    v_pos = {v: i for i, v in enumerate(virus_ents)}
    bact = [b for b in map(str, Y_df.index) if b in b_pos]
    virus = [v for v in map(str, Y_df.columns) if v in v_pos]
    if len(bact) < 2 or len(virus) < 2:
        raise SystemExit("Almost nothing aligned between Y and the feature bundles.")

    Y = (Y_df.loc[bact, virus].to_numpy() > 0).astype(np.int64)
    W, W_mask = align_w(W_df, bact, virus)
    print(f"  grid {len(bact)} x {len(virus)} | Y+ {Y.sum()} | W observed {W_mask.sum()}")

    np.savez(GRAPH_DATA_FILE,
             Xb=Xb[[b_pos[b] for b in bact]], Xv=Xv[[v_pos[v] for v in virus]],
             W_adjacency=W, Y_adjacency=Y, W_mask_adjacency=W_mask,
             bact_ids=np.array(bact, dtype=object), virus_ids=np.array(virus, dtype=object),
             bact_procs=np.array(bact_procs, dtype=object),
             virus_procs=np.array(virus_procs, dtype=object),
             n_bact=len(bact), n_virus=len(virus))
    print(f"wrote {GRAPH_DATA_FILE}")


# =============================================================================
# 2. preprocess: features, labels, splits
# =============================================================================
# Representation: protein clusters present in both vocabularies (shared order),
# presence/absence, pair feature = bacterium + virus.
def select_shared_clusters(data: dict):
    """Column indices into Xb and Xv of the shared clusters, same order on both."""
    Xb = np.asarray(data["Xb"], dtype=np.float32)
    Xv = np.asarray(data["Xv"], dtype=np.float32)
    b_names = np.asarray(data["bact_procs"], dtype=object).ravel().astype(str)
    v_names = np.asarray(data["virus_procs"], dtype=object).ravel().astype(str)
    shared = sorted(set(b_names) & set(v_names))
    b_pos = {n: i for i, n in enumerate(b_names)}
    v_pos = {n: i for i, n in enumerate(v_names)}
    ib = np.array([b_pos[n] for n in shared], dtype=int)
    iv = np.array([v_pos[n] for n in shared], dtype=int)
    keep = (Xb[:, ib] > 0).any(axis=0) & (Xv[:, iv] > 0).any(axis=0)   # drop all-zero sides
    if not keep.any():
        raise ValueError("No shared protein cluster is present on both sides.")
    print(f"[preprocess] {len(shared)} shared clusters, {int(keep.sum())} kept")
    return {"Xb": ib[keep], "Xv": iv[keep]}


def feature_stats(X: np.ndarray, idx: np.ndarray, source: str, names) -> pd.DataFrame:
    X = np.asarray(X, dtype=np.float64)
    means, var = X.mean(axis=0), X.var(axis=0, ddof=1)
    selected = np.zeros(X.shape[1], dtype=bool)
    selected[idx] = True
    df = pd.DataFrame({"source": source, "feature_id": np.arange(X.shape[1]),
                       "mean": means, "var": var, "score": var, "selected": selected})
    df["feature_name"] = (np.asarray(names).astype(str) if names is not None and len(names) == len(df)
                          else [f"{source}_feature_{i}" for i in df["feature_id"]])
    df["total_abundance"] = df["mean"] * X.shape[0]
    df["threshold"] = float(var[idx].min()) if idx.size else 0.0
    return df


def task_labels(data: dict, task: dict):
    """(label vector, mask or None) for one task config."""
    y = np.asarray(data[task["label_source"]]).ravel()
    if task["task_type"] == "binary":
        y = (y > float(task.get("binarize_threshold", 0.0))).astype(int) \
            if task.get("binarize") else y.astype(int)
    else:
        y = y.astype(float)
    ms = task.get("mask_source")
    return y, (None if ms is None else np.asarray(data[ms]).ravel().astype(int))


def preprocess(cfg: dict, run_id: str) -> None:
    out = run_dir(run_id, "preprocessing")
    with open(run_dir(run_id) / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    data = load_npz(GRAPH_DATA_FILE)
    W = np.asarray(data["W_adjacency"], dtype=np.float64)
    n_bact, n_virus = W.shape
    # Row-major pairs: pair i = (i // n_virus, i % n_virus).
    data["bact_idx"] = np.repeat(np.arange(n_bact), n_virus).astype(np.int64)
    data["virus_idx"] = np.tile(np.arange(n_virus), n_bact).astype(np.int64)
    data["Y_binary"] = np.asarray(data["Y_adjacency"], dtype=np.int64).flatten()
    data["W_continuous"] = W.flatten()
    data["Mask_observed"] = np.asarray(data["W_mask_adjacency"], dtype=np.int64).flatten()
    n_pairs = len(data["Y_binary"])
    print(f"[preprocess] {n_pairs} pairs | Y+ {int(data['Y_binary'].sum())} "
          f"| W observed {int(data['Mask_observed'].sum())}")

    # Features.
    idx = select_shared_clusters(data)
    arrays, stats = {}, []
    for source, names_key in (("Xb", "bact_procs"), ("Xv", "virus_procs")):
        X = np.asarray(data[source], dtype=np.float32)
        kept = X[:, idx[source]]
        arrays[f"{source}_filtered"] = (kept > 0).astype(np.float32)
        arrays[f"{source}_raw_filtered"] = kept.astype(np.float32)
        arrays[f"{source}_selected_idx"] = idx[source]
        stats.append(feature_stats(X, idx[source], source, data.get(names_key)))
    stats = pd.concat(stats, ignore_index=True)
    write_csv(stats, out / "feature_stats.csv")
    rest = {k: v for k, v in data.items() if k not in ("Xb", "Xv")}
    np.savez_compressed(out / "data_filtered.npz", **arrays, **rest)

    # Splits, stratified on the first binary task (unmasked).
    strat = next(t for t in cfg["tasks"] if t["task_type"] == "binary")
    y_strat = task_labels(data, {**strat, "mask_source": None})[0]
    rcfg = cfg["resampling"]
    write_csv(make_splits(y_strat, rcfg["n_outer"], rcfg["seed"]), out / "splits.csv")

    # Labels, one row per (sample, task).
    rows = []
    for t in cfg["tasks"]:
        y, mask = task_labels(data, t)
        rows.append(pd.DataFrame({"sample_id": np.arange(len(y)), "task": t["name"],
                                  "task_type": t["task_type"], "y": y,
                                  "mask": mask if mask is not None else 1}))
    write_csv(pd.concat(rows, ignore_index=True), out / "labels.csv")

    write_json({
        "run_id": run_id, "n_pairs": n_pairs,
        "n_Y_pos": int(data["Y_binary"].sum()), "n_W_observed": int(data["Mask_observed"].sum()),
        "representation": {"selection": "intersect", "transform": "presence_absence",
                           "combine": "sum"},
        "feature_summary": {s: {"n_input": int((stats["source"] == s).sum()),
                                "n_selected": int(stats.loc[stats["source"] == s, "selected"].sum())}
                            for s in ("Xb", "Xv")},
        "splits": {"n_outer": rcfg["n_outer"], "seed": rcfg["seed"], "stratify_on": strat["name"]},
        "tasks": [{k: t.get(k) for k in ("name", "task_type", "label_source", "mask_source")}
                  for t in cfg["tasks"]],
    }, out / "preprocess_log.json")
    print(f"[preprocess] done -> {out}")


# =============================================================================
# 3. describe: descriptive numbers and matrix CSVs for the R plots
# =============================================================================
def degree_stats(A: np.ndarray, top_k: int = 10) -> dict:
    A = (A > 0).astype(int)
    db, dv = A.sum(axis=1), A.sum(axis=0)
    total = int(db.sum())
    return {"n_edges": total,
            "genus_degree_median": float(np.median(db)), "genus_degree_mean": float(db.mean()),
            "genus_degree_max": int(db.max()) if db.size else 0,
            "genus_degree_zero": int((db == 0).sum()),
            "votu_degree_median": float(np.median(dv)), "votu_degree_mean": float(dv.mean()),
            "votu_degree_max": int(dv.max()) if dv.size else 0,
            "votu_degree_zero": int((dv == 0).sum()),
            f"top{top_k}_genera_share_of_edges":
                float(np.sort(db)[::-1][:top_k].sum() / total) if total else 0.0}


def sparsity_stats(X: np.ndarray) -> dict:
    nz = X != 0
    per = nz.sum(axis=1)
    return {"n_entities": int(X.shape[0]), "n_clusters": int(X.shape[1]),
            "density": float(nz.mean()), "clusters_per_entity_median": float(np.median(per)),
            "clusters_per_entity_min": int(per.min()), "clusters_per_entity_max": int(per.max())}


def descriptives(d: dict) -> dict:
    Y = np.asarray(d["Y_adjacency"]).astype(int)
    W = np.asarray(d["W_adjacency"]).astype(float)
    M = np.asarray(d["W_mask_adjacency"]).astype(bool)
    rows, cols = M.any(axis=1), M.any(axis=0)
    n_obs = int(M.sum())
    assert rows.sum() * cols.sum() == n_obs, "observation mask is not a block"
    y_m, w_m = Y[M], W[M]
    n_y, n_w = int((y_m > 0).sum()), int((w_m > 0).sum())
    n_both = int(((y_m > 0) & (w_m > 0)).sum())
    expected = n_y * n_w / n_obs if n_obs else 0.0
    ratio = lambda a: float(a / n_obs) if n_obs else 0.0
    out = {
        "grid": {"n_bacteria": Y.shape[0], "n_viruses": Y.shape[1], "n_pairs": Y.size,
                 "n_crispr_positive": int((Y > 0).sum()), "crispr_prevalence": float((Y > 0).mean())},
        "mask": {"n_bacteria_matched": int(rows.sum()), "n_viruses_matched": int(cols.sum()),
                 "n_observed": n_obs, "coverage_of_grid": float(n_obs / Y.size),
                 "n_crispr_positive": n_y, "crispr_prevalence": ratio(n_y),
                 "n_glasso_edge": n_w, "glasso_prevalence": ratio(n_w),
                 "w_mean": float(w_m.mean()) if n_obs else 0.0,
                 "w_sd": float(w_m.std()) if n_obs else 0.0,
                 "w_median": float(np.median(w_m)) if n_obs else 0.0,
                 "w_frac_exactly_zero": float((w_m == 0).mean()) if n_obs else 0.0,
                 "w_frac_above_half": float((w_m > 0.5).mean()) if n_obs else 0.0,
                 "n_both": n_both, "n_both_expected_if_independent": float(expected),
                 "both_observed_over_expected": float(n_both / expected) if expected else 0.0},
        "degree_grid": degree_stats(Y),
        "degree_mask": degree_stats(Y[np.ix_(rows, cols)]),
        "features_bacteria": sparsity_stats(np.asarray(d["Xb"]).astype(float)),
        "features_viruses": sparsity_stats(np.asarray(d["Xv"]).astype(float)),
    }
    return out


def describe(run_id: str) -> None:
    d = np.load(GRAPH_DATA_FILE, allow_pickle=True)
    stats = descriptives(d)
    met = run_dir(run_id, "metrics")
    write_json(stats, met / "descriptives.json")
    with open(met / "descriptives.csv", "w", encoding="utf-8") as fh:
        fh.write("scope,metric,value\n")
        for scope, block in stats.items():
            for metric, value in block.items():
                fh.write(f"{scope},{metric},{value}\n")
    m = stats["mask"]
    print(f"[describe] observed {m['n_observed']} pairs | CRISPR+ {m['n_crispr_positive']} "
          f"| W>0 {m['n_glasso_edge']} | both {m['n_both']} "
          f"(expected {m['n_both_expected_if_independent']:.0f})")

    # Headerless matrix CSVs read by plots/data.R.
    gm = run_dir(run_id, "graph_matrices")
    np.savetxt(gm / "Y_adjacency.csv", np.asarray(d["Y_adjacency"]).astype(float), delimiter=",", fmt="%d")
    np.savetxt(gm / "W_adjacency.csv", np.asarray(d["W_adjacency"]).astype(float), delimiter=",", fmt="%.6g")
    np.savetxt(gm / "W_mask_adjacency.csv", np.asarray(d["W_mask_adjacency"]).astype(int), delimiter=",", fmt="%d")
    for key, col in (("bact_ids", "bact_id"), ("virus_ids", "virus_id")):
        with open(gm / f"{key}.csv", "w", encoding="utf-8") as fh:
            fh.write(col + "\n" + "".join(f"{x}\n" for x in np.asarray(d[key]).astype(str)))
    filt = run_dir(run_id, "preprocessing") / "data_filtered.npz"
    if filt.exists():
        f = np.load(filt, allow_pickle=True)
        for key, name in (("Xb_raw_filtered", "Xb_presence.csv"), ("Xv_raw_filtered", "Xv_presence.csv")):
            np.savetxt(gm / name, (np.asarray(f[key]) > 0).astype(np.int8), delimiter=",", fmt="%d")
    print(f"[describe] done -> {met}, {gm}")


if __name__ == "__main__":
    stage = sys.argv.pop(1) if len(sys.argv) > 1 else ""
    if stage == "build":
        build()
    elif stage in ("preprocess", "describe"):
        args = stage_args().parse_args()
        if stage == "preprocess":
            preprocess(load_config(args.config), args.run_id)
        else:
            describe(args.run_id)
    else:
        raise SystemExit(__doc__)
