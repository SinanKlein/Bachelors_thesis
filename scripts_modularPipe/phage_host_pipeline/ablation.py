"""
ablation.py — the whole representation ablation

WHAT THE ABLATION IS
A 4-arm ladder over the FEATURE REPRESENTATION only. Models, folds, labels and
metrics are identical in every arm, so any arm-to-arm difference is attributable
to the representation and nothing else. Each step flips exactly one factor, and
the order respects that summation REQUIRES equal-dimension selection:

  arm 1  clr              + quantile  + concat     
  arm 2  presence/absence + quantile  + concat     
  arm 3  presence/absence + equal_dim + concat     
  arm 4  presence/absence + equal_dim + sum        

Run on BOTH binary tasks: w_class (the target of interest) and y (the learnable
positive control, which shows whether the arms move at all when signal exists).

FAMILY / GENUS
Post-hoc, zero extra model fits: each bacterial family's member-genus
predictions are collapsed to a median probability and scored by AUC/AP,
fold-wise so the family cells carry an SD. Note that a median over k members
lowers score variance and can raise AUC whether or not information was added, so
read the genus-vs-family gap with that in mind.

OUTPUTS  (all under <run>/ablation/)
  ablation_metrics.csv        arm x task x model x fold  (auc, ap, role)
  ablation_predictions.csv.gz per-pair test predictions
  family_metrics.csv          genus vs family
  ablation_log.json           config echo, timings, family map stats

Usage:
  python ablation.py --config default.yaml --run-id <run_id>
  python ablation.py --config default.yaml --run-id <run_id> --arms 1_baseline_clr_concat_quantile
  python ablation.py --config default.yaml --run-id <run_id> --skip-family
"""
from __future__ import annotations
import argparse
import functools
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

print = functools.partial(print, flush=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import GRAPH_DATA_FILE, DATASET_NAME, ablation_dir, find_taxonomy_file
from utils import (load_config, load_graph_data, write_csv, write_json,
                   make_splits_dataframe, basic_metrics, set_global_seeds,
                   derive_fold_seed, get_arms, anchor_arm)
from features import select_and_transform, pair_matrix
# label handling lives in preprocess.py — imported, never duplicated
from preprocess import flatten_pair_labels, extract_label
from models import build_model

# Majority rule for collapsing member genera into one family call: a family
# counts as positive when at least half its member genera are. This is NOT the W
# edge threshold, which is 0 everywhere (tasks[].binarize_threshold).
MAJORITY = 0.5

# 1. Features: the three switches that define an arm
def build_arm_features(data: dict, pcfg: dict, arm: dict) -> tuple[np.ndarray, dict]:
    """Pair feature matrix for one arm, via the shared features.py helpers."""
    arrays, _ = select_and_transform(data, pcfg, arm["selection"], arm["transform"],
                                     log="[ablation]   ")
    X = pair_matrix(arrays["Xb_filtered"], arrays["Xv_filtered"],
                    data["bact_idx"], data["virus_idx"], arm["combine"])
    info = {"p_b_kept": int(arrays["Xb_filtered"].shape[1]),
            "p_v_kept": int(arrays["Xv_filtered"].shape[1]),
            "feature_dim": int(X.shape[1])}
    return X.astype(np.float32), info

# 2. Family / genus
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def build_family_map(bact_ids, taxonomy_csv: Path | None):
    """Genus -> family. Unmapped genera become their own singleton family."""
    fam_of = {}
    if taxonomy_csv and Path(taxonomy_csv).exists():
        tax = pd.read_csv(taxonomy_csv, index_col=0)
        if {"Genus", "Family"}.issubset(tax.columns):
            tax = tax.dropna(subset=["Genus", "Family"])
            g2f = {_norm(g): f for g, f in zip(tax["Genus"], tax["Family"])}
            fam_of = {b: g2f.get(_norm(b)) for b in bact_ids}
        else:
            print(f"[family] WARNING: {taxonomy_csv} has no Genus/Family columns.")
    else:
        print(f"[family] WARNING: taxonomy not found at {taxonomy_csv}; "
              f"every genus becomes its own family (family arm == genus arm).")

    labels = [fam_of.get(b) or f"SOLO::{b}" for b in bact_ids]
    names = sorted(set(labels))
    idx = {f: i for i, f in enumerate(names)}
    fam_id = np.array([idx[f] for f in labels], dtype=int)
    sizes = pd.Series(labels).value_counts()
    stats = {"n_genera": int(len(bact_ids)),
             "n_mapped": int(sum(1 for l in labels if not l.startswith("SOLO::"))),
             "n_units": int(len(names)),
             "n_singletons": int((sizes == 1).sum()),
             "n_families_ge2": int((sizes >= 2).sum()),
             "largest_family_size": int(sizes.max()),
             "median_family_size": float(sizes.median())}
    print(f"[family] {stats['n_mapped']}/{stats['n_genera']} genera mapped -> "
          f"{stats['n_units']} units ({stats['n_singletons']} singletons, "
          f"largest {stats['largest_family_size']})")
    return fam_id, stats


def aggregate_once(prob: np.ndarray, y: np.ndarray, bact_of_pair: np.ndarray,
                   virus_of_pair: np.ndarray, fam_id: np.ndarray, n_virus: int,
                   truth: str) -> tuple[np.ndarray, np.ndarray]:
    """Collapse genus x virus to family x virus for ONE fold.

    Returns (median score, family truth) over the family x virus cells that have
    at least one member prediction.
    """
    key = fam_id[bact_of_pair] * n_virus + virus_of_pair
    order = np.argsort(key, kind="stable")
    key_s, prob_s, y_s = key[order], prob[order], y[order]
    bounds = np.flatnonzero(np.diff(key_s)) + 1

    scores, truths = [], []
    for g in np.split(np.arange(len(key_s)), bounds):
        if not len(g):
            continue
        p, t = prob_s[g], y_s[g]
        scores.append(np.median(p))
        frac = float(np.mean(t))
        truths.append(1 if (frac >= MAJORITY if truth == "majority" else frac > 0) else 0)
    return np.asarray(scores, float), np.asarray(truths, int)


def _binary_metrics(y, score):
    """AUC and AP for one aggregation level, plus the unit count and prevalence.
    """
    out = {"n": int(len(y)), "prevalence": float(np.mean(y))}
    if len(np.unique(y)) >= 2:
        out["auc"] = roc_auc_score(y, score)
        out["ap"] = average_precision_score(y, score)
    else:
        out["auc"] = np.nan
        out["ap"] = np.nan
    return out


def _msd(dicts, key):
    v = [d[key] for d in dicts if np.isfinite(d.get(key, np.nan))]
    return (float(np.mean(v)), float(np.std(v))) if v else (np.nan, np.nan)


def run_family(preds: pd.DataFrame, bact_of_pair, virus_of_pair, n_virus,
               fam_id, fcfg) -> pd.DataFrame:
    """Genus vs family, for every arm x task x model."""
    truth = str(fcfg.get("primary_truth", "majority")).lower()
    rows = []

    for (arm, task, model), sub in preds.groupby(["arm", "task", "model"]):
        genus_folds, fam_folds = [], []
        for fold, grp in sub.groupby("outer_fold"):
            sid = grp["sample_id"].to_numpy()
            p = grp["prob"].to_numpy(float)
            y = grp["y_true"].to_numpy(int)
            ok = np.isfinite(p)
            sid, p, y = sid[ok], p[ok], y[ok]
            if len(np.unique(y)) < 2:
                continue
            b, v = bact_of_pair[sid], virus_of_pair[sid]
            genus_folds.append(_binary_metrics(y, p))
            s, t = aggregate_once(p, y, b, v, fam_id, n_virus, truth)
            if len(np.unique(t)) >= 2:
                fam_folds.append(_binary_metrics(t, s))

        if not genus_folds:
            continue
        base = {"arm": arm, "task": task, "model": model, "family_truth": truth}
        for level, folds in (("genus", genus_folds), ("family", fam_folds)):
            if not folds:
                continue
            row = dict(base, level=level, n_folds=len(folds))
            for k in ("auc", "ap"):
                m, s = _msd(folds, k)
                row[k] = round(m, 4) if np.isfinite(m) else np.nan
                row[f"{k}_sd"] = round(s, 4) if np.isfinite(s) else np.nan
            row["n_units"] = int(np.mean([d["n"] for d in folds]))
            row["prevalence"] = round(float(np.mean([d["prevalence"] for d in folds])), 4)
            rows.append(row)

        if fam_folds:
            fam_auc, _ = _msd(fam_folds, "auc")
            gen_auc, _ = _msd(genus_folds, "auc")
            print(f"  [{arm[:22]:<22} {task:<8} {model:<16}] genus={gen_auc:.3f} "
                  f"family={fam_auc:.3f} lift={fam_auc - gen_auc:+.3f}")
    return pd.DataFrame(rows)

# 3. The arm x task x model x fold loop
def _run_arm(arm, X, tasks, models, task_data, splits, folds, seed,
             state, t0) -> None:
    """Fit every (task, model, fold) cell for ONE arm, appending into `state`.

    Every arm sees the identical folds and the identical labels; only X differs.
    That is what makes an arm-to-arm difference attributable to the
    representation and nothing else.
    """
    for t in tasks:
        y_all, mask_all = task_data[t["name"]]
        for m in models:
            # target_task pins a model to one task (the two per-task MLPs)
            if m.get("target_task") and m["target_task"] != t["name"]:
                continue
            for f in folds:
                cell = _fit_cell(arm, t, m, f, X, y_all, mask_all, splits, seed)
                if cell is None:
                    continue
                met, preds, seconds, n_tr, n_ev = cell
                state["met_rows"].extend(met)
                state["pred_frames"].append(preds)
                state["n_fits"] += 1
                print(f"[ablation] fit {state['n_fits']:>3} | {arm['name'][:22]:<22} "
                      f"{t['name']:<8} {m['name']:<16} fold {f} | "
                      f"n_tr={n_tr} n_ev={n_ev} | {seconds:.1f}s "
                      f"(total {(time.time()-t0)/60:.1f}m)")


def _fit_cell(arm, task, model_cfg, fold, X, y_all, mask_all, splits, seed):
    """One (arm, task, model, fold) fit.

    Returns (metric rows, prediction frame, seconds, n_train, n_eval), or None
    when the fold is unusable — empty after masking, or single-class in train.
    """
    sf = splits[splits["outer_fold"] == fold]
    tr = sf.loc[sf.role == "train", "sample_id"].to_numpy()
    ev = sf.loc[sf.role == "test", "sample_id"].to_numpy()
    tr = tr[mask_all[tr] == 1]
    ev = ev[mask_all[ev] == 1]
    if len(tr) == 0 or len(ev) == 0 or len(np.unique(y_all[tr])) < 2:
        return None

    params = dict(model_cfg.get("params", {}))
    params["random_state"] = derive_fold_seed(seed, int(fold), -1)
    tick = time.time()
    clf = build_model(model_cfg["class"], **params).fit(X[tr], y_all[tr])
    prob = clf.predict_proba(X[ev])

    tag = {"arm": arm["name"], "task": task["name"], "model": model_cfg["name"],
           "outer_fold": int(fold)}
    met = [dict(tag, role="test", **basic_metrics(y_all[ev], prob))]
    preds = pd.DataFrame(dict(tag, role="test", sample_id=ev,
                              y_true=y_all[ev], prob=prob))
    return met, preds, time.time() - tick, len(tr), len(ev)

# 4. Main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--arms", default=None, help="Comma list; default: every configured arm.")
    ap.add_argument("--tasks", default=None, help="Comma list; default: ablation.tasks.")
    ap.add_argument("--skip-family", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    acfg = cfg.get("ablation", {}) or {}
    if not bool(acfg.get("enabled", True)):
        print("[ablation] disabled in config; nothing to do.")
        return

    out_dir = ablation_dir(args.run_id)
    if (out_dir / "ablation_metrics.csv").exists() and not args.force:
        print(f"[ablation] SKIPPED (already saved): {out_dir/'ablation_metrics.csv'}")
        return

    seed = int(cfg.get("resampling", {}).get("seed", 42))
    set_global_seeds(seed)

    arms = get_arms(cfg)
    if args.arms:
        want = {a.strip() for a in args.arms.split(",")}
        arms = [a for a in arms if a["name"] in want]
    anchor = anchor_arm(cfg)

    task_names = ([t.strip() for t in args.tasks.split(",")] if args.tasks
                  else list(acfg.get("tasks", ["y", "w_class"])))
    tasks = [t for t in cfg["tasks"]
             if t["name"] in task_names and t.get("task_type", "binary") == "binary"]

    model_names = list(acfg.get("models", []))
    models = [m for m in cfg["models"]
              if (not model_names or m["name"] in model_names)
              and m.get("task_type") == "binary"]

    print(f"[ablation] dataset={DATASET_NAME} run_id={args.run_id}")
    print(f"[ablation] arms   : {[a['name'] for a in arms]}  (anchor: {anchor})")
    print(f"[ablation] tasks  : {[t['name'] for t in tasks]}")
    print(f"[ablation] models : {[m['name'] for m in models]}")

    # data + labels (label logic imported from preprocess.py) 
    data = flatten_pair_labels(load_graph_data(GRAPH_DATA_FILE))
    n_bact, n_virus = np.asarray(data["W_adjacency"]).shape
    bact_of_pair = np.asarray(data["bact_idx"]).ravel().astype(int)
    virus_of_pair = np.asarray(data["virus_idx"]).ravel().astype(int)

    task_data = {}
    for t in tasks:
        y, mask = extract_label(data, t["label_source"], t.get("mask_source"),
                                binarize=t.get("binarize", False), task_type="binary",
                                binarize_threshold=float(t.get("binarize_threshold", 0.0)))
        task_data[t["name"]] = (y, np.ones_like(y) if mask is None else mask)
        obs = task_data[t["name"]][1] == 1
        print(f"[ablation]   {t['name']}: n={int(obs.sum())} prevalence="
              f"{y[obs].mean():.4f}")

    # ONE split set, stratified on the first binary task, shared by every arm
    strat = tasks[0]["name"]
    splits = make_splits_dataframe(task_data[strat][0],
                                   n_outer=int(cfg["resampling"]["n_outer"]),
                                   n_inner=0,
                                   stratify=bool(cfg["resampling"].get("stratify", True)),
                                   seed=seed)
    folds = sorted(splits["outer_fold"].unique())
    print(f"[ablation] {len(folds)} shared folds, stratified on '{strat}'")

    state = {"met_rows": [], "pred_frames": [], "arm_info": {}, "n_fits": 0}
    t0 = time.time()

    for arm in arms:
        X, info = build_arm_features(data, cfg["preprocess"], arm)
        state["arm_info"][arm["name"]] = info
        print(f"\n[ablation] === {arm['name']} === "
              f"{arm['transform']} / {arm['selection']} / {arm['combine']} "
              f"-> X {X.shape}")
        _run_arm(arm, X, tasks, models, task_data, splits, folds, seed,
                 state, t0)
        del X

    met_rows, pred_frames = state["met_rows"], state["pred_frames"]
    arm_info, n_fits = state["arm_info"], state["n_fits"]

    if not met_rows:
        raise SystemExit("[ablation] nothing was fit — check tasks / masks / models.")

    metrics = pd.DataFrame(met_rows)
    preds = pd.concat(pred_frames, ignore_index=True)
    write_csv(metrics, out_dir / "ablation_metrics.csv")
    preds.to_csv(out_dir / "ablation_predictions.csv.gz", index=False,
                 compression="gzip")
    print(f"\n[ablation] wrote ablation_metrics.csv ({len(metrics)} rows) and "
          f"ablation_predictions.csv.gz ({len(preds)} rows)")

    # family / genus
    fam_stats = {}
    fcfg = acfg.get("family", {}) or {}
    if not args.skip_family and bool(fcfg.get("enabled", True)):
        bact_ids = ([str(x) for x in np.asarray(data["bact_ids"]).ravel()]
                    if "bact_ids" in data else [f"b{i}" for i in range(n_bact)])
        tax = find_taxonomy_file(DATASET_NAME)
        print(f"\n[family] taxonomy = {tax}")
        fam_id, fam_stats = build_family_map(bact_ids, tax)
        fam = run_family(preds, bact_of_pair, virus_of_pair, n_virus, fam_id, fcfg)
        if not fam.empty:
            write_csv(fam, out_dir / "family_metrics.csv")
            print(f"[family] wrote family_metrics.csv ({len(fam)} rows)")

    # summary
    test = metrics[metrics.role == "test"]
    summ = (test.groupby(["task", "arm", "model"], as_index=False)
            .agg(auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
                 ap_mean=("ap", "mean"), n_folds=("auc", "count")))
    print("\n=== AUC by task x arm x model (test) ===")
    print(summ.to_string(index=False))

    write_json({"run_id": args.run_id, "dataset": DATASET_NAME, "seed": seed,
                "anchor": anchor, "arms": {a["name"]: {**a, **arm_info.get(a["name"], {})}
                                           for a in arms},
                "tasks": [t["name"] for t in tasks],
                "models": [m["name"] for m in models],
                "n_folds": len(folds), "n_fits": n_fits,
                "family_map_stats": fam_stats,
                "elapsed_sec": round(time.time() - t0, 2)},
               out_dir / "ablation_log.json")
    print(f"\n[ablation] DONE in {(time.time()-t0)/60:.1f} min ({n_fits} fits) -> {out_dir}")


if __name__ == "__main__":
    main()
