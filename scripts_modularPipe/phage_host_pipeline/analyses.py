"""Analyses on top of a finished preprocess (and, for cka, experiment) stage.

  python analyses.py w_threshold --config C --run-id R   AUC of 1[W > t] over thresholds t
  python analyses.py shuffle     --config C --run-id R   joint model, real vs permuted Y
  python analyses.py stability   --config C --run-id R   stability selection of clusters
  python analyses.py cka         --config C --run-id R   linear CKA between latent spaces
"""
from __future__ import annotations

import itertools
import sys
import time
import warnings

import numpy as np
import pandas as pd

from common import (DATASET_NAME, binary_metrics, fold_seed, iter_folds, load_config,
                    load_run, record_stage, run_dir, set_global_seeds, stage_args, write_csv,
                    write_json)
from models import MLPYWJoint

warnings.filterwarnings("ignore", category=FutureWarning, message=".*penalty.*was deprecated.*")


# =============================================================================
# W threshold sweep
# =============================================================================
def w_threshold(cfg: dict, run_id: str) -> None:
    """For interior thresholds t, fit L2 logistic (and XGBoost) on 1[W > t]."""
    from joblib import Parallel, delayed
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    wcfg = cfg["w_threshold"]
    seed = int(cfg["resampling"]["seed"])
    set_global_seeds(seed)
    _, X, tasks, splits = load_run(run_id, cfg)
    _, w_all, mask = tasks[wcfg.get("w_task", "w_reg")]
    folds = {f: (tr[mask[tr] == 1], te[mask[te] == 1]) for f, tr, te in iter_folds(splits)}
    n_cuts = int(wcfg.get("n_cuts", 10))
    thresholds = np.linspace(0.0, 1.0, n_cuts + 2)[1:-1]
    model_names = ["l2_logistic"] + (["xgboost"] if wcfg.get("run_xgboost", True) else [])

    def one(fold, t, model_name):
        tr, ev = folds[fold]
        ytr, yev = (w_all[tr] > t).astype(int), (w_all[ev] > t).astype(int)
        rec = dict(model=model_name, threshold=float(t), outer_fold=fold,
                   n_pos_train=int(ytr.sum()), n_pos_test=int(yev.sum()),
                   n_train=len(tr), n_test=len(ev), auc=np.nan)
        if not len(tr) or not len(ev) or len(np.unique(ytr)) < 2 or len(np.unique(yev)) < 2:
            return rec
        if model_name == "l2_logistic":
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=float(wcfg.get("C", 1.0)), penalty="l2",
                                     solver=wcfg.get("solver", "lbfgs"),
                                     max_iter=int(wcfg.get("max_iter", 1000)), class_weight="balanced")
            p = clf.fit(sc.transform(X[tr]), ytr).predict_proba(sc.transform(X[ev]))[:, 1]
        else:
            from xgboost import XGBClassifier
            n_pos = int(ytr.sum())
            clf = XGBClassifier(eval_metric="logloss", n_jobs=1, verbosity=0,
                                scale_pos_weight=(len(ytr) - n_pos) / n_pos if n_pos else 1.0,
                                random_state=seed + fold, **(wcfg.get("xgb_params") or {}))
            p = clf.fit(X[tr], ytr).predict_proba(X[ev])[:, 1]
        rec["auc"] = float(roc_auc_score(yev, p))
        return rec

    t0 = time.time()
    jobs = [(f, float(t), m) for m in model_names for t in thresholds for f in folds]
    print(f"[w_threshold] {len(jobs)} fits")
    rows = Parallel(n_jobs=int(wcfg.get("n_jobs", -1)))(delayed(one)(*j) for j in jobs)
    per_split = pd.DataFrame(rows).sort_values(["model", "threshold", "outer_fold"])
    summary = (per_split.groupby(["model", "threshold"])["auc"]
               .agg(auc_mean="mean", auc_sd="std", n_splits="count").reset_index())
    met = run_dir(run_id, "metrics")
    write_csv(per_split, met / "w_threshold_sweep.csv")
    write_csv(summary, met / "w_threshold_sweep_summary.csv")
    write_json({"run_id": run_id, "thresholds": thresholds.tolist(), "models": model_names,
                **{k: v for k, v in wcfg.items()}, "elapsed_sec": round(time.time() - t0, 2)},
               met / "w_threshold_log.json")
    print(f"[w_threshold] done in {time.time() - t0:.1f}s")


# =============================================================================
# Shuffled-Y control for the joint model
# =============================================================================
def shuffle(cfg: dict, run_id: str, n_shuffles: int = 5, epochs: int | None = None) -> None:
    """Refit the joint model per fold with real Y, then with Y permuted within the
    training set. W labels and all evaluation labels stay real."""
    from scipy import stats

    t0 = time.time()
    seed = int(cfg["resampling"]["seed"])
    m_cfg = next(m for m in cfg["models"] if m["task_type"] == "joint_y_w")
    params = dict(m_cfg.get("params", {}))
    if epochs:
        params["n_epochs"] = int(epochs)
    _, X, tasks, splits = load_run(run_id, cfg)
    _, y, mask = tasks["y"]
    _, w, _ = tasks["w_class"]
    cohort = DATASET_NAME.replace("_outputs", "")

    rows = []
    for f, tr, ev in iter_folds(splits):
        tr, ev = tr[mask[tr] == 1], ev[mask[ev] == 1]
        if not len(tr) or not len(ev) or len(np.unique(y[tr])) < 2 or len(np.unique(w[tr])) < 2:
            print(f"[shuffle] fold {f}: skipped")
            continue
        for s in [None] + list(range(n_shuffles)):
            tick = time.time()
            y_tr = y[tr].copy()
            if s is not None:
                y_tr = y_tr[np.random.default_rng(10_000 * s + f).permutation(len(y_tr))]
            model = MLPYWJoint(**params, random_state=fold_seed(seed, f))
            pred = model.fit(X[tr], y_tr, w[tr].astype(float)).predict(X[ev])
            for head, truth, p in (("y", y[ev], pred["y_prob"]), ("w_class", w[ev], pred["w_pred"])):
                m = binary_metrics(truth, p)
                rows.append({"cohort": cohort, "run_id": run_id, "outer_fold": f,
                             "condition": "real" if s is None else "shuffled",
                             "shuffle": -1 if s is None else s, "head": head,
                             "auc": m["auc"], "ap": m["ap"], "n": m["n"], "n_pos": m["n_pos"],
                             "fit_sec": round(time.time() - tick, 1)})
        print(f"[shuffle] fold {f} done")
    df = pd.DataFrame(rows)

    summary = []
    for head, g in df.groupby("head", sort=False):
        real = g[g.condition == "real"].set_index("outer_fold")
        shuf = g[g.condition == "shuffled"].groupby("outer_fold")[["auc", "ap"]].mean()
        j = real[["auc", "ap"]].join(shuf, rsuffix="_shuf", how="inner").dropna()
        if len(j) < 2:
            continue
        d = j["auc"] - j["auc_shuf"]
        per_shuffle = g[g.condition == "shuffled"].groupby("shuffle")["auc"].mean()
        summary.append({
            "cohort": cohort, "head": head, "n_folds": len(j),
            "auc_real": j["auc"].mean(), "auc_shuffled": j["auc_shuf"].mean(),
            "auc_diff": d.mean(), "auc_diff_sd": d.std(), "folds_real_higher": int((d > 0).sum()),
            "p_paired": stats.ttest_rel(j["auc"], j["auc_shuf"]).pvalue,
            "ap_real": j["ap"].mean(), "ap_shuffled": j["ap_shuf"].mean(),
            "shuffled_auc_min": per_shuffle.min(), "shuffled_auc_max": per_shuffle.max(),
            "n_shuffles": int(per_shuffle.size)})

    out = run_dir(run_id, "shuffle_probe")
    df.to_csv(out / "shuffle_probe_metrics.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "shuffle_probe_summary.csv", index=False)
    write_json({"cohort": cohort, "run_id": run_id, "n_shuffles": n_shuffles,
                "epochs_override": epochs, "base_seed": seed,
                "elapsed_sec": round(time.time() - t0, 1)}, out / "shuffle_probe_log.json")
    print(pd.DataFrame(summary).round(4).to_string(index=False))


# =============================================================================
# Stability selection (complementary pairs over nodes, truncated lasso path)
# =============================================================================
def node_splits(bact_idx, virus_idx, y, n_pairs, min_pos, rng):
    """Disjoint halves of the bacteria x halves of the viruses; both halves used."""
    n_b, n_v = int(bact_idx.max()) + 1, int(virus_idx.max()) + 1
    out = []
    for _ in range(n_pairs):
        for _attempt in range(20):
            in_b, in_v = np.zeros(n_b, bool), np.zeros(n_v, bool)
            in_b[rng.permutation(n_b)[:n_b // 2]] = True
            in_v[rng.permutation(n_v)[:n_v // 2]] = True
            a = in_b[bact_idx] & in_v[virus_idx]
            c = (~in_b)[bact_idx] & (~in_v)[virus_idx]
            if min(y[a].sum(), y[c].sum()) >= min_pos and min(a.sum(), c.sum()) > 0:
                break
        else:
            print(f"[stability] WARNING: no split with >= {min_pos} positives in both halves")
        out += [a, c]
    return out


def lasso_support(X, y, q, grid, tol, max_iter):
    """Standardise, then walk lambda down the grid; return the support once it reaches q.

    Balanced class weights, as in the prediction model. With balanced weights the
    intercept-only fit is p = 0.5, so lambda_max = max_j |sum_i w_i x_ij (y_i - 0.5)| / n
    is exact even though liblinear penalises the intercept.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    n = len(y)
    if len(np.unique(y)) < 2:
        return None
    X = StandardScaler().fit_transform(X)
    w = np.where(y == 1, n / (2.0 * y.sum()), n / (2.0 * (n - y.sum())))
    lmax = float(np.abs(X.T @ (w * (y - 0.5))).max() / n)
    if not np.isfinite(lmax) or lmax <= 0:
        return None
    for r in grid:
        coef = LogisticRegression(penalty="l1", solver="liblinear", C=1.0 / (n * lmax * r),
                                  class_weight="balanced", tol=tol,
                                  max_iter=max_iter).fit(X, y).coef_[0]
        support = np.flatnonzero(np.abs(coef) > 0)
        if len(support) >= q:
            break
    return support, len(support), float(r)


def stability(cfg: dict, run_id: str, task: str | None = None) -> None:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    s = cfg["stability"]
    seed = int(s.get("seed", cfg["resampling"]["seed"]))
    set_global_seeds(seed)      # liblinear draws from the global numpy RNG
    rng = np.random.default_rng(seed)
    task = task or s.get("task", "y")
    tau, target_ev = float(s.get("tau", 0.8)), float(s.get("target_ev", 1.0))
    grid = np.geomspace(1.0, float(s.get("ratio_min", 0.01)), int(s.get("n_grid", 30)))

    data, X, tasks, _ = load_run(run_id, cfg)
    names = np.asarray(data["bact_procs"], dtype=object).ravel().astype(str)
    feat = names[np.asarray(data["Xb_selected_idx"]).ravel().astype(int)]
    rows = tasks["y"][2] == 1                       # W-observed block
    X, y = X[rows].astype(np.float64), tasks[task][1][rows]
    bact = np.unique(np.asarray(data["bact_idx"]).ravel()[rows], return_inverse=True)[1]
    virus = np.unique(np.asarray(data["virus_idx"]).ravel()[rows], return_inverse=True)[1]

    # Drop rare and constant clusters.
    support = (X > 0).sum(axis=0)
    keep = (support >= int(s.get("min_pair_support", 10))) & (X.min(axis=0) != X.max(axis=0))
    X, feat, support = X[:, keep], feat[keep], support[keep]
    p = X.shape[1]
    if p == 0:
        raise SystemExit("[stability] no feature left after filtering.")
    q = int(max(2, np.floor(np.sqrt(target_ev * (2 * tau - 1) * p))))   # from E[V] bound
    ev_bound = q ** 2 / ((2 * tau - 1) * p)
    print(f"[stability] task={task} n={len(y)} p={p} tau={tau} q={q} E[V]<={ev_bound:.2f}")

    t0 = time.time()
    subsamples = node_splits(bact, virus, y, int(s.get("n_complementary_pairs", 25)),
                             int(s.get("min_positives", 5)), rng)
    counts, reached, stops = np.zeros(p, int), [], []
    for rowmask in subsamples:
        res = lasso_support(X[rowmask], y[rowmask], q, grid,
                            float(s.get("tol", 1e-4)), int(s.get("max_iter", 1000)))
        if res is None:
            continue
        counts[res[0]] += 1
        reached.append(res[1])
        stops.append(res[2])
    used = len(reached)
    if not used:
        raise SystemExit("[stability] no usable subsample.")

    prob = counts / used
    df = pd.DataFrame({"variant": "plain", "task": task, "feature": feat,
                       "selection_prob": np.round(prob, 4), "n_selected": counts,
                       "n_subsamples": used, "pair_support": support, "stable": prob >= tau})
    df = df.sort_values("selection_prob", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    out = run_dir(run_id, "stability")
    write_csv(df, out / "stability_selection.csv")
    write_json({"run_id": run_id, "dataset": DATASET_NAME, "task": task, "seed": seed,
                "n_rows": len(y), "p": p, "prevalence": float(y.mean()), "tau": tau, "q": q,
                "target_ev": target_ev, "expected_false_positives_bound": round(ev_bound, 3),
                "B": len(subsamples), "n_subsamples_used": used,
                "mean_q_reached": float(np.mean(reached)),
                "n_subsamples_below_q": int((np.array(reached) < q).sum()),
                "median_lambda_over_lambda_max": float(np.median(stops)),
                "n_stable": int((prob >= tau).sum()), "config": s,
                "elapsed_sec": round(time.time() - t0, 2)}, out / "stability_log.json")
    print(f"[stability] mean support {np.mean(reached):.1f} (q={q}), {int((np.array(reached) < q).sum())} "
          f"subsample(s) below q, {time.time() - t0:.0f}s")
    print(f"[stability] {int((prob >= tau).sum())} stable feature(s) -> {out}")


# =============================================================================
# Linear CKA between latent spaces
# =============================================================================
def linear_cka(X, Y) -> float:
    Xc, Yc = X - X.mean(axis=0, keepdims=True), Y - Y.mean(axis=0, keepdims=True)
    nx, ny = np.linalg.norm(Xc.T @ Xc, ord="fro"), np.linalg.norm(Yc.T @ Yc, ord="fro")
    return float("nan") if nx == 0 or ny == 0 else float(
        np.linalg.norm(Xc.T @ Yc, ord="fro") ** 2 / (nx * ny))


def unit_scale(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    sd = Xc.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return Xc / sd


def cka(cfg: dict, run_id: str) -> None:
    """Per outer fold (rows encoded by one network) and pooled, raw and unit-scaled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [m["name"] for m in cfg["models"] if m.get("export_latent")]
    splits = pd.read_csv(run_dir(run_id, "preprocessing") / "splits.csv")
    test_fold = splits.loc[splits["role"] == "test", ["sample_id", "outer_fold"]]
    lat = {}
    for n in names:
        df = pd.read_csv(run_dir(run_id, "latent", n) / "latent_space.csv.gz")
        mu = [c for c in df.columns if c.startswith("mu_")]
        lat[n] = (df[["sample_id"] + mu].merge(test_fold, on="sample_id"), mu)

    by_fold, summary = [], []
    for a, b in itertools.combinations(names, 2):
        (da, ma), (db, mb) = lat[a], lat[b]
        ca, cb = [f"{c}_a" for c in ma], [f"{c}_b" for c in mb]
        pair = f"{a} vs {b}"

        def both(m):
            Xa, Xb_ = m[ca].to_numpy(dtype=float), m[cb].to_numpy(dtype=float)
            return linear_cka(Xa, Xb_), linear_cka(unit_scale(Xa), unit_scale(Xb_))

        full = both(da.merge(db, on="sample_id", suffixes=("_a", "_b")))
        vals = []
        for fold, ga in da.groupby("outer_fold"):
            m = ga.merge(db[db["outer_fold"] == fold], on="sample_id", suffixes=("_a", "_b"))
            if len(m) < 2:
                continue
            vals.append(both(m))
            by_fold.append({"pair": pair, "outer_fold": int(fold), "n": len(m),
                            "cka": vals[-1][0], "cka_scaled": vals[-1][1]})
        raw, scaled = (np.array([x[i] for x in vals], dtype=float) for i in (0, 1))
        stat = lambda fn, arr: float(fn(arr)) if len(arr) else float("nan")
        summary.append({"pair": pair, "model_a": a, "model_b": b, "n_folds": len(raw),
                        "cka_fold_mean": stat(np.nanmean, raw), "cka_fold_std": stat(np.nanstd, raw),
                        "cka_scaled_fold_mean": stat(np.nanmean, scaled),
                        "cka_scaled_fold_std": stat(np.nanstd, scaled),
                        "cka_full_pool": full[0], "cka_scaled_full_pool": full[1],
                        "n_rows_full_pool": len(da.merge(db, on="sample_id"))})
        print(f"[cka] {pair}: fold mean {summary[-1]['cka_fold_mean']:.3f} "
              f"(scaled {summary[-1]['cka_scaled_fold_mean']:.3f})")
    by_fold, summary = pd.DataFrame(by_fold), pd.DataFrame(summary)
    out, figs = run_dir(run_id, "cka"), run_dir(run_id, "plots", "cka")
    write_csv(by_fold, out / "cka_by_fold.csv")
    write_csv(summary, out / "cka_summary.csv")

    # Heatmaps of the fold-mean CKA.
    idx = {m: i for i, m in enumerate(names)}
    for col, fname, sub in (("cka_fold_mean", "cka_heatmap.png", "raw activations"),
                            ("cka_scaled_fold_mean", "cka_heatmap_scaled.png", "unit-variance coordinates")):
        mat = np.full((len(names), len(names)), np.nan)
        np.fill_diagonal(mat, 1.0)
        for _, r in summary.iterrows():
            mat[idx[r.model_a], idx[r.model_b]] = mat[idx[r.model_b], idx[r.model_a]] = r[col]
        fig, ax = plt.subplots(figsize=(1.6 * len(names) + 2, 1.6 * len(names) + 1.5))
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
        ax.set_yticks(range(len(names)), names)
        for i, j in itertools.product(range(len(names)), repeat=2):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if mat[i, j] < 0.6 else "black")
        ax.set_title(f"Linear CKA between latent spaces\n(mean across outer folds, {sub})")
        fig.colorbar(im, ax=ax, shrink=0.8, label="CKA")
        fig.tight_layout()
        fig.savefig(figs / fname, dpi=150)
        plt.close(fig)

    # Per-fold strip plot.
    pairs = list(by_fold["pair"].unique())
    fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 4), sharey=True, squeeze=False)
    for k, (ax, pair) in enumerate(zip(axes[0], pairs)):
        d = by_fold[by_fold["pair"] == pair].sort_values("outer_fold")
        ax.scatter(d["outer_fold"], d["cka"], s=40, label="raw" if k == 0 else None)
        ax.scatter(d["outer_fold"], d["cka_scaled"], s=40, facecolors="none",
                   edgecolors="tab:orange", label="unit-scaled" if k == 0 else None)
        ax.axhline(d["cka"].mean(), color="gray", linestyle="--", linewidth=1)
        ax.axhline(d["cka_scaled"].mean(), color="tab:orange", linestyle=":", linewidth=1)
        ax.set(title=pair, xlabel="outer fold", ylim=(0, 1))
        ax.title.set_fontsize(10)
    axes[0][0].set_ylabel("linear CKA")
    axes[0][0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Per-fold CKA (each point = one outer fold's test set)")
    fig.tight_layout()
    fig.savefig(figs / "cka_by_fold.png", dpi=150)
    plt.close(fig)
    print(f"[cka] done -> {out}")


if __name__ == "__main__":
    stage = sys.argv.pop(1) if len(sys.argv) > 1 else ""
    ap = stage_args()
    if stage == "shuffle":
        ap.add_argument("--n-shuffles", type=int, default=5)
        ap.add_argument("--epochs", type=int, default=None, help="override n_epochs (smoke test)")
    elif stage == "stability":
        ap.add_argument("--task", default=None)
    elif stage not in ("w_threshold", "cka"):
        raise SystemExit(__doc__)
    args = ap.parse_args()
    record_stage(args.run_id, stage, args.config)
    cfg = load_config(args.config)
    if stage == "w_threshold":
        w_threshold(cfg, args.run_id)
    elif stage == "shuffle":
        shuffle(cfg, args.run_id, args.n_shuffles, args.epochs)
    elif stage == "stability":
        stability(cfg, args.run_id, args.task)
    else:
        cka(cfg, args.run_id)
