"""
Linear probe analysis for the three exported latent spaces.

This is not a new predictive baseline on the original ProC matrix. It probes the
learned latent vectors z_Y, z_W, and z_YW to quantify which target information is
linearly recoverable from each representation.

Writes:
  metrics/latent_probe_metrics.csv
  metrics/latent_probe_summary.csv

Usage:
  python probe_latent_spaces.py --config default.yaml --run-id <run_id>
"""
from __future__ import annotations
import argparse
import functools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

print = functools.partial(print, flush=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import preprocessing_dir, run_dir, metrics_dir
from utils import load_config, write_csv, basic_metrics, regression_metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from scipy.stats import spearmanr

LATENT_MODELS = ["twotower_y_only", "twotower_w_only", "twotower_yw_joint", "mlp_yw_joint"]


def _latent_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in df.columns if c.startswith("mu_")]
    if not cols:
        raise ValueError("latent file contains no mu_* columns")
    return df[cols].to_numpy(dtype=float)


def _read_splits(pre_dir: Path) -> pd.DataFrame:
    for name in ("splits.csv", "cv_splits.csv", "splits_df.csv"):
        p = pre_dir / name
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(f"Could not find splits CSV in {pre_dir}")


def _probe_y(Z_tr, y_tr, Z_ev, y_ev, seed: int) -> dict:
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_ev)) < 2:
        return {"n": len(y_ev), "n_pos": int(np.sum(y_ev)), "n_neg": int(len(y_ev) - np.sum(y_ev)),
                "auc": np.nan, "ap": np.nan, "brier": np.nan}
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs", max_iter=2000,
            class_weight="balanced", random_state=seed,
        ),
    )
    clf.fit(Z_tr, y_tr.astype(int))
    prob = clf.predict_proba(Z_ev)[:, 1]
    return basic_metrics(y_ev.astype(int), prob)


def _probe_w(Z_tr, w_tr, Z_ev, w_ev) -> dict:
    reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    reg.fit(Z_tr, w_tr.astype(float))
    pred = reg.predict(Z_ev)
    out = regression_metrics(w_ev.astype(float), pred)
    try:
        out["spearman"] = float(spearmanr(w_ev.astype(float), pred, nan_policy="omit").correlation)
    except Exception:
        out["spearman"] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("resampling", {}).get("seed", 42))
    pre_dir = preprocessing_dir(args.run_id)
    splits = _read_splits(pre_dir)
    out_dir = metrics_dir(args.run_id)

    rows = []
    for latent_model in LATENT_MODELS:
        latent_path = run_dir(args.run_id) / "latent" / latent_model / "latent_space.csv.gz"
        if not latent_path.exists():
            print(f"[probe] skipping {latent_model}: missing {latent_path}")
            continue
        df = pd.read_csv(latent_path)
        Z = _latent_matrix(df)
        sample_ids = df["sample_id"].to_numpy().astype(int)
        sid_to_row = {int(sid): i for i, sid in enumerate(sample_ids)}

        for outer_fold in sorted(splits["outer_fold"].unique()):
            tr_ids = splits[(splits["outer_fold"] == outer_fold) & (splits["role"] == "train")]["sample_id"].to_numpy().astype(int)
            ev_ids = splits[(splits["outer_fold"] == outer_fold) & (splits["role"] == "test")]["sample_id"].to_numpy().astype(int)
            tr_ids = np.array([sid for sid in tr_ids if int(sid) in sid_to_row], dtype=int)
            ev_ids = np.array([sid for sid in ev_ids if int(sid) in sid_to_row], dtype=int)
            if len(tr_ids) == 0 or len(ev_ids) == 0:
                continue
            tr = np.array([sid_to_row[int(s)] for s in tr_ids], dtype=int)
            ev = np.array([sid_to_row[int(s)] for s in ev_ids], dtype=int)

            Z_tr, Z_ev = Z[tr], Z[ev]
            y_tr = df.iloc[tr]["y"].to_numpy().astype(int)
            y_ev = df.iloc[ev]["y"].to_numpy().astype(int)
            wcls_tr = df.iloc[tr]["w"].to_numpy().astype(int)        # W class (1[W > 0.5])
            wcls_ev = df.iloc[ev]["w"].to_numpy().astype(int)
            wreg_tr = df.iloc[tr]["w_reg"].to_numpy().astype(float)  # raw continuous W
            wreg_ev = df.iloc[ev]["w_reg"].to_numpy().astype(float)
            wmask_tr = df.iloc[tr]["w_mask"].to_numpy().astype(int)
            wmask_ev = df.iloc[ev]["w_mask"].to_numpy().astype(int)

            m = _probe_y(Z_tr, y_tr, Z_ev, y_ev, seed + int(outer_fold))
            rows.append({"latent_model": latent_model, "probe_target": "y", "probe_type": "logistic_ridge",
                         "outer_fold": int(outer_fold), **m})

            if np.sum(wmask_tr == 1) >= 5 and np.sum(wmask_ev == 1) >= 2:
                # continuous-W recoverability (kept for reference)
                m = _probe_w(Z_tr[wmask_tr == 1], wreg_tr[wmask_tr == 1], Z_ev[wmask_ev == 1], wreg_ev[wmask_ev == 1])
                rows.append({"latent_model": latent_model, "probe_target": "w_reg", "probe_type": "ridge",
                             "outer_fold": int(outer_fold), **m})

                # W-class recoverability (the target the latent analysis now uses)
                m = _probe_y(Z_tr[wmask_tr == 1], wcls_tr[wmask_tr == 1],
                             Z_ev[wmask_ev == 1], wcls_ev[wmask_ev == 1], seed + int(outer_fold))
                rows.append({"latent_model": latent_model, "probe_target": "w_class", "probe_type": "logistic_ridge",
                             "outer_fold": int(outer_fold), **m})

    if not rows:
        raise SystemExit("[probe] no latent spaces found to probe")
    res = pd.DataFrame(rows)
    write_csv(res, out_dir / "latent_probe_metrics.csv")

    metric_cols = [c for c in res.columns if c not in ("latent_model", "probe_target", "probe_type", "outer_fold")]
    summary = (res.groupby(["latent_model", "probe_target", "probe_type"], as_index=False)[metric_cols]
               .mean(numeric_only=True))
    write_csv(summary, out_dir / "latent_probe_summary.csv")
    print(f"[probe] wrote {out_dir / 'latent_probe_metrics.csv'}")
    print(f"[probe] wrote {out_dir / 'latent_probe_summary.csv'}")


if __name__ == "__main__":
    main()
