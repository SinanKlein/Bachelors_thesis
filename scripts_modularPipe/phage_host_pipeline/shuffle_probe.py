"""
Shuffled-Y control for the joint (Y + W-class) MLP.

This stage separates them. It refits the SAME joint model on the SAME folds,
once with the real Y labels and several times with the Y labels randomly
permuted inside the training set. Real W labels throughout; evaluation always
against real held-out labels.

  real lift  >  shuffled lift   ->  (a) shared structure
  real lift  ~= shuffled lift   ->  (b) regularisation


    <run_dir>/shuffle_probe/shuffle_probe_metrics.csv   one row per fold x condition x head
    <run_dir>/shuffle_probe/shuffle_probe_summary.csv   real vs shuffled, paired over folds
    <run_dir>/shuffle_probe/shuffle_probe_log.json      config echo + validation vs this run

USAGE
-----
    python shuffle_probe.py --config default.yaml --run-id <run_id>
    python shuffle_probe.py --config default.yaml --run-id <run_id> --n-shuffles 10
    python shuffle_probe.py --config default.yaml --run-id <run_id> --epochs 10 --n-shuffles 1

"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from paths import DATASET_NAME, CONFIG_DIR, run_dir          # noqa: E402
from models import MLPYWJoint                                # noqa: E402
from utils import derive_fold_seed, basic_metrics            # noqa: E402

# Fallback hyperparameters: the `mlp_latent_yw` entry of default.yaml. Only used
# if neither the run's frozen config nor --config can be read.
JOINT_PARAMS_FALLBACK = dict(
    hidden_dim=64, dropout=0.3, batch_size=512, lr=3e-4,
    weight_decay=1e-3, n_epochs=200, lambda_y=1.0, lambda_w=1.0,
)
BASE_SEED_FALLBACK = 42

# Config and inputs
def load_joint_params(rdir: Path, config_path: Path):
    """mlp_latent_yw's params and the base seed.

    The run's frozen config.yaml wins over --config: it is what this run was
    actually fitted with, and default.yaml may have moved on since.
    """
    for src in (rdir / "config.yaml", config_path):
        if src is None or not Path(src).exists():
            continue
        try:
            import yaml
            cfg = yaml.safe_load(Path(src).read_text())
        except Exception:
            continue
        params = dict(JOINT_PARAMS_FALLBACK)
        for m in cfg.get("models", []):
            if m.get("name") == "mlp_latent_yw":
                params.update(m.get("params", {}) or {})
                break
        seed = int(cfg.get("resampling", {}).get("seed", BASE_SEED_FALLBACK))
        return params, seed, str(src)
    return dict(JOINT_PARAMS_FALLBACK), BASE_SEED_FALLBACK, "fallback"


def load_run(rdir: Path):
    """X (pair features), y, w_class, observation mask, and the fold table.

    X is rebuilt exactly as the pipeline builds it under the anchor
    representation: presence/absence, equal_dim selection, combine = sum, so
    X[pair] = Xb_filtered[bact] + Xv_filtered[virus].
    """
    npz_path = rdir / "preprocessing" / "data_filtered.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found. Run preprocess.py for this run id first.")
    npz = np.load(npz_path, allow_pickle=True)

    Xb, Xv = npz["Xb_filtered"], npz["Xv_filtered"]
    if Xb.shape[1] != Xv.shape[1]:
        raise ValueError(
            f"combine='sum' needs equal feature dims, got Xb={Xb.shape[1]} "
            f"Xv={Xv.shape[1]}. This run did not use the equal_dim anchor.")
    X = (Xb[npz["bact_idx"]] + Xv[npz["virus_idx"]]).astype(np.float32)

    y = npz["Y_binary"].astype(int)
    w = (npz["W_continuous"] > 0.0).astype(int)      # w_class, threshold 0
    mask = npz["Mask_observed"].astype(int)          # both tasks use this mask

    splits = pd.read_csv(rdir / "preprocessing" / "splits.csv")
    splits = splits[splits["inner_fold"] == -1]      # outer folds only
    return X, y, w, mask, splits

# One fold, one condition
def fit_one(X, y, w, tr, ev, params, seed, shuffle_seed=None):
    """Fit the joint model on one fold; return metrics for both heads.

    shuffle_seed=None  -> real Y labels.
    shuffle_seed=int   -> Y training labels permuted with that seed. W labels
                          and every evaluation label stay real.
    """
    y_tr = y[tr].copy()
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        y_tr = y_tr[rng.permutation(len(y_tr))]

    model = MLPYWJoint(**params, random_state=seed)
    model.fit(X[tr], y_tr, w[tr].astype(float))
    pred = model.predict(X[ev])
    return {
        "y":       basic_metrics(y[ev], pred["y_prob"]),
        "w_class": basic_metrics(w[ev], pred["w_pred"]),
    }


def run_probe(rdir: Path, run_id: str, params, base_seed, n_shuffles, verbose=True):
    X, y, w, mask, splits = load_run(rdir)
    folds = sorted(splits["outer_fold"].unique())
    cohort = DATASET_NAME.replace("_outputs", "")
    rows = []

    if verbose:
        print(f"[shuffle_probe] dataset={DATASET_NAME}  run_id={run_id}")
        print(f"[shuffle_probe] pairs={len(y)}  observed={int(mask.sum())}  "
              f"features={X.shape[1]}  folds={len(folds)}  "
              f"epochs={params['n_epochs']}  n_shuffles={n_shuffles}")
        print(f"[shuffle_probe] {len(folds) * (1 + n_shuffles)} model fits to do",
              flush=True)

    for f in folds:
        sf = splits[splits["outer_fold"] == f]
        tr = sf.loc[sf["role"] == "train", "sample_id"].to_numpy()
        ev = sf.loc[sf["role"] == "test",  "sample_id"].to_numpy()
        # The joint model trains where both targets are observed; both tasks are
        # masked to Mask_observed, so that is simply mask == 1.
        tr = tr[mask[tr] == 1]
        ev = ev[mask[ev] == 1]
        if len(tr) == 0 or len(ev) == 0:
            print(f"[shuffle_probe]   fold {f}: empty after masking, skipped")
            continue
        if len(np.unique(y[tr])) < 2 or len(np.unique(w[tr])) < 2:
            print(f"[shuffle_probe]   fold {f}: single-class training target, skipped")
            continue

        seed = derive_fold_seed(base_seed, int(f), -1)
        for cond, s in [("real", None)] + [("shuffled", i) for i in range(n_shuffles)]:
            # Same network seed in every condition, so the ONLY difference
            # between real and shuffled is the Y label vector.
            shuffle_seed = None if s is None else 10_000 * s + int(f)
            tick = time.time()
            met = fit_one(X, y, w, tr, ev, params, seed, shuffle_seed)
            dt = time.time() - tick
            for head, m in met.items():
                rows.append({
                    "cohort": cohort, "run_id": run_id, "outer_fold": int(f),
                    "condition": cond, "shuffle": (-1 if s is None else int(s)),
                    "head": head, "auc": m["auc"], "ap": m["ap"],
                    "n": m["n"], "n_pos": m["n_pos"],
                    "fit_sec": round(dt, 1),
                })
        if verbose:
            cur = [r for r in rows if r["outer_fold"] == f and r["head"] == "w_class"]
            real = next(r["auc"] for r in cur if r["condition"] == "real")
            shuf = np.mean([r["auc"] for r in cur if r["condition"] == "shuffled"])
            print(f"[shuffle_probe]   fold {f}: n_tr={len(tr):6d} n_ev={len(ev):6d} | "
                  f"w_class real={real:.4f}  shuffled={shuf:.4f}  ({dt:.0f}s/fit)",
                  flush=True)

    return pd.DataFrame(rows)

# Reporting
def summarise(df):
    """Per head: real vs shuffled, paired across the shared folds."""
    from scipy import stats

    out = []
    for head, g in df.groupby("head", sort=False):
        real = g[g.condition == "real"].set_index("outer_fold")
        shuf = (g[g.condition == "shuffled"]
                .groupby("outer_fold")[["auc", "ap"]].mean())     # avg over shuffles
        j = real[["auc", "ap"]].join(shuf, rsuffix="_shuf", how="inner").dropna()
        if len(j) < 2:
            continue
        d_auc = j["auc"] - j["auc_shuf"]
        try:
            p_auc = stats.ttest_rel(j["auc"], j["auc_shuf"]).pvalue
        except Exception:
            p_auc = np.nan
        per_shuffle = g[g.condition == "shuffled"].groupby("shuffle")["auc"].mean()
        out.append({
            "cohort": g["cohort"].iloc[0], "head": head, "n_folds": len(j),
            "auc_real": j["auc"].mean(), "auc_shuffled": j["auc_shuf"].mean(),
            "auc_diff": d_auc.mean(), "auc_diff_sd": d_auc.std(),
            "folds_real_higher": int((d_auc > 0).sum()),
            "p_paired": p_auc,
            "ap_real": j["ap"].mean(), "ap_shuffled": j["ap_shuf"].mean(),
            "shuffled_auc_min": per_shuffle.min(),
            "shuffled_auc_max": per_shuffle.max(),
            "n_shuffles": int(per_shuffle.size),
        })
    return pd.DataFrame(out)


def validate_against_run(df, rdir: Path):
    """Compare the real condition to this run's own mlp_latent_yw numbers.

    A sanity check that the stage is fitting the same thing, not a hash test:
    torch is not bit-reproducible across platforms.
    """
    f = rdir / "metrics" / "metrics_by_fold.csv"
    if not f.exists():
        return None
    ref = pd.read_csv(f)
    ref = ref[(ref["role"] == "test") & (ref["model"] == "mlp_latent_yw")]
    got = df[df["condition"] == "real"]
    rep = {}
    for head in ["y", "w_class"]:
        a = got[got["head"] == head].set_index("outer_fold")["auc"]
        b = ref[ref["task"] == head].set_index("outer_fold")["auc"]
        j = pd.concat([a, b], axis=1, keys=["probe", "run"]).dropna()
        if len(j):
            rep[head] = {
                "probe_mean": round(float(j.probe.mean()), 4),
                "run_mean": round(float(j.run.mean()), 4),
                "max_abs_fold_diff": round(float((j.probe - j.run).abs().max()), 4),
            }
    return rep

def main():
    ap = argparse.ArgumentParser(
        description="Shuffled-Y control for the joint model, for one cohort and one run.")
    ap.add_argument("--config", type=Path, default=CONFIG_DIR / "default.yaml",
                    help="only used if the run folder has no frozen config.yaml")
    ap.add_argument("--run-id", required=True,
                    help="run id under RESULTS_DIR / DATASET_NAME")
    ap.add_argument("--n-shuffles", type=int, default=5,
                    help="permutations per fold; each is a full model fit")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override n_epochs, e.g. --epochs 10 for a smoke test")
    args = ap.parse_args()

    t0 = time.time()
    rdir = run_dir(args.run_id)
    params, base_seed, cfg_src = load_joint_params(rdir, args.config)
    if args.epochs:
        params["n_epochs"] = int(args.epochs)
    print(f"[shuffle_probe] params from {cfg_src}")

    df = run_probe(rdir, args.run_id, params, base_seed, args.n_shuffles)
    if df.empty:
        print("[shuffle_probe] no folds produced results; nothing written.")
        return

    out = rdir / "shuffle_probe"
    out.mkdir(parents=True, exist_ok=True)
    summary = summarise(df)
    df.to_csv(out / "shuffle_probe_metrics.csv", index=False)
    summary.to_csv(out / "shuffle_probe_summary.csv", index=False)

    validation = validate_against_run(df, rdir)
    json.dump({
        "cohort": DATASET_NAME.replace("_outputs", ""),
        "dataset_name": DATASET_NAME,
        "run_id": args.run_id,
        "n_shuffles": args.n_shuffles,
        "epochs_override": args.epochs,
        "params_source": cfg_src,
        "base_seed": base_seed,
        "elapsed_sec": round(time.time() - t0, 1),
        "validation_vs_run_mlp_latent_yw": validation,
        "note": ("Real vs shuffled-Y control for the joint model. The Y training labels "
                 "are permuted within each fold; W labels and all evaluation labels are "
                 "real. Computed from this run's own splits and features, so no refitting "
                 "of the pipeline is required."),
    }, open(out / "shuffle_probe_log.json", "w"), indent=2)

    print("\n" + "=" * 78)
    print(f"REAL vs SHUFFLED Y — {DATASET_NAME} / {args.run_id}  (AUC, paired over folds)")
    print("=" * 78)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(summary.round(4).to_string(index=False))
    print("\nReading the w_class row:")
    print("  auc_diff ~ 0 and p large  -> the joint gain is regularisation")
    print("  auc_diff > 0 and p small  -> Y carries structure W can use")
    print("The y row is a manipulation check: shuffled should collapse to ~0.5.")
    print(f"\n[shuffle_probe] written to {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
