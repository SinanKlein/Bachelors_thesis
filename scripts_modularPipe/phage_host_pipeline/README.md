# Phage–host pipeline

```bash
pip install -r requirements.txt
python run.py                 # one cohort, every stage
bash run_cloud.sh CRC IBD     # several cohorts in parallel (Linux)
```

Set the cohort and data/results roots in `common.py` (or with the environment
variables `PIPELINE_COHORT`, `PIPELINE_DATA_ROOT`, `PIPELINE_SPLITS_ROOT`,
`PIPELINE_RESULTS_DIR`). Model and analysis settings are in `config.yaml`.

## Files

| file | contents |
|---|---|
| `common.py` | paths, IO, seeds, splits, metrics, `load_run` |
| `data.py` | `build`, `preprocess`, `describe` stages |
| `models.py` | model registry (sparse logistic, ridge, XGBoost, MLPs) |
| `experiment.py` | 10-fold experiment and out-of-fold latent export |
| `analyses.py` | `w_threshold`, `shuffle`, `stability`, `cka` |
| `run.py`, `run_cloud.sh` | runners |
| `plots/common.R` | R paths, theme, palette |
| `plots/data.R`, `plots/results.R`, `plots/analyses.R` | figures |

## Stages

Every stage takes `--config config.yaml --run-id <id>` (except `data.py build`)
and can be rerun on its own.

| # | command | reads | writes (under `<results>/<cohort>_outputs/<run_id>/`) |
|---|---|---|---|
| 1 | `python data.py build` | raw X / Y / W tables | `graph_data.npz` (in `PIPELINE_SPLITS_ROOT`) |
| 2 | `python data.py preprocess` | `graph_data.npz` | `preprocessing/`: `data_filtered.npz`, `feature_stats.csv`, `labels.csv`, `splits.csv` |
| 3 | `python experiment.py` | `preprocessing/` | `predictions/`, `metrics/` (`metrics_by_fold`, `cutoff_sweep`, `lambda_cv*`), `latent/<model>/` |
| 4 | `python analyses.py w_threshold` | `preprocessing/` | `metrics/w_threshold_sweep*.csv` |
| 5 | `python analyses.py shuffle` | `preprocessing/` | `shuffle_probe/` |
| 6 | `python data.py describe` | `graph_data.npz` | `metrics/descriptives.*`, `graph_matrices/` |
| 7 | `python analyses.py stability` | `preprocessing/` | `stability/` |
| 8 | `Rscript plots/<x>.R <run_id> <cohort>_outputs` | the above | `plots/` |
| – | `python analyses.py cka` | `latent/` | `cka/`, `plots/cka/` (not run by `run.py`) |

## Representation

Protein clusters present in both the bacterial and viral vocabularies (same
order on both sides, clusters all-zero on one side dropped), presence/absence,
pair feature = bacterium vector + virus vector.

## Tasks

| task | type | label | mask |
|---|---|---|---|
| `y` | binary | `Y_binary` (CRISPR linkage) | `Mask_observed` |
| `w_reg` | regression | `W_continuous` (glasso edge probability, logit-transformed for fitting) | `Mask_observed` |
| `w_class` | binary | `1[W > 0]` | `Mask_observed` |

Metrics: AUC and AP for binary tasks (plus the cutoff sweep), R² for regression.

## Models

A model runs on every task of its `task_type`, or only on `target_task` if set.
Every fit in fold `f` uses the seed `seed * 10000 + f * 100`.

- `sparse_logistic` (L1, penalty by inner CV), `xgb_clf` — `y`, `w_class`
- `linreg`, `xgb_reg`, `mlp_reg` — `w_reg`
- `mlp_clf_y`, `mlp_clf_w_class` — one-hidden-layer MLPs on `y` / `w_class`
- `mlp_latent_yw` — shared hidden layer with a `y` head and a `w_class` head

Models with `export_latent: true` write the post-ReLU hidden activation of every
test pair to `latent/<model>/latent_space.csv.gz`, encoded by the network of the
fold that pair was tested in. Hidden units are not aligned across folds.
