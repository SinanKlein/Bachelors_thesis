# Phage–host pipeline

```bash
pip install -r requirements.txt
python run.py                 # one cohort, every stage
bash run_cloud.sh CRC IBD     # several cohorts in parallel (Linux)
```

Model and analysis settings are in `config.yaml`.

## Paths

No path is hardcoded. By default everything is relative to this folder:

```
phage_host_pipeline/
├── inputs/<cohort>/                      raw tables, one folder per cohort (CRC, GvHD, IBD)
│   ├── abundance_derived/procs/          protein-cluster counts + __rows/__cols sidecars
│   ├── abundance_derived/network/crispr_genus_by_votu_binary.csv.gz          (Y)
│   ├── Networks/Bipartite_edge_probability/glasso/<cohort>_glasso_edge_probability_bacteria_virus_stars02.csv.gz  (W)
│   └── graph_data.npz                    written by `python data.py build`
└── outputs/<cohort>_outputs/<run_id>/    written by every other stage
```

To keep data elsewhere (e.g. an unpacked Zenodo archive), set any of these
environment variables; relative values resolve against the working directory.

| variable | meaning | default |
|---|---|---|
| `PIPELINE_COHORT` | `CRC`, `GvHD` or `IBD` | `GvHD` |
| `PIPELINE_DATA_ROOT` | folder with one subfolder per cohort | `inputs/` |
| `PIPELINE_GRAPH_ROOT` | folder with `<cohort>/graph_data.npz` | same as `PIPELINE_DATA_ROOT` |
| `PIPELINE_RESULTS_DIR` | folder receiving `<cohort>_outputs/` | `outputs/` |
| `RSCRIPT` | Rscript executable for the plots | `Rscript` on `PATH` |

```bash
PIPELINE_DATA_ROOT=/path/to/zenodo/inputs PIPELINE_RESULTS_DIR=/path/to/outputs \
  bash run_cloud.sh CRC GvHD IBD
```

On Windows (PowerShell): `$env:PIPELINE_DATA_ROOT="D:\zenodo\inputs"; python run.py`.

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
| 1 | `python data.py build` | raw X / Y / W tables | `<cohort>/graph_data.npz` (in `PIPELINE_GRAPH_ROOT`) |
| 2 | `python data.py preprocess` | `graph_data.npz` | `preprocessing/`: `data_filtered.npz`, `feature_stats.csv`, `labels.csv`, `splits.csv` |
| 3 | `python experiment.py` | `preprocessing/` | `predictions/`, `metrics/` (`metrics_by_fold`, `cutoff_sweep`, `lambda_cv*`), `latent/<model>/` |
| 4 | `python analyses.py w_threshold` | `preprocessing/` | `metrics/w_threshold_sweep*.csv` |
| 5 | `python analyses.py shuffle` | `preprocessing/` | `shuffle_probe/` |
| 6 | `python data.py describe` | `graph_data.npz` | `metrics/descriptives.*`, `graph_matrices/` |
| 7 | `python analyses.py stability` | `preprocessing/` | `stability/` |
| 8 | `Rscript plots/<x>.R <run_id> <cohort>_outputs` | the above | `plots/` |
| 3b | `python analyses.py cka` | `latent/` | `cka/`, `plots/cka/` |

## Reproducibility

- Every stage appends a line to `<run>/provenance/stages.jsonl` (time, stage, git commit,
  uncommitted changes yes/no, SHA-256 of the config and of `graph_data.npz`) and copies the
  config it used to `provenance/config_<stage>.yaml`. The first stage of a run also writes
  `provenance/pip_freeze.txt` and, if R is available, `provenance/R_sessionInfo.txt`.
- Run ids are `YYYYMMDD_HHMMSS_<git short hash>` in both runners.
- Both runners rebuild `graph_data.npz` from the raw tables (`run_cloud.sh`: `SKIP_GRAPH_DATA=1` reuses it).
- Seeds are fixed (`resampling.seed`, per-fold seeds below). XGBoost and PyTorch results can
  still differ in the last digits across machines, thread counts and library versions.
- For a publishable run: commit, tag, then run from a clean checkout.

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
