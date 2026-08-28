# Predicting phage–bacteria interactions from proteome representations

Bachelor's thesis, LMU Munich — Sinan Klein.

Two independent proxies for bacteria–virus interaction are predicted from the same
annotation-free protein-cluster representation, on the same genus × vOTU pairs, in
three cohorts (CRC, GvHD, IBD):

| target | meaning | source |
|---|---|---|
| `y` | CRISPR linkage (binary) | spacer matches, SpacePHARER |
| `w_reg` | glasso edge probability in [0,1] | sparse graphical lasso + StARS |
| `w_class` | `1[W > 0]` | the same, thresholded |

Both partners are described only by protein clusters formed *de novo* within each
cohort with MMseqs2 — no reference database, no functional annotation.

## Headline result

`Y` is robustly learnable, `W` is essentially not, and this replicates across all
three cohorts (mean test AUC over 10 outer folds):

| task | CRC | GvHD | IBD |
|---|---|---|---|
| `y`, best MLP | 0.812 | 0.789 | 0.810 |
| `w_class`, best model | 0.587 | 0.601 | 0.608 |
| `w_reg`, R² | −0.14 | −0.14 | −0.12 |

A shuffled-`y` control confirms it: permuting the `y` training labels collapses the
`y` head to chance (−0.31 AUC, p < 1e-7 in every cohort) but leaves the `w_class`
head untouched (|Δ| ≤ 0.005), so the joint model gains nothing from shared structure.

## Layout

```
scripts_modularPipe/phage_host_pipeline/   the pipeline (see its own README)
thesis_writing/                            LaTeX source, figures, bibliography
results_modularPipe/                       plots, metrics and run logs per cohort
```

`scripts_modularPipe/phage_host_pipeline/README.md` documents the nine pipeline
stages, the representation ablation, the latent export and the tests.

## Running it

```bash
cd scripts_modularPipe/phage_host_pipeline
pip install -r requirements.txt
python tests/test_static.py       # fast source checks
python tests/test_r_static.py     # the same for the R layer
python run_all.py                 # the full pipeline for one cohort
```

The cohort is set by `COHORT` in `paths.py`, which also holds the absolute input
and output paths — edit both before running on another machine.

## Data

The input data (~1.9 GB: abundance tables, CRISPR matrices, protein clusters,
glasso networks) is **not** in this repository and is archived separately.
`build_graph_data.py` regenerates `graph_data.npz` from it.

The CRISPR and glasso networks were provided by the supervising group and were not
generated as part of this thesis.

## Results in this repository

`results_modularPipe/` carries every figure, every metric CSV and every run log —
enough to check any number in the thesis without re-running anything. The bulky
per-pair intermediates (`predictions.csv`, `latent_space.csv.gz`, `feature_stats.csv`,
`splits.csv`, `*.npz`) are excluded; the pipeline regenerates them.

The results were produced by these runs:

| cohort | run id | wall time |
|---|---|---|
| CRC | `20260827_211118_nogit` | ~2.1 h |
| GvHD | `20260828_000654_nogit` | ~11.9 h |
| IBD | `20260828_132354_nogit` | ~2.7 h |

Every run log records its environment under an `env` key. All three were produced
with Python 3.13.12, numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, torch 2.11.0
(CPU) and xgboost 3.2.0; `requirements.lock.txt` pins the full set.
