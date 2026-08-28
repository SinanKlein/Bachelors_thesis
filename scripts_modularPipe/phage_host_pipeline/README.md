# Phage–host pipeline with MLP latent interaction spaces

```bash
pip install -r requirements.txt
python tests/test_static.py     # fast source checks (~1s)
python tests/test_r_static.py   # the same, for the R layer
python run_all.py               # the real thing
```

## Tests

Two suites, both source-only: no data, no torch, no R runtime. They run in
about a second between them.

**`tests/test_static.py`** — the Python modules:

* no name defined twice at module level (a second definition silently shadows
  the first)
* every call to a module-level function fits that function's signature
* every module imports cleanly, with no side effects on import

It exists because of a real bug: merging the two latent scripts produced two
`_model_targets` functions with different signatures, and the surviving one
broke four call sites. That surfaced minutes into a real run.

**`tests/test_r_static.py`** — the same idea for the R scripts, which no Python
test can reach: balanced delimiters, well-formed `library()` calls, no duplicate
top-level assignment, and every `source(...)` target present. It exists because
two merge bugs reached a live run through the R layer.

**The golden-output test was removed.** `tests/test_golden.py`, its synthetic
fixture and its hash baseline ran every sklearn stage on a small seeded dataset
and compared an MD5 of every artefact against a committed baseline, on the
contract that refactoring must never change a number. That contract is exactly
what this cull breaks — metrics columns were deliberately dropped from
`metrics_by_fold.csv`, `family_metrics.csv` and the latent export — so every
hash in the baseline is stale by construction. If you want that guarantee back,
regenerate the baseline against the current output rather than restoring the old
one, and be aware that nothing enforces the contract in the meantime.

## Pipeline order

`python run_all.py` runs these in order. Each is one script with one job, and
each is independently runnable with the same `--config` / `--run-id`, so a
failed stage can be repeated on its own.

| # | script | does |
|---|---|---|
| 1 | `build_graph_data.py` | `graph_data.npz` from the raw X / Y / W tables |
| 2 | `preprocess.py` | feature selection + transform, 10 stratified splits, labels |
| 3 | `experiment.py` | the 10-split train/test experiment |
| 4 | `analyses.py` | W-threshold sweep |
| 5 | `ablation.py` | 4-arm representation ladder + family/genus |
| 6 | `latent.py` | out-of-fold latent export |
| 7 | `graph_export.py` | adjacency CSVs + the bipartite network figure |
| 8 | `descriptives.py` | every number the Data chapter reports, as JSON + tidy CSV |
| 9 | `plot_*.R` | three scripts, one per stage; all share one theme and palette |

Supporting modules: `paths.py` (all paths), `features.py` (the one place the
representation is built), `utils.py` (IO, reducers, splits, metrics),
`models.py` (the model registry), `load_results.R` (shared R theme + palette).

The plotting layer mirrors the stages one-for-one:

| R script | plots the output of |
|---|---|
| `plot_data.R` | `preprocess.py` + `graph_export.py` — features, labels, interaction matrices, nestedness |
| `plot_results.R` | `experiment.py` — prediction performance |
| `plot_analyses.R` | `ablation.py` + `analyses.py` — arm ladder, family/genus, W threshold |

Each is several independent sections wrapped in `local({...})`, so a section's
variables cannot leak into the next and a missing input CSV skips only that
section rather than the whole script.

**`features.py` is the important one.** Selection, transform and pair assembly
live there and nowhere else, so `preprocess.py`, `experiment.py`, `latent.py`
and `ablation.py` cannot drift apart — the ablation measures exactly the
representation the pipeline uses.

The dataset is selected in `paths.py` (`DATASET_NAME`, `RAW_DATA_DIR`) and
`build_graph_data.py` (`BASE`).

## Descriptive statistics

`descriptives.py` reads `graph_data.npz` and nothing else, and writes
`metrics/descriptives.json` plus a tidy `metrics/descriptives.csv`. It exists so
that no number in the thesis has to be read off a plot title: grid and mask
sizes, CRISPR and glasso prevalence on the masked set, the mean/sd/zero-share of
`W`, the CRISPR–glasso overlap together with what independence would predict,
degree summaries on both the full grid and the mask block, and the density of
the two protein-cluster matrices.

Two things worth knowing. Degrees are reported on **both** scopes because they
differ a lot — the mask block truncates the high-degree tail, so CRC's most
connected genus has 54 CRISPR links on the full grid but 25 inside the block.
Quote whichever you mean and say which. And the script asserts that
`n_bacteria_matched * n_viruses_matched == n_observed`; if that ever fails the
observation mask is no longer a block, and the Data chapter's description of it
is wrong.

```bash
python descriptives.py --config default.yaml --run-id <run_id>
```

## Tasks

| task | type | label | notes |
|---|---|---|---|
| `y` | binary | `Y_binary` | CRISPR linkage, on the W-observed subset (intersection) |
| `w_reg` | regression | `W_continuous` | raw glasso edge probability (logit target transform) |
| `w_class` | binary | `1[W > 0]` | glasso edge present vs absent, on the W-observed subset |

All three tasks are masked to the **W-observed intersection** (`Mask_observed`): every model — `y`-only, `w`-only, and the joint — is trained, evaluated, and exported on the identical pair set. Because `Y` is defined on the whole grid, `Y-observed ∩ W-observed == W-observed`, so a single `Mask_observed` on `y` achieves the intersection. Revert `y`'s `mask_source` to `null` in `default.yaml` to train `Y` on the full grid.

Key finding replicated across cohorts: `Y` is robustly learnable, `W` is essentially not. The latent-space analysis uses **`w_class`** (never `w_reg`).

## Metrics

**AUC and AP for the binary tasks, R² for `w_reg`, and nothing else.** Precision,
recall, F1, the top-k discovery diagnostics, MSE, RMSE, MAE and the target
descriptives were all removed. None of them was read by a plot, a table or the
write-up, and every one of the binary ones needed a probability cutoff that no
claim in the thesis rests on — AUC and AP are ranking metrics and need no cutoff,
which is precisely why they are the two that survived. `n`, `n_pos` and `n_neg`
stay: they are fold bookkeeping, not scores.

The **generalization gap went with them.** `evaluation.log_train_metrics`, the
`role="train"` rows it emitted in `metrics_by_fold.csv` and `ablation_metrics.csv`,
and the four `gap_*.png` figures are gone, so every fit now predicts once instead
of twice. `metrics_by_fold.csv` contains only `role="test"` rows; the R scripts
still filter on `role` so an older run's CSV keeps plotting.

The threshold sweep is untouched: `cutoff_sweep.csv` still carries the full
precision/recall/F1 curve over the cutoff grid, if a threshold-dependent number
is ever needed.

## Models

Each model has a `task_type`. A model may also set **`target_task`** to be trained / evaluated / plotted on a single task only; without it, a model runs on every task matching its type (the baseline sweep).

**Baselines**
- `sparse_logistic`, `xgb_clf` — binary, run on both `y` and `w_class`.
- `linreg`, `xgb_reg` — regression, run on `w_reg`.
- `mlp_clf_y`, `mlp_clf_w_class`, `mlp_reg` — plain MLPs, one pinned per task (`y`, `w_class`, `w_reg`).

**Latent MLPs** (exported for the latent-space analysis; same architecture as the plain MLPs)
- `mlp_latent_y` — trained and plotted only on `y`.
- `mlp_latent_w_class` — trained and plotted only on `w_class`.
- `mlp_latent_yw` — joint (`y` + `w_class`), evaluated on both.

Every model is tested and plotted strictly on the task(s) it is trained to predict — a `y` latent never appears in a `W` metric and vice versa.

## Ablation

Everything lives in **one script, `ablation.py`**. It loads `graph_data.npz`,
runs every arm × task × model × fold in memory, does the family/genus
aggregation on its own predictions, and writes tidy CSVs. No subprocesses, no
per-arm folders, no intermediate npz files. `preprocess.py` and `experiment.py`
know nothing about arms.

A 4-arm ladder over the FEATURE REPRESENTATION only. Models, folds, labels and
metrics are identical in every arm, so any difference is attributable to the
representation and nothing else. Each step flips exactly one factor, and the
order respects that summation REQUIRES equal-dimension selection.

| arm | transform | selection | combine | isolates |
|---|---|---|---|---|
| 1 baseline | clr | quantile | concat | — (the anchor) |
| 2 +presence | presence/absence | quantile | concat | presence vs abundance |
| 3 +equal-dim | presence/absence | equal_dim | concat | selection scheme |
| 4 full | presence/absence | equal_dim | **sum** | sum vs concat |

Run on **both** binary tasks: `w_class` (the target of interest) and `y` (the
learnable positive control, which shows whether the arms move at all when signal
is present). One split set, stratified on `y`, is shared by every arm and task —
identical folds are what make the arms comparable.

```bash
python ablation.py --config default.yaml --run-id <run_id>
python ablation.py --config default.yaml --run-id <run_id> --arms 2_presence_concat_quantile
Rscript plot_analyses.R <run_id> <DATASET_NAME>
```

Outputs, all in one folder:

| file | contents |
|---|---|
| `ablation/ablation_metrics.csv` | arm × task × model × fold (auc, ap, role) |
| `ablation/ablation_predictions.csv.gz` | per-pair test predictions |
| `ablation/family_metrics.csv` | genus vs family |
| `ablation/ablation_log.json` | config echo, timings, family map stats |
| `plots/ablation/arm_ladder.png` | drawn by `plot_analyses.R` |
| `plots/ablation/family_vs_genus.png` | drawn by `plot_analyses.R` |

**The anchor is arm 4**, mirrored in the `preprocess:` block so the pipeline and
the ablation describe the same representation. It was selected on the **`y`**
control (never on `w_class`, so the W claims stay uncontaminated): arm 4 has the
highest mean AUC for both nonlinear models and beats arm 1 paired across the
shared folds (mlp +0.044 p=0.006, xgb +0.039 p=0.007).

Carry these caveats into the write-up: arms 2, 3 and 4 are **not** separable
from each other (arm4 vs arm3 p=0.46 / 0.40), so the whole gain over arm 1 is
already delivered by presence/absence alone; `equal_dim` costs the linear
baseline ~0.02 AUC; and on `w_class` the arms are indistinguishable from one
another, so the choice is irrelevant to the W result — which is the point of the
ablation.

## Family / genus

Runs inside `ablation.py`, purely post-hoc, **zero extra model fits**. Models
are fit at genus level as usual, then each bacterial family's member predictions
are collapsed to a median probability and scored by AUC/AP.

Aggregation is **fold-wise**, so family cells carry an SD and are comparable to
the genus cells. Read the genus-vs-family gap with one caveat in mind: a median
over *k* members lowers score variance and can raise AUC whether or not
information was added — on CRC, 18 of 27 family units are singletons and the
largest holds 28 of 75 genera, exactly the shape averaging alone would produce.
On the data, family AUC exceeds genus in only 19 of 72 cohort x arm x task x
model cells (mean lift -0.011), so the aggregation does not help.

A shuffled-family control (genera reassigned to fake families of the same size
distribution) was removed: as implemented it compared the fold-averaged family
AUC against percentiles of the pooled per-fold x per-shuffle null, which is far
wider than the sampling distribution of a mean and so never rejected. The direct
comparison above is the reading that carries the result.

`majority` truth is the default: `any` goes near-degenerate for large families,
while `majority` is prevalence-matched to the genus level.

## Latent-space export

The exported latent is the post-ReLU hidden activation of the MLP trunk (`encode_latent`). Latents are produced **out-of-fold**: each pair is encoded by a model trained only on that pair's outer-fold training set, so the encoding is leakage-free. Each latent space is encoded over the pairs relevant to its task; with `y` masked to `Mask_observed`, all models (`y`, `w_class`, joint) share the same W-observed intersection.

Run manually:

```bash
python latent.py --config default.yaml --run-id <run_id>
python latent.py --config default.yaml --run-id <run_id> --model mlp_latent_yw
```

Outputs under `latent/<model_name>/`:

| File | Meaning |
|---|---|
| `latent_space.csv.gz` | one row per encoded pair; `mu_*`, `y`, `w`, predictions, agreement class |
| `latent_export_log.json` | metadata and `plot_targets` |

**There is deliberately no linear probe.** Each fold trains its own network from its own initialisation, so the 64 hidden units are not aligned across folds; a probe fitted on latents pooled from several encoders reads inconsistent columns and understates decodability. Measured on CRC `mlp_latent_y`, holding the training set fixed and changing only the test encoder moves the probe from 0.679 (same network) to 0.566 (different network). The claim a probe would support comes free from the architecture: the network's output layer is itself a linear map on this latent and reaches AUC 0.81.

**And no PCA either — it was removed.** The same non-alignment that sinks the probe sinks a PCA fitted on pooled latents: the components mix coordinate systems across folds, so the panels could only ever be illustration, never evidence, and an illustration of an artefact is worse than no figure. Gone with it: the `PC1`/`PC2` columns, `pca_loadings.csv`, `feature_pc_correlations.csv`, the explained-variance entry in the export log, and `plot_latent_space.R` — which drew nothing but PCA scatters, so the whole script went. What remains is the export itself: the raw out-of-fold `mu_*` vectors, ready for whatever analysis replaces it.
