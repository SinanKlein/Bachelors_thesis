# load_results.R

suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
  library(dplyr)
  library(tibble)
  library(jsonlite)
})

# Directory of the running script. The plot scripts inline a two-line version
# of this to bootstrap (they need it before this file is sourced); this is the
# canonical definition for everything afterwards.
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  hit <- grep("^--file=", args, value = TRUE)
  if (length(hit)) return(normalizePath(dirname(sub("^--file=", "", hit[1]))))
  if (sys.nframe() > 0) {
    f <- try(sys.frame(1)$ofile, silent = TRUE)
    if (!inherits(f, "try-error") && !is.null(f)) return(normalizePath(dirname(f)))
  }
  normalizePath(".")
}

# user should fill
RESULTS_DIR <- "C:/Sinan_Klein/LMU/lmu_thesis/results_modularPipe"   # same path as Python's RESULTS_DIR

# Dataset name. All outputs live under RESULTS_DIR/<DATASET_NAME>/<run_id>.
# run_all.py passes it as the 2nd CLI arg; falls back to "IBD" if absent.
.get_dataset_name <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) >= 2 && nzchar(args[2])) return(args[2])
  "IBD_outputs"
}
DATASET_NAME <- .get_dataset_name()

# load helpers
run_path <- function(run_id, ...) {
  file.path(RESULTS_DIR, DATASET_NAME, run_id, ...)
}

load_preprocessing <- function(run_id) {
  list(
    feature_stats = read_csv(run_path(run_id, "preprocessing", "feature_stats.csv"),
                             show_col_types = FALSE),
    labels        = read_csv(run_path(run_id, "preprocessing", "labels.csv"),
                             show_col_types = FALSE),
    splits        = read_csv(run_path(run_id, "preprocessing", "splits.csv"),
                             show_col_types = FALSE),
    summary       = jsonlite::fromJSON(run_path(run_id, "preprocessing",
                                                "preprocess_log.json"))
  )
}

load_results <- function(run_id) {
  list(
    predictions = read_csv(run_path(run_id, "predictions", "predictions.csv"),
                           show_col_types = FALSE),
    metrics     = read_csv(run_path(run_id, "metrics", "metrics_by_fold.csv"),
                           show_col_types = FALSE),
    cutoff      = read_csv(run_path(run_id, "metrics", "cutoff_sweep.csv"),
                           show_col_types = FALSE)
  )
}

# Shared color scheme
# One fixed color per variable, reused by EVERY plot script so a given model /
# source always renders in the same color across the whole thesis. xgboost is
# always orange, the logistic family always blue, etc.
SOURCE_COLORS <- c(Xb = "#1F77B4", Xv = "#FF7F0E")   # bacteria = blue, virus = orange

# Aggregation level (genus vs family) in the ablation plots
LEVEL_COLORS <- c(genus = "#9EC1DC", family = "#2B5D8A")


# Shared publication theme
# FIG_SCALE multiplies every text size in the theme. The cohort panels are
FIG_SCALE <- 2.0

pub_theme <- theme_minimal(base_size = 12 * FIG_SCALE) +
  theme(
    plot.title       = element_text(face = "bold", size = 13 * FIG_SCALE),
    plot.subtitle    = element_text(color = "grey40", size = 10 * FIG_SCALE),
    plot.caption     = element_text(color = "grey55", size = 8 * FIG_SCALE, hjust = 0),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "grey92", linewidth = 0.3 * FIG_SCALE),
    strip.background = element_rect(fill = "grey95", color = NA),
    strip.text       = element_text(face = "bold", size = 11 * FIG_SCALE),
    axis.title       = element_text(size = 11 * FIG_SCALE),
    axis.text        = element_text(size = 10 * FIG_SCALE, color = "grey25"),
    legend.position  = "right",
    legend.title     = element_text(size = 10 * FIG_SCALE, face = "bold"),
    legend.text      = element_text(size = 9 * FIG_SCALE),
    plot.margin      = margin(10, 12, 10, 10)
  )

# For plots whose x axis is a discrete category (models, arms): rotate the tick
# labels and drop the vertical grid lines, which carry no meaning there.
theme_categorical_x <- function() {
  theme(panel.grid.major.x = element_blank(),
        axis.text.x = element_text(size = 11 * FIG_SCALE, color = "grey15",
                                   angle = 20, hjust = 1))
}

MODEL_COLORS <- c(
  #  linear / logistic family (blue) 
  sparse_logistic    = "#4C78A8",
  l2_logistic        = "#4C78A8",   # W-threshold sweep logistic
  linreg             = "#3B6EA5",
  #  xgboost family (orange) everywhere 
  xgb_clf            = "#F58518",
  xgb_reg            = "#F58518",
  xgboost            = "#F58518",   # W-threshold sweep xgboost
  #  plain MLP baselines (green) 
  mlp_clf_y          = "#54A24B",
  mlp_clf_w_class    = "#54A24B",
  mlp_reg            = "#54A24B",
  #  latent MLPs (distinct accents) 
  mlp_latent_y       = "#E45756",
  mlp_latent_w_class = "#B279A2",
  mlp_latent_yw      = "#9D755D"
)

# Fallback palette for any model name not listed above (keeps plots from erroring
# if a new model is added). Deterministic by sorted name.
.fallback_palette <- c("#72B7B2", "#EECA3B", "#BAB0AC", "#FF9DA6", "#8C564B",
                       "#17BECF", "#BCBD22", "#7F7F7F")

# Return a named color vector covering exactly the supplied model names.
pal_for <- function(models) {
  models <- sort(unique(as.character(models)))
  cols <- MODEL_COLORS[models]
  missing <- which(is.na(cols))
  if (length(missing)) {
    cols[missing] <- .fallback_palette[(seq_along(missing) - 1) %% length(.fallback_palette) + 1]
  }
  setNames(as.character(cols), models)
}

# Latent-space MLPs are named mlp_latent_*; everything else is a baseline.
is_latent_model <- function(model) grepl("^mlp_latent", as.character(model))

# Canonical left-to-right order for the model axis in every plot:
# linear/regression -> MLP -> XGBoost -> latent MLPs. Unlisted names go last.
MODEL_ORDER <- c(
  "sparse_logistic", "l2_logistic", "linreg",          # linear / regression
  "mlp_clf_y", "mlp_clf_w_class", "mlp_reg",           # plain MLP
  "xgb_clf", "xgb_reg", "xgboost",                     # XGBoost
  "mlp_latent_y", "mlp_latent_w_class", "mlp_latent_yw" # latent MLPs
)

# Return `models` as an ordered factor using MODEL_ORDER (present names only;
# any unknown name is appended at the end so nothing is dropped).
order_models <- function(models) {
  present <- unique(as.character(models))
  lev <- c(intersect(MODEL_ORDER, present), setdiff(present, MODEL_ORDER))
  factor(as.character(models), levels = lev)
}

# output dirs

# Adjacency matrices exported by graph_export.py (for interaction /
# nestedness plots). Lives next to the other per-run outputs.
graph_matrices_dir <- function(run_id) {
  d <- run_path(run_id, "graph_matrices")
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  d
}

plots_data_dir <- function(run_id) {
  d <- run_path(run_id, "plots", "data")
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  d
}
plots_results_dir <- function(run_id) {
  d <- run_path(run_id, "plots", "results")
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  d
}

# parse run_id from CLI args, default to last available
get_run_id <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) >= 1) return(args[1])
  ds_dir <- file.path(RESULTS_DIR, DATASET_NAME)
  runs <- sort(list.dirs(ds_dir, recursive = FALSE, full.names = FALSE),
               decreasing = TRUE)
  if (length(runs) == 0) stop("No runs found in ", ds_dir)
  message("Using latest run: ", runs[1])
  runs[1]
}
