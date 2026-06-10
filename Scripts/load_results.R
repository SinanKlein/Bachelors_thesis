# load_results.R

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tibble)
  library(jsonlite)
})

# user should fill
RESULTS_DIR <- "C:/Sinan_Klein/LMU/lmu_thesis/results_modularPipe"   # same path as Python's RESULTS_DIR

# load helpers
run_path <- function(run_id, ...) {
  file.path(RESULTS_DIR, run_id, ...)
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

# output dirs
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
  runs <- sort(list.dirs(RESULTS_DIR, recursive = FALSE, full.names = FALSE),
               decreasing = TRUE)
  if (length(runs) == 0) stop("No runs found in ", RESULTS_DIR)
  message("Using latest run: ", runs[1])
  runs[1]
}
