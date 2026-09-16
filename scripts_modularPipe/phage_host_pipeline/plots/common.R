# common.R  —  paths, loaders, theme and palette shared by every plot script.
# Usage of the plot scripts:  Rscript plots/<script>.R <run_id> <DATASET_NAME>

suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
  library(dplyr)
  library(tibble)
  library(jsonlite)
})

`%||%` <- function(a, b) if (is.null(a)) b else a

# Same root and override variable as common.py.
RESULTS_DIR <- Sys.getenv("PIPELINE_RESULTS_DIR",
                          unset = "C:/Sinan_Klein/LMU/lmu_thesis/results_modularPipe")

.args <- commandArgs(trailingOnly = TRUE)
DATASET_NAME <- if (length(.args) >= 2 && nzchar(.args[2])) .args[2] else "IBD_outputs"

RESPONSE_LABELS <- c(
  y       = "CRISPR linkage",
  w_class = "glasso edge (binarised)",
  w_reg   = "glasso edge probability"
)

response_label <- function(x) {
  out <- RESPONSE_LABELS[as.character(x)]
  out[is.na(out)] <- as.character(x)[is.na(out)]
  unname(out)
}

run_path <- function(run_id, ...) file.path(RESULTS_DIR, DATASET_NAME, run_id, ...)

out_dir <- function(run_id, ...) {
  d <- run_path(run_id, ...)
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  d
}

read_run_csv <- function(run_id, ...) read_csv(run_path(run_id, ...), show_col_types = FALSE)

# First CLI arg, else the latest run of this dataset.
get_run_id <- function() {
  if (length(.args) >= 1) return(.args[1])
  runs <- sort(list.dirs(file.path(RESULTS_DIR, DATASET_NAME), recursive = FALSE,
                         full.names = FALSE), decreasing = TRUE)
  if (length(runs) == 0) stop("No runs found in ", file.path(RESULTS_DIR, DATASET_NAME))
  message("Using latest run: ", runs[1])
  runs[1]
}

# Theme ----------------------------------------------------------------------
pub_theme <- theme_minimal(base_size = 12) +
  theme(
    plot.title       = element_text(face = "bold", size = 13),
    plot.subtitle    = element_text(color = "grey40", size = 10),
    plot.caption     = element_text(color = "grey55", size = 8, hjust = 0),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "grey92", linewidth = 0.3),
    strip.background = element_rect(fill = "grey95", color = NA),
    strip.text       = element_text(face = "bold", size = 11),
    axis.title       = element_text(size = 11),
    axis.text        = element_text(size = 10, color = "grey25"),
    legend.position  = "right",
    legend.title     = element_text(size = 10, face = "bold"),
    legend.text      = element_text(size = 9),
    plot.margin      = margin(10, 12, 10, 10)
  )

# Rotated category labels, no vertical grid.
theme_categorical_x <- function() {
  theme(panel.grid.major.x = element_blank(),
        axis.text.x = element_text(size = 11, color = "grey15", angle = 20, hjust = 1))
}

# Model colours and order ------------------------------------------------------
MODEL_COLORS <- c(
  sparse_logistic = "#4C78A8", l2_logistic = "#4C78A8", linreg = "#3B6EA5",
  xgb_clf = "#F58518", xgb_reg = "#F58518", xgboost = "#F58518",
  mlp_clf_y = "#54A24B", mlp_clf_w_class = "#54A24B", mlp_reg = "#54A24B",
  mlp_latent_yw = "#9D755D"
)
.fallback_palette <- c("#72B7B2", "#EECA3B", "#BAB0AC", "#FF9DA6", "#8C564B",
                       "#17BECF", "#BCBD22", "#7F7F7F")

pal_for <- function(models) {
  models <- sort(unique(as.character(models)))
  cols <- MODEL_COLORS[models]
  missing <- which(is.na(cols))
  if (length(missing)) {
    cols[missing] <- .fallback_palette[(seq_along(missing) - 1) %% length(.fallback_palette) + 1]
  }
  setNames(as.character(cols), models)
}

is_latent_model <- function(model) grepl("^mlp_latent", as.character(model))

MODEL_ORDER <- c("sparse_logistic", "l2_logistic", "linreg",
                 "mlp_clf_y", "mlp_clf_w_class", "mlp_reg",
                 "xgb_clf", "xgb_reg", "xgboost",
                 "mlp_latent_yw")

order_models <- function(models) {
  present <- unique(as.character(models))
  factor(as.character(models),
         levels = c(intersect(MODEL_ORDER, present), setdiff(present, MODEL_ORDER)))
}
