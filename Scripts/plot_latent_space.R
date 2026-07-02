# plot_latent_space.R
# PCA and feature-interpretation plots for exported latent spaces.
# Usage:
#   Rscript plot_latent_space.R <run_id> <dataset_name>

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  match <- grep(file_arg, args)
  if (length(match)) return(normalizePath(dirname(sub(file_arg, "", args[match]))))
  if (sys.nframe() > 0) {
    f <- try(sys.frame(1)$ofile, silent = TRUE)
    if (!inherits(f, "try-error") && !is.null(f)) return(normalizePath(dirname(f)))
  }
  return(normalizePath("."))
}
source(file.path(get_script_dir(), "load_results.R"))

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(scales)
})

`%||%` <- function(a, b) if (is.null(a)) b else a

args_trailing <- commandArgs(trailingOnly = TRUE)
run_id <- get_run_id()
# Optional third command-line argument: latent model subfolder, e.g. twotower_y_only.
latent_model_arg <- if (length(args_trailing) >= 3) args_trailing[[3]] else NA_character_
base_latent_dir <- run_path(run_id, "latent")
if (!is.na(latent_model_arg) && nzchar(latent_model_arg)) {
  latent_dir <- file.path(base_latent_dir, latent_model_arg)
  out <- run_path(run_id, "plots", "latent", latent_model_arg)
} else {
  latent_dir <- base_latent_dir
  out <- run_path(run_id, "plots", "latent")
}
dir.create(out, recursive = TRUE, showWarnings = FALSE)

latent_path <- file.path(latent_dir, "latent_space.csv.gz")
log_path <- file.path(latent_dir, "latent_export_log.json")

if (!file.exists(latent_path)) {
  stop("No latent_space.csv.gz found at: ", latent_path,
       "\nRun: python export_latent_space.py --config default.yaml --run-id ", run_id)
}

latent <- read_csv(latent_path, show_col_types = FALSE)
if (file.exists(log_path)) {
  log <- jsonlite::fromJSON(log_path)
  pca_txt <- paste0("PC explained variance: ",
                    paste(sprintf("%.1f%%", 100 * unlist(log$pca_explained_variance_ratio)), collapse = ", "))
} else {
  pca_txt <- ""
}

pub_theme <- theme_minimal(base_size = 12) +
  theme(
    plot.title       = element_text(face = "bold", size = 13),
    plot.subtitle    = element_text(color = "grey40", size = 10),
    plot.caption     = element_text(color = "grey55", size = 8, hjust = 0),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "grey92", linewidth = 0.3),
    strip.background = element_rect(fill = "grey95", color = NA),
    strip.text       = element_text(face = "bold", size = 10),
    legend.position  = "right",
    legend.title     = element_text(size = 10, face = "bold"),
    legend.text      = element_text(size = 9),
    plot.margin      = margin(10, 12, 10, 10)
  )
theme_set(pub_theme)

# -----------------------------------------------------------------------------
# 01-04. Latent PCA geometry
# -----------------------------------------------------------------------------
if (!all(c("PC1", "PC2") %in% names(latent))) {
  stop("latent_space.csv.gz must contain PC1 and PC2 columns.")
}

p1 <- ggplot(latent, aes(x = PC1, y = PC2, color = factor(y))) +
  geom_point(alpha = 0.35, size = 0.7, stroke = 0) +
  scale_color_manual(values = c("0" = "grey70", "1" = "#D62728"), name = "Y") +
  labs(
    title = "Latent interaction space colored by CRISPR label",
    subtitle = pca_txt,
    x = "PC1 of latent space", y = "PC2 of latent space",
    caption = sprintf("run = %s", run_id)
  )
ggsave(file.path(out, "latent_01_pca_by_Y.png"), p1, width = 7, height = 5)

if ("w" %in% names(latent)) {
  p2 <- ggplot(latent, aes(x = PC1, y = PC2, color = factor(w))) +
    geom_point(alpha = 0.35, size = 0.7, stroke = 0) +
    scale_color_manual(values = c("0" = "grey70", "1" = "#2166AC"), name = "W class") +
    labs(
      title = "Latent interaction space colored by W class",
      subtitle = paste0("W class = 1[W > 0.5]. ", pca_txt),
      x = "PC1 of latent space", y = "PC2 of latent space",
      caption = sprintf("run = %s", run_id)
    )
  ggsave(file.path(out, "latent_02_pca_by_W.png"), p2, width = 7, height = 5)
}

if ("w_pred" %in% names(latent)) {
  p3 <- ggplot(latent, aes(x = PC1, y = PC2, color = w_pred)) +
    geom_point(alpha = 0.35, size = 0.7, stroke = 0) +
    scale_color_viridis_c(name = "P(W=1)", option = "C", labels = label_number()) +
    labs(
      title = "Latent interaction space colored by predicted W probability",
      subtitle = pca_txt,
      x = "PC1 of latent space", y = "PC2 of latent space",
      caption = sprintf("run = %s", run_id)
    )
  ggsave(file.path(out, "latent_03_pca_by_W_pred.png"), p3, width = 7, height = 5)
}

if ("agreement_class" %in% names(latent)) {
  latent$agreement_class <- factor(
    latent$agreement_class,
    levels = c("Y=1 & W=1", "Y=1 only", "W=1 only", "background")
  )
  p4 <- ggplot(latent, aes(x = PC1, y = PC2, color = agreement_class)) +
    geom_point(alpha = 0.45, size = 0.75, stroke = 0) +
    labs(
      title = "Latent interaction regimes",
      subtitle = "CRISPR (Y) vs glasso edge class (W = 1[W > 0.5]) agreement",
      x = "PC1 of latent space", y = "PC2 of latent space",
      color = "Regime",
      caption = sprintf("run = %s", run_id)
    )
  ggsave(file.path(out, "latent_04_pca_by_agreement_class.png"), p4, width = 8, height = 5)
}

message(sprintf("[plot_latent_space] wrote latent-space plots to %s", out))
