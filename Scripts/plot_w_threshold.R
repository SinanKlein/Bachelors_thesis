# plot_w_threshold.R  — AUC vs W edge-probability threshold (slide 21 style)
#
# Reads metrics/w_threshold_sweep.csv (one row per threshold x outer split) and
# plots mean test AUC vs threshold with +/-1 SD error bars across the 10 splits,
# plus a smooth trend fit. Usage: Rscript plot_w_threshold.R <run_id> <dataset>

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
  library(ggplot2); library(dplyr); library(readr); library(scales)
})

pub_theme <- theme_minimal(base_size = 12) +
  theme(
    plot.title       = element_text(face = "bold", size = 13),
    plot.subtitle    = element_text(color = "grey40", size = 10),
    plot.caption     = element_text(color = "grey55", size = 8, hjust = 0),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "grey92", linewidth = 0.3),
    axis.title       = element_text(size = 11),
    axis.text        = element_text(size = 10, color = "grey25"),
    plot.margin      = margin(10, 12, 10, 10)
  )
theme_set(pub_theme)

run_id <- get_run_id()
out <- run_path(run_id, "plots", "w_threshold")
dir.create(out, recursive = TRUE, showWarnings = FALSE)

sweep_path <- run_path(run_id, "metrics", "w_threshold_sweep.csv")
if (!file.exists(sweep_path)) {
  stop("No w_threshold_sweep.csv at: ", sweep_path,
       "\nRun experiment.py with the W-threshold block enabled.")
}
sweep <- read_csv(sweep_path, show_col_types = FALSE)

agg <- sweep %>%
  group_by(threshold) %>%
  summarise(
    auc_mean = mean(auc, na.rm = TRUE),
    auc_sd   = sd(auc, na.rm = TRUE),
    n_splits = sum(!is.na(auc)),
    .groups  = "drop"
  ) %>%
  filter(n_splits > 0)

p <- ggplot(agg, aes(x = threshold, y = auc_mean)) +
  geom_errorbar(aes(ymin = auc_mean - auc_sd, ymax = auc_mean + auc_sd),
                width = 0.012, color = "#4C78A8", linewidth = 0.4, na.rm = TRUE) +
  geom_smooth(method = "loess", se = FALSE, color = "grey55",
              linewidth = 0.5, na.rm = TRUE, formula = y ~ x) +
  geom_point(size = 2.4, color = "#4C78A8", na.rm = TRUE) +
  scale_x_continuous(limits = c(0, 1)) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title    = "AUC vs threshold over W, with SD and fit",
    subtitle = sprintf("Logistic 1[W > t] | X over %d evenly spaced cuts; bars = +/-1 SD across splits",
                       nrow(agg)),
    x = "Threshold over W", y = "AUC",
    caption  = sprintf("run = %s", run_id)
  )
ggsave(file.path(out, "w_threshold_auc.png"), p, width = 7, height = 5, dpi = 150)
write_csv(agg, file.path(out, "w_threshold_auc_summary.csv"))
message(sprintf("[plot_w_threshold] wrote %s", file.path(out, "w_threshold_auc.png")))
