# plot_stability.R  — lasso stability-selection frequencies
#
# Reads metrics/stability_selection.csv (one row per ProC feature, with its
# selection frequency across B bootstrap lasso fits) and plots the top features
# as a ranked lollipop, with the pi selection threshold marked.
# Usage: Rscript plot_stability.R <run_id> <dataset>

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
  library(ggplot2); library(dplyr); library(readr); library(jsonlite)
})

pub_theme <- theme_minimal(base_size = 12) +
  theme(
    plot.title       = element_text(face = "bold", size = 13),
    plot.subtitle    = element_text(color = "grey40", size = 10),
    plot.caption     = element_text(color = "grey55", size = 8, hjust = 0),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(color = "grey92", linewidth = 0.3),
    axis.text        = element_text(size = 9, color = "grey25"),
    legend.position  = "top",
    plot.margin      = margin(10, 12, 10, 10)
  )
theme_set(pub_theme)

`%||%` <- function(a, b) if (is.null(a)) b else a

run_id <- get_run_id()
out <- run_path(run_id, "plots", "stability")
dir.create(out, recursive = TRUE, showWarnings = FALSE)

sel_path <- run_path(run_id, "metrics", "stability_selection.csv")
if (!file.exists(sel_path)) {
  stop("No stability_selection.csv at: ", sel_path,
       "\nRun experiment.py with the stability-selection block enabled.")
}
sel <- read_csv(sel_path, show_col_types = FALSE)

log_path <- run_path(run_id, "metrics", "stability_log.json")
pi <- if (file.exists(log_path)) (jsonlite::fromJSON(log_path)$pi_threshold %||% 0.6) else 0.6

short_label <- function(x, n = 48) {
  x <- as.character(x)
  ifelse(nchar(x) > n, paste0(substr(x, 1, n - 3), "..."), x)
}

n_top <- min(30, nrow(sel))
d <- sel %>%
  arrange(desc(selection_freq)) %>%
  slice_head(n = n_top) %>%
  mutate(label = paste0(feature_source, ": ", short_label(feature_name)),
         label = factor(label, levels = rev(label)))

p <- ggplot(d, aes(x = selection_freq, y = label, color = feature_source)) +
  geom_vline(xintercept = pi, linetype = "dashed", color = "grey50", linewidth = 0.4) +
  geom_segment(aes(x = 0, xend = selection_freq, yend = label), linewidth = 0.4) +
  geom_point(size = 2.2) +
  scale_x_continuous(limits = c(0, 1)) +
  labs(
    title    = "Lasso stability selection: top features",
    subtitle = sprintf("Selection frequency across bootstrap lasso fits; dashed line = pi = %.2f", pi),
    x = "Selection frequency", y = NULL, color = "Source",
    caption  = sprintf("run = %s", run_id)
  )
ggsave(file.path(out, "stability_selection.png"), p, width = 8, height = 7, dpi = 150)
message(sprintf("[plot_stability] wrote %s (%d features shown, %d selected at pi>=%.2f)",
                file.path(out, "stability_selection.png"),
                n_top, sum(sel$selected, na.rm = TRUE), pi))
