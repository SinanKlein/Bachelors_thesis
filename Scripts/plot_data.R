# =============================================================================
# plot_data.R  —  descriptive plots of preprocessing outputs
# =============================================================================
# Reads stage 1 outputs and produces publication-quality descriptive plots.
# Usage:
#   Rscript plot_data.R                # latest run
#   Rscript plot_data.R <run_id>

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
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(scales)
  library(patchwork)   # for marginal-hist composites
})

`%||%` <- function(a, b) if (is.null(a)) b else a

run_id <- get_run_id()
out    <- plots_data_dir(run_id)
pre    <- load_preprocessing(run_id)

# Publication theme (applied to every plot)

pub_theme <- theme_minimal(base_size = 12, base_family = "") +
  theme(
    plot.title       = element_text(face = "bold", size = 13),
    plot.subtitle    = element_text(color = "grey40", size = 10),
    plot.caption     = element_text(color = "grey55", size = 8, hjust = 0),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "grey92", linewidth = 0.3),
    strip.background = element_rect(fill = "grey95", color = NA),
    strip.text       = element_text(face = "bold", size = 11),
    axis.title       = element_text(size = 11),
    axis.text        = element_text(size = 9, color = "grey25"),
    legend.position  = "right",
    legend.title     = element_text(size = 10, face = "bold"),
    legend.text      = element_text(size = 9),
    plot.margin      = margin(10, 12, 10, 10)
  )
theme_set(pub_theme)

# colors  (colorblind friendly, dark accent for "selected/kept")
col_kept    <- "#D62728"   # red
col_dropped <- "#BDBDBD"   # grey
col_source  <- c(Xb = "#1F77B4", Xv = "#FF7F0E")   # blue / orange


# Helper: per source mean var scatter with marginal histograms

plot_mean_var_single <- function(df_one, source_name, accent_color) {
  # df_one: feature_stats rows for one source

  # main scatter — log10(1 + x) on both axes handles zeros, taming the long tail
  p_main <- ggplot(df_one, aes(x = mean, y = var, color = selected)) +
    geom_point(alpha = 0.45, size = 1.1, stroke = 0) +
    scale_color_manual(
      values = c(`FALSE` = col_dropped, `TRUE` = accent_color),
      labels = c(`FALSE` = "dropped", `TRUE` = "kept"),
      name   = "filter"
    ) +
    scale_x_continuous(
      trans  = scales::pseudo_log_trans(base = 10),
      breaks = c(0, 0.1, 1, 10, 100),
      labels = c("0", "0.1", "1", "10", "100")
    ) +
    scale_y_continuous(
      trans  = scales::pseudo_log_trans(base = 10),
      breaks = c(0, 0.01, 1, 100, 10000),
      labels = c("0", "0.01", "1", "100", "10k")
    ) +
    labs(x = "mean (log)", y = "variance (log)") +
    theme(legend.position = "bottom")

  # marginal hist of mean
  p_top <- ggplot(df_one, aes(x = mean)) +
    geom_histogram(bins = 50, fill = accent_color, alpha = 0.7, color = NA) +
    scale_x_continuous(trans = scales::pseudo_log_trans(base = 10)) +
    theme_void() +
    theme(plot.margin = margin(0, 0, 0, 0))

  # marginal hist of variance
  p_right <- ggplot(df_one, aes(x = var)) +
    geom_histogram(bins = 50, fill = accent_color, alpha = 0.7, color = NA) +
    scale_x_continuous(trans = scales::pseudo_log_trans(base = 10)) +
    coord_flip() +
    theme_void() +
    theme(plot.margin = margin(0, 0, 0, 0))

  n_total <- nrow(df_one)
  n_kept  <- sum(df_one$selected)
  thr     <- unique(df_one$threshold)[1]

  composite <- p_top + plot_spacer() +
               p_main + p_right +
    plot_layout(
      ncol = 2, nrow = 2,
      widths  = c(4, 1),
      heights = c(1, 4)
    ) +
    plot_annotation(
      title    = sprintf("Feature mean vs variance — %s", source_name),
      subtitle = sprintf("%s features total | %s kept (top %.0f%% by log-variance) | threshold = %.3g",
                         format(n_total, big.mark = ","),
                         format(n_kept,  big.mark = ","),
                         100 * (n_kept / n_total),
                         thr),
      caption  = sprintf("run = %s   axes: pseudo-log10 (handles zeros)", run_id),
      theme    = pub_theme
    )
  composite
}

# 1. & 2. Per-source mean variance plots  (Xb only, Xv only)

fs <- pre$feature_stats
fs$selected <- as.logical(fs$selected)

sources <- unique(fs$source)
for (s in sources) {
  p <- plot_mean_var_single(fs %>% filter(source == s),
                            source_name = s,
                            accent_color = col_source[[s]] %||% col_kept)
  ggsave(file.path(out, sprintf("0%d_mean_variance_%s.pdf",
                                which(sources == s), s)),
         p, width = 7, height = 6.5)
}

# 3. Combined mean variance (Xb + Xv side by side, free scales)

p3 <- ggplot(fs, aes(x = mean, y = var, color = selected)) +
  geom_point(alpha = 0.45, size = 0.9, stroke = 0) +
  scale_color_manual(
    values = c(`FALSE` = col_dropped, `TRUE` = col_kept),
    labels = c(`FALSE` = "dropped", `TRUE` = "kept"),
    name   = "filter"
  ) +
  scale_x_continuous(
    trans  = scales::pseudo_log_trans(base = 10),
    breaks = c(0, 0.1, 1, 10, 100)
  ) +
  scale_y_continuous(
    trans  = scales::pseudo_log_trans(base = 10),
    breaks = c(0, 0.01, 1, 100, 10000),
    labels = c("0", "0.01", "1", "100", "10k")
  ) +
  facet_wrap(~ source, scales = "free") +
  labs(
    title    = "Feature mean vs variance — Xb & Xv",
    subtitle = "Pseudo-log axes. Color = top-quantile selection by log-variance score.",
    x = "mean (log)", y = "variance (log)",
    caption = sprintf("run = %s", run_id)
  )
ggsave(file.path(out, "03_mean_variance_combined.pdf"),
       p3, width = 9, height = 4.5)

# 4. Score distribution

p4 <- ggplot(fs, aes(x = score, fill = selected)) +
  geom_histogram(bins = 60, alpha = 0.9, position = "identity", color = NA) +
  geom_vline(aes(xintercept = threshold),
             linetype = "dashed", color = "grey20", linewidth = 0.5) +
  facet_wrap(~ source, scales = "free") +
  scale_fill_manual(
    values = c(`FALSE` = col_dropped, `TRUE` = col_kept),
    labels = c(`FALSE` = "dropped", `TRUE` = "kept"),
    name   = "filter"
  ) +
  labs(
    title    = "Log-variance score distribution",
    subtitle = "Dashed line = top-quantile cutoff",
    x = "log-variance score", y = "feature count",
    caption  = sprintf("run = %s", run_id)
  )
ggsave(file.path(out, "04_score_distribution.pdf"), p4,
       width = 9, height = 4)


# 5. Feature counts 

fk <- fs %>%
  group_by(source) %>%
  summarise(total = n(), kept = sum(selected), .groups = "drop") %>%
  mutate(pct = sprintf("%.0f%% kept", 100 * kept / total)) %>%
  pivot_longer(c(total, kept), names_to = "type", values_to = "n_features") %>%
  mutate(type = factor(type, levels = c("total", "kept")))

ann <- fk %>% filter(type == "kept")

p5 <- ggplot(fk, aes(x = source, y = n_features, fill = type)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6) +
  geom_text(data = ann, aes(label = pct),
            position = position_dodge(width = 0.7),
            vjust = -0.5, size = 3.5, color = "grey25") +
  scale_fill_manual(
    values = c(total = "grey75", kept = col_kept),
    name   = NULL
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12)),
                     labels = scales::label_comma()) +
  labs(
    title    = "Features before vs after filter",
    subtitle = "Variance-based selection keeps the top quantile per source",
    x = NULL, y = "feature count",
    caption  = sprintf("run = %s", run_id)
  )
ggsave(file.path(out, "05_feature_counts.pdf"), p5,
       width = 6, height = 4)

# 6. Feature sparsity — fraction of zeros per feature  (NEW)

p6 <- ggplot(fs, aes(x = source, y = mean, fill = source)) +
  geom_violin(alpha = 0.6, color = NA, scale = "width") +
  geom_boxplot(width = 0.12, alpha = 0.9, outlier.size = 0.4,
               outlier.alpha = 0.3, color = "grey20") +
  scale_fill_manual(values = col_source, guide = "none") +
  scale_y_continuous(trans = scales::pseudo_log_trans(base = 10),
                     breaks = c(0, 0.01, 0.1, 1, 10, 100)) +
  labs(
    title    = "Per-feature mean activity by source",
    subtitle = "Distribution of feature means — proxy for how sparse / dense each source is",
    x = NULL, y = "feature mean (pseudo-log)",
    caption  = sprintf("run = %s", run_id)
  )
ggsave(file.path(out, "06_feature_activity.pdf"), p6,
       width = 6, height = 4)

# 7. Label balance per task
# Binary tasks: show n_pos / n_neg as proportional stacked bars
# Regression tasks: show n observed only (no positive/negative)
lb_bin <- pre$labels %>%
  filter(task_type == "binary", mask == 1) %>%
  group_by(task) %>%
  summarise(n_pos = sum(y == 1), n_neg = sum(y == 0),
            n_total = n(),
            pos_frac = mean(y == 1),
            .groups = "drop")

lb_long <- lb_bin %>%
  pivot_longer(c(n_pos, n_neg), names_to = "class", values_to = "n") %>%
  mutate(class = factor(class, levels = c("n_neg", "n_pos"),
                        labels = c("y = 0", "y = 1")))

p7 <- ggplot(lb_long, aes(x = task, y = n, fill = class)) +
  geom_col(width = 0.6) +
  geom_text(data = lb_bin,
            aes(x = task, y = n_total,
                label = sprintf("n = %s\n%.1f%% pos",
                                format(n_total, big.mark = ","),
                                100 * pos_frac)),
            inherit.aes = FALSE,
            vjust = -0.4, size = 3.2, color = "grey25") +
  scale_fill_manual(
    values = c("y = 0" = "grey75", "y = 1" = col_kept),
    name   = "class"
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18)),
                     labels = scales::label_comma()) +
  labs(
    title    = "Label balance per binary task",
    subtitle = "Counts after masking; positive class is the minority — class imbalance is real.",
    x = NULL, y = "count (observed rows)",
    caption  = sprintf("run = %s", run_id)
  )
ggsave(file.path(out, "07_label_balance.pdf"), p7,
       width = 6.5, height = 4.5)

# 8. W distribution 

reg_labels <- pre$labels %>% filter(task_type == "regression", mask == 1)
if (nrow(reg_labels) > 0) {
  reg_task_name <- unique(reg_labels$task)[1]
  p8 <- ggplot(reg_labels, aes(x = y)) +
    geom_histogram(bins = 50, fill = col_source[["Xb"]],
                   alpha = 0.8, color = NA) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey30") +
    labs(
      title    = sprintf("Distribution of %s (observed rows)", reg_task_name),
      subtitle = sprintf("n = %s observed | mean = %.3f | sd = %.3f",
                         format(nrow(reg_labels), big.mark = ","),
                         mean(reg_labels$y),
                         sd(reg_labels$y)),
      x = "W value", y = "count",
      caption = sprintf("run = %s", run_id)
    )
  ggsave(file.path(out, "08_W_distribution.pdf"), p8,
         width = 6.5, height = 4)
}

# 9. Stratification sanity — positive fraction per outer fold

prim_task <- pre$summary$tasks$name[1]
y_prim <- pre$labels %>% filter(task == prim_task) %>%
  select(sample_id, y)

bal <- pre$splits %>%
  filter(inner_fold == -1) %>%
  inner_join(y_prim, by = "sample_id") %>%
  group_by(outer_fold, role) %>%
  summarise(pos_frac = mean(y == 1), n = n(), .groups = "drop")

# baseline positive fraction overall
baseline <- mean(y_prim$y == 1)

p9 <- ggplot(bal, aes(x = factor(outer_fold), y = pos_frac, fill = role)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6) +
  geom_hline(yintercept = baseline, linetype = "dashed",
             color = "grey30", linewidth = 0.4) +
  annotate("text", x = 0.7, y = baseline,
           label = sprintf("overall = %.2f%%", 100 * baseline),
           vjust = -0.5, hjust = 0, size = 3, color = "grey30") +
  scale_fill_manual(values = c(train = "grey60", test = col_kept),
                    name = "role") +
  scale_y_continuous(labels = scales::label_percent(accuracy = 0.1),
                     expand = expansion(mult = c(0, 0.15))) +
  labs(
    title    = sprintf("Stratification check — task = %s", prim_task),
    subtitle = "Positive fraction per outer fold. Bars should match the overall rate.",
    x = "outer fold", y = "P(y = 1)",
    caption  = sprintf("run = %s", run_id)
  )
ggsave(file.path(out, "09_split_balance.pdf"), p9,
       width = 8, height = 4)

cat(sprintf("[plot_data] wrote %d plots to %s\n",
            length(list.files(out, pattern = "\\.pdf$")), out))
