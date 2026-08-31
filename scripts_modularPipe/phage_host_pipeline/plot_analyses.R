# plot_analyses.R  —  figures for the ablation and the supplementary analyses
# Two independent sections, one per analysis, previously separate scripts:
#
#   1. ablation      ablation.py   -> plots/ablation/
#                    arm_ladder.png       AUC by arm x model, one facet per task,
#                                         anchor shaded.
#                                         Flat bars = the representation is not
#                                         the binding constraint.
#                    family_vs_genus.png  genus vs family AUC, one facet per
#                                         task.
#                    Also prints the ANCHOR CHECK to the console.
#
#   2. w_threshold   analyses.py   -> plots/w_threshold/
#                    w_threshold_auc.png       test AUC against the W cut, i.e.
#                                              whether W becomes learnable at ANY
#                                              threshold rather than just the one
#                                              the w_class task uses.
#
# Each section is skipped with a message if its input CSV is absent, so a partial
# run still produces every figure it can. Colors, model ordering and the theme
# come from load_results.R, so these match every other figure in the thesis.
#
# Usage:  Rscript plot_analyses.R <run_id> <DATASET_NAME>

source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE),
       value = TRUE)[1])), "load_results.R"))

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(jsonlite)
  library(readr)
  library(scales)
})

`%||%` <- function(a, b) if (is.null(a)) b else a

run_id <- get_run_id()
message(sprintf("[plot_analyses] dataset=%s run=%s", DATASET_NAME, run_id))

# 1. Representation ablation + family / genus
# Reads <run>/ablation/*.csv written by ablation.py.
# Wrapped in local() so this section's variables (out, p, ...) cannot collide
# with the others: each block previously lived in its own script.
local({
  out <- run_path(run_id, "plots", "ablation")
  dir.create(out, recursive = TRUE, showWarnings = FALSE)

  # Theme, model palette and level colors all come from load_results.R.
  # Arms on x; the fill legend is doing real work here, so it stays.
  theme_set(pub_theme + theme_categorical_x())

  # Arms are named "1_baseline_clr_concat_quantile" etc. The leading digit fixes
  # the ladder order; strip it for the axis label so the panels stay readable.
  arm_label <- function(a) gsub("^[0-9]+_", "", as.character(a))

  # 1. Arm ladder
  met_path <- run_path(run_id, "ablation", "ablation_metrics.csv")
  if (!file.exists(met_path)) {
    stop("No ablation_metrics.csv at: ", met_path,
         "\nRun ablation.py for this run_id first.")
  }
  met <- read_csv(met_path, show_col_types = FALSE)

  summ <- met %>%
    filter(role == "test", !is.na(auc)) %>%
    group_by(task, arm, model) %>%
    summarise(auc_mean = mean(auc), auc_sd = sd(auc), n_folds = dplyr::n(),
              .groups = "drop") %>%
    mutate(arm_lab = factor(arm_label(arm), levels = arm_label(sort(unique(arm)))),
           model   = order_models(model))

  # anchor, for the shaded band
  anchor <- NA_character_
  log_path <- run_path(run_id, "ablation", "ablation_log.json")
  if (file.exists(log_path)) {
    lg <- jsonlite::fromJSON(log_path)
    if (!is.null(lg$anchor)) anchor <- arm_label(lg$anchor)
  }

  p <- ggplot(summ, aes(x = arm_lab, y = auc_mean, fill = model))
  if (!is.na(anchor) && anchor %in% levels(summ$arm_lab)) {
    p <- p + annotate("rect",
                      xmin = which(levels(summ$arm_lab) == anchor) - 0.5,
                      xmax = which(levels(summ$arm_lab) == anchor) + 0.5,
                      ymin = -Inf, ymax = Inf, fill = "grey90", alpha = 0.55)
  }
  p <- p +
    geom_col(position = position_dodge(width = 0.8), width = 0.72) +
    geom_errorbar(aes(ymin = auc_mean - auc_sd, ymax = auc_mean + auc_sd),
                  position = position_dodge(width = 0.8), width = 0.18,
                  linewidth = 0.35, color = "grey25") +
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey55", linewidth = 0.35) +
    facet_wrap(~ task, nrow = 1) +
    scale_fill_manual(values = pal_for(summ$model), name = NULL) +
    coord_cartesian(ylim = c(0.4, 1.0)) +
    labs(title = sprintf("%s - representation ablation", DATASET_NAME),
         x = NULL, y = "test AUC (mean +/- SD over folds)") +
    theme(legend.position = "right")

  ggsave(file.path(out, "arm_ladder.png"), p,
         width = 5.2 * length(unique(summ$task)), height = 4.6, dpi = 150)
  message(sprintf("[plot_analyses:ablation] wrote %s", file.path(out, "arm_ladder.png")))

  # 2. Anchor check — the pre-committed switching rule, printed not plotted
  if (!is.na(anchor)) {
    message("\n=== ANCHOR CHECK (switch only if a gap exceeds one fold SD) ===")
    for (tk in unique(summ$task)) {
      for (md in unique(summ$model[summ$task == tk])) {
        s <- summ %>% filter(task == tk, model == md)
        a <- s %>% filter(as.character(arm_lab) == anchor)
        if (!nrow(a) || nrow(s) < 2) next
        best <- s[which.max(s$auc_mean), ]
        gap <- best$auc_mean - a$auc_mean
        sd0 <- ifelse(is.na(a$auc_sd), 0, a$auc_sd)
        message(sprintf("  %-8s %-18s anchor=%.4f (sd %.4f) | best=%s %.4f | gap=%+.4f -> %s",
                        tk, md, a$auc_mean, sd0, best$arm_lab, best$auc_mean, gap,
                        ifelse(gap > sd0, "SWITCH", "keep anchor")))
      }
    }
  }

  # 3. Family vs genus.
  fam_path <- run_path(run_id, "ablation", "family_metrics.csv")
  if (!file.exists(fam_path)) {
    message("[plot_analyses:ablation] no family_metrics.csv - skipping the family figure.")
  } else {
    fam <- read_csv(fam_path, show_col_types = FALSE)

    # only the anchor arm: the family question is about taxonomy, not representation
    arm_keep <- if (!is.na(anchor)) {
      unique(fam$arm)[arm_label(unique(fam$arm)) == anchor]
    } else unique(fam$arm)[1]
    fam <- fam %>% filter(arm %in% arm_keep)

    bars <- fam %>%
      filter(level %in% c("genus", "family")) %>%
      mutate(level = factor(level, levels = c("genus", "family")),
             model = order_models(model))
    pf <- ggplot(bars, aes(x = model, y = auc, fill = level)) +
      geom_col(position = position_dodge(width = 0.8), width = 0.72) +
      geom_errorbar(aes(ymin = auc - auc_sd, ymax = auc + auc_sd),
                    position = position_dodge(width = 0.8), width = 0.18,
                    linewidth = 0.35, color = "grey25")

    pf <- pf +
      geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey55", linewidth = 0.35) +
      # free_x: each task only trains a subset of models (e.g. mlp_clf_y is
      # y-only, mlp_clf_w_class is w_class-only); a shared/fixed x scale would
      # reserve an empty slot for the other task's model, leaving a visible gap.
      facet_wrap(~ task, nrow = 1, scales = "free_x") +
      scale_fill_manual(values = LEVEL_COLORS, name = NULL) +
      coord_cartesian(ylim = c(0.4, 1.0)) +
      labs(title = sprintf("%s - family vs genus aggregation", DATASET_NAME),
           x = NULL, y = "AUC (mean +/- SD over folds)") +
      theme(legend.position = "right")

    ggsave(file.path(out, "family_vs_genus.png"), pf,
           width = 5.2 * length(unique(bars$task)), height = 4.6, dpi = 150)
    message(sprintf("[plot_analyses:ablation] wrote %s", file.path(out, "family_vs_genus.png")))
  }

  message("[plot_analyses:ablation] done.")
})

# 2. AUC against the W threshold
# Reads metrics/w_threshold_sweep.csv written by analyses.py.
local({
  # Theme: the shared base from load_results.R (no overrides needed here).
  theme_set(pub_theme)

  out <- run_path(run_id, "plots", "w_threshold")
  dir.create(out, recursive = TRUE, showWarnings = FALSE)

  sweep_path <- run_path(run_id, "metrics", "w_threshold_sweep.csv")
  if (!file.exists(sweep_path)) {
    stop("No w_threshold_sweep.csv at: ", sweep_path,
         "\nRun experiment.py with the W-threshold block enabled.")
  }
  sweep <- read_csv(sweep_path, show_col_types = FALSE)

  # Older sweeps had no model column (logistic only); default it so the plot works.
  if (!"model" %in% names(sweep)) sweep$model <- "l2_logistic"

  agg <- sweep %>%
    group_by(model, threshold) %>%
    summarise(
      auc_mean = mean(auc, na.rm = TRUE),
      auc_sd   = sd(auc, na.rm = TRUE),
      n_splits = sum(!is.na(auc)),
      .groups  = "drop"
    ) %>%
    filter(n_splits > 0)

  pal <- pal_for(agg$model)

  # Title carries the peak AUC for each model so the number is visible alongside
  # the curve; per-point values are also written next to each marker.
  peaks <- agg %>% group_by(model) %>% slice_max(auc_mean, n = 1, with_ties = FALSE) %>% ungroup()
  peak_txt <- paste(sprintf("%s peak AUC=%.3f @ t=%.2f",
                            peaks$model, peaks$auc_mean, peaks$threshold),
                    collapse = "   |   ")

  p <- ggplot(agg, aes(x = threshold, y = auc_mean, color = model, fill = model)) +
    geom_errorbar(aes(ymin = auc_mean - auc_sd, ymax = auc_mean + auc_sd),
                  width = 0.012, linewidth = 0.4, na.rm = TRUE) +
    geom_line(linewidth = 0.5, na.rm = TRUE) +
    geom_point(size = 2.4, na.rm = TRUE) +
    geom_text(aes(label = sprintf("%.2f", auc_mean)), vjust = -0.9,
              size = 2.6, show.legend = FALSE, na.rm = TRUE) +
    scale_color_manual(values = pal) +
    scale_fill_manual(values = pal) +
    scale_x_continuous(limits = c(0, 1)) +
    scale_y_continuous(limits = c(0, 1)) +
    labs(title = "AUC vs threshold W",
         x = "Threshold over W", y = "AUC", color = "model", fill = "model")
  message(sprintf("[plot_analyses:w_threshold] %s", peak_txt))
  ggsave(file.path(out, "w_threshold_auc.png"), p, width = 8, height = 5, dpi = 150)
  write_csv(agg, file.path(out, "w_threshold_auc_summary.csv"))
  message(sprintf("[plot_analyses:w_threshold] wrote %s", file.path(out, "w_threshold_auc.png")))
})

# 3. Shuffled-Y control  —  shuffle_probe.py -> plots/shuffle_probe/
#    real_vs_permuted.png   per-fold test AUC with real vs permuted Y training
#                           labels, one facet per head. W labels and every
#                           evaluation label stay real.
local({
  sp_path  <- run_path(run_id, "shuffle_probe", "shuffle_probe_metrics.csv")
  sum_path <- run_path(run_id, "shuffle_probe", "shuffle_probe_summary.csv")
  if (!file.exists(sp_path)) {
    message("[plot_analyses:shuffle] no shuffle_probe_metrics.csv; section skipped.")
    return(invisible(NULL))
  }
  out <- run_path(run_id, "plots", "shuffle_probe")
  dir.create(out, recursive = TRUE, showWarnings = FALSE)

  head_lab <- c(y = "Y  (CRISPR)", w_class = "W  (glasso edge)")

  # One value per fold per condition: the shuffles are averaged within a fold,
  # so real and permuted are paired 1:1 across the outer folds.
  fold <- read_csv(sp_path, show_col_types = FALSE) %>%
    mutate(condition = ifelse(condition == "real", "real", "permuted")) %>%
    group_by(head, outer_fold, condition) %>%
    summarise(auc = mean(auc, na.rm = TRUE), .groups = "drop") %>%
    mutate(condition = factor(condition, levels = c("real", "permuted")),
           head      = factor(head, levels = names(head_lab), labels = head_lab))

  ann <- NULL
  if (file.exists(sum_path)) {
    ann <- read_csv(sum_path, show_col_types = FALSE) %>%
      transmute(head = factor(head, levels = names(head_lab), labels = head_lab),
                lab  = sprintf("diff = %+.3f    p = %s", auc_diff,
                               ifelse(p_paired < 1e-3,
                                      formatC(p_paired, format = "e", digits = 1),
                                      sprintf("%.3f", p_paired))))
  }

  p <- ggplot(fold, aes(x = condition, y = auc)) +
    geom_hline(yintercept = 0.5, linetype = "dashed", colour = "grey60") +
    geom_line(aes(group = outer_fold), colour = "grey75", linewidth = 0.4) +
    geom_point(aes(colour = condition), size = 2.4) +
    stat_summary(fun = mean, geom = "crossbar", width = 0.45,
                 linewidth = 0.5, colour = "black") +
    scale_colour_manual(values = c(real = "#1B7F79", permuted = "#C24E4E"),
                        guide = "none") +
    scale_y_continuous(limits = c(0, 1)) +
    facet_wrap(~ head) +
    labs(title = sprintf("%s - real vs permuted Y", DATASET_NAME),
         x = NULL, y = "AUC")

  if (!is.null(ann)) {
    p <- p + geom_text(data = ann, aes(x = 1.5, y = 0.06, label = lab),
                       inherit.aes = FALSE, size = 3.2)
  }

  ggsave(file.path(out, "real_vs_permuted.png"), p, width = 8, height = 5, dpi = 150)
  message(sprintf("[plot_analyses:shuffle] wrote %s",
                  file.path(out, "real_vs_permuted.png")))
})

message("[plot_analyses] done.")
