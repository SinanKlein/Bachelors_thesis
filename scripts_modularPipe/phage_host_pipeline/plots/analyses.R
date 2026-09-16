# analyses.R  —  figures for the analyses; a section is skipped when its input is missing.
#   1. w_threshold  metrics/w_threshold_sweep.csv            -> plots/w_threshold/
#   2. shuffle      shuffle_probe/shuffle_probe_*.csv          -> plots/shuffle_probe/
#   3. stability    stability/stability_selection.csv          -> plots/stability/
#   4. lambda_cv    metrics/lambda_cv.csv, lambda_cv_curve.csv -> plots/lambda_cv/
source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE),
       value = TRUE)[1])), "common.R"))

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(jsonlite)
  library(readr)
  library(scales)
})

run_id <- get_run_id()
message(sprintf("[plot_analyses] dataset=%s run=%s", DATASET_NAME, run_id))

# 1. AUC against the W threshold
local({
  sweep_path <- run_path(run_id, "metrics", "w_threshold_sweep.csv")
  if (!file.exists(sweep_path)) {
    message("[plot_analyses:w_threshold] no w_threshold_sweep.csv; skipped.")
    return(invisible(NULL))
  }
  theme_set(pub_theme)
  out <- out_dir(run_id, "plots", "w_threshold")
  sweep <- read_csv(sweep_path, show_col_types = FALSE)

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

  # Peak AUC per model, printed to the console.
  peaks <- agg %>% group_by(model) %>% slice_max(auc_mean, n = 1, with_ties = FALSE) %>% ungroup()
  peak_txt <- paste(sprintf("%s peak AUC=%.3f @ t=%.2f",
                            peaks$model, peaks$auc_mean, peaks$threshold),
                    collapse = "   |   ")

  p <- ggplot(agg, aes(x = threshold, y = auc_mean, color = model, fill = model)) +
    geom_errorbar(aes(ymin = auc_mean - auc_sd, ymax = auc_mean + auc_sd),
                  width = 0.012, linewidth = 0.4, na.rm = TRUE) +
    geom_line(linewidth = 0.5, na.rm = TRUE) +
    geom_point(size = 2.4, na.rm = TRUE) +
    scale_color_manual(values = pal) +
    scale_fill_manual(values = pal) +
    scale_x_continuous(limits = c(0, 1)) +
    scale_y_continuous(limits = c(0, 1)) +
    labs(x = "binarisation threshold on the glasso edge probability", y = "AUC",
         color = "model", fill = "model")
  message(sprintf("[plot_analyses:w_threshold] %s", peak_txt))
  ggsave(file.path(out, "w_threshold_auc.png"), p, width = 8, height = 5, dpi = 150)
  write_csv(agg, file.path(out, "w_threshold_auc_summary.csv"))
  message(sprintf("[plot_analyses:w_threshold] wrote %s", file.path(out, "w_threshold_auc.png")))
})

# 2. Real vs permuted Y, per-fold AUC per head
local({
  sp_path  <- run_path(run_id, "shuffle_probe", "shuffle_probe_metrics.csv")
  sum_path <- run_path(run_id, "shuffle_probe", "shuffle_probe_summary.csv")
  if (!file.exists(sp_path)) {
    message("[plot_analyses:shuffle] no shuffle_probe_metrics.csv; section skipped.")
    return(invisible(NULL))
  }
  out <- out_dir(run_id, "plots", "shuffle_probe")

  head_lab <- c(y = "CRISPR linkage", w_class = "glasso edge (binarised)")

  # Shuffles averaged within a fold, so real and permuted pair up per fold.
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
    labs(x = NULL, y = "AUC")

  if (!is.null(ann)) {
    p <- p + geom_text(data = ann, aes(x = 1.5, y = 0.06, label = lab),
                       inherit.aes = FALSE, size = 3.2)
  }

  ggsave(file.path(out, "real_vs_permuted.png"), p, width = 8, height = 5, dpi = 150)
  message(sprintf("[plot_analyses:shuffle] wrote %s",
                  file.path(out, "real_vs_permuted.png")))
})

# 3. Stability selection
local({
  sel_path <- run_path(run_id, "stability", "stability_selection.csv")
  if (!file.exists(sel_path)) {
    message("[plot_analyses:stability] no stability_selection.csv - skipping.")
    return(invisible(NULL))
  }
  out <- out_dir(run_id, "plots", "stability")
  theme_set(pub_theme)

  sel <- read_csv(sel_path, show_col_types = FALSE)
  tau <- 0.8
  log_path <- run_path(run_id, "stability", "stability_log.json")
  if (file.exists(log_path)) {
    lg <- jsonlite::fromJSON(log_path)
    tau <- lg$tau %||% tau
  }

  # Top 25 features per variant; long names shortened to head..tail.
  top <- sel %>%
    group_by(variant) %>%
    arrange(desc(selection_prob), .by_group = TRUE) %>%
    slice_head(n = 25) %>%
    ungroup() %>%
    mutate(feature = ifelse(nchar(feature) > 30,
                            paste0(substr(feature, 1, 14), "..",
                                   substr(feature, nchar(feature) - 11, nchar(feature))),
                            feature),
           feature = factor(feature, levels = rev(unique(feature[order(selection_prob)]))))

  p <- ggplot(top, aes(x = selection_prob, y = feature,
                       colour = selection_prob >= tau)) +
    geom_segment(aes(x = 0, xend = selection_prob, yend = feature),
                 linewidth = 0.4) +
    geom_point(size = 2.2) +
    geom_vline(xintercept = tau, linetype = "dashed", colour = "grey35", linewidth = 0.4) +
    scale_colour_manual(values = c(`TRUE` = "#1B7F79", `FALSE` = "grey72"), guide = "none") +
    scale_x_continuous(limits = c(0, 1)) +
    facet_wrap(~ variant, scales = "free_y") +
    labs(x = "selection probability", y = NULL)

  h <- max(4, 0.18 * nrow(top) / length(unique(top$variant)) + 2)
  ggsave(file.path(out, "stability_selection.png"), p,
         width = 11, height = h, dpi = 150, limitsize = FALSE)
  message(sprintf("[plot_analyses:stability] wrote %s",
                  file.path(out, "stability_selection.png")))

  for (v in unique(sel$variant)) {
    n <- sum(sel$variant == v & sel$stable)
    message(sprintf("[plot_analyses:stability] %-18s %d feature(s) with prob >= %.2f",
                    v, n, tau))
  }
})

# 4. L1 penalty selection: inner-CV curve per fold (1SE pick in green), chosen ratio and support
local({
  curve_path  <- run_path(run_id, "metrics", "lambda_cv_curve.csv")
  choice_path <- run_path(run_id, "metrics", "lambda_cv.csv")
  if (!file.exists(curve_path)) {
    message("[plot_analyses:lambda] no lambda_cv_curve.csv - skipping.")
    return(invisible(NULL))
  }
  out <- out_dir(run_id, "plots", "lambda_cv")
  theme_set(pub_theme)

  cur <- read_csv(curve_path, show_col_types = FALSE) %>%
    filter(!is.na(score_mean))
  if (nrow(cur) == 0) {
    message("[plot_analyses:lambda] lambda_cv_curve.csv has no scored rows - skipping.")
    return(invisible(NULL))
  }
  metric_lab <- if (length(unique(cur$metric))) as.character(unique(cur$metric)[1]) else "score"
  picks <- cur %>% filter(is_1se)

  p <- ggplot(cur, aes(x = ratio, y = score_mean, group = factor(outer_fold))) +
    geom_line(colour = "grey70", linewidth = 0.4) +
    geom_point(colour = "grey55", size = 1.1) +
    geom_point(data = picks, colour = "#1B7F79", size = 2.6) +
    scale_x_continuous(trans = scales::reverse_trans(),
                       breaks = sort(unique(cur$ratio)),
                       labels = function(v) sprintf("%.2f", v)) +
    facet_wrap(~ task, nrow = 1, scales = "free_y",
               labeller = labeller(task = response_label)) +
    labs(x = "lambda / lambda_max   (sparse -> dense)",
         y = sprintf("inner-CV %s", metric_lab))
  ggsave(file.path(out, "lambda_cv_curve.png"), p,
         width = 5.2 * length(unique(cur$task)), height = 4.2, dpi = 150)
  message(sprintf("[plot_analyses:lambda] wrote %s",
                  file.path(out, "lambda_cv_curve.png")))

  if (file.exists(choice_path)) {
    ch <- read_csv(choice_path, show_col_types = FALSE)
    ch2 <- rbind(
      data.frame(task = ch$task, quantity = "chosen lambda / lambda_max",
                 value = ch$chosen_ratio, stringsAsFactors = FALSE),
      data.frame(task = ch$task, quantity = "non-zero coefficients",
                 value = ch$n_nonzero, stringsAsFactors = FALSE)
    )
    pc <- ggplot(ch2, aes(x = response_label(task), y = value)) +
      geom_boxplot(width = 0.5, alpha = 0.85, fill = "#4C78A8",
                   outlier.shape = NA, colour = "grey25", linewidth = 0.3) +
      geom_point(position = position_jitter(width = 0.08, height = 0, seed = 1),
                 size = 1.6, colour = "grey20", alpha = 0.8) +
      facet_wrap(~ quantity, scales = "free_y") +
      labs(x = NULL, y = NULL)
    ggsave(file.path(out, "lambda_cv_choice.png"), pc,
           width = 8, height = 4.2, dpi = 150)
    message(sprintf("[plot_analyses:lambda] wrote %s",
                    file.path(out, "lambda_cv_choice.png")))

    # How often the pick sits at the largest ratio of the grid.
    gmax <- max(cur$ratio)
    for (tk in unique(ch$task)) {
      r <- ch$chosen_ratio[ch$task == tk]
      message(sprintf("[plot_analyses:lambda] %-8s chosen ratio min=%.3f max=%.3f | at the grid maximum in %d/%d folds",
                      tk, min(r), max(r), sum(r >= gmax), length(r)))
    }
  }
})

message("[plot_analyses] done.")
