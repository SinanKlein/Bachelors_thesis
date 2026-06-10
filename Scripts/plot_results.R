# plot_results.R  (performance plots)


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
  library(patchwork)   # used for the XGB grid plot (heatmap + strip plot)
})

`%||%` <- function(a, b) if (is.null(a)) b else a

run_id <- get_run_id()
out    <- plots_results_dir(run_id)
res    <- load_results(run_id)

# Publication theme

pub_theme <- theme_minimal(base_size = 12) +
  theme(
    plot.title       = element_text(face = "bold", size = 13),
    plot.subtitle    = element_text(color = "grey40", size = 10),
    plot.caption     = element_text(color = "grey55", size = 8, hjust = 0),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "grey92", linewidth = 0.3),
    strip.background = element_rect(fill = "grey95", color = NA),
    strip.text       = element_text(face = "bold", size = 11),
    axis.title       = element_text(size = 11),
    axis.text        = element_text(size = 10, color = "grey25"),
    axis.text.x      = element_text(size = 11, color = "grey15"),
    legend.position  = "right",
    legend.title     = element_text(size = 10, face = "bold"),
    legend.text      = element_text(size = 9),
    legend.key.size  = unit(0.8, "lines"),
    plot.margin      = margin(10, 12, 10, 10)
  )
theme_set(pub_theme)

model_palette <- c(
  "#4C78A8",  # blue
  "#F58518",  # orange
  "#54A24B",  # green
  "#E45756",  # red
  "#B279A2",  # purple
  "#9D755D"   # brown
)

# split inner vs outer once
inner <- res$metrics %>% filter(inner_fold != -1, role != "train")
outer <- res$metrics %>% filter(inner_fold == -1, role != "train")

inner_bin <- inner %>% filter(task_type == "binary")
inner_reg <- inner %>% filter(task_type == "regression")

# pretty task labels: y for AUC(Y), w_neg for AUC(W⁻), w_pos for AUC(W⁺), etc.
# This map handles common task names; unknown names pass through unchanged.
prettify_task <- function(x) {
  m <- c(
    "y"     = "Y",
    "y_bin" = "Y",
    "w_pos" = "W⁺",
    "w_neg" = "W⁻",
    "w_any" = "W > 0",
    "w_reg" = "W"
  )
  out <- m[x]
  out[is.na(out)] <- x[is.na(out)]
  unname(out)
}

# BINARY 

if (nrow(inner_bin) > 0) {

  inner_bin <- inner_bin %>% mutate(task_label = prettify_task(task))

  # ordering: lock task order so panels read left-to-right consistently
  task_order <- c("Y", "W⁻", "W⁺", "W > 0", "W")
  inner_bin$task_label <- factor(
    inner_bin$task_label,
    levels = c(intersect(task_order, unique(inner_bin$task_label)),
               setdiff(unique(inner_bin$task_label), task_order))
  )

  models_present <- unique(inner_bin$model)
  pal <- setNames(model_palette[seq_along(models_present)], models_present)

  #  01. Main AUC plot
  p1 <- ggplot(inner_bin, aes(x = task_label, y = auc, fill = model)) +
    geom_boxplot(
      position  = position_dodge(width = 0.75),
      width     = 0.6,
      outlier.size  = 0.7,
      outlier.alpha = 0.5,
      alpha     = 0.85,
      color     = "grey25",
      linewidth = 0.3
    ) +
    geom_hline(yintercept = 0.5, linetype = "dashed",
               color = "grey50", linewidth = 0.4) +
    scale_fill_manual(values = pal, name = "model") +
    scale_y_continuous(limits = c(0.4, 1.0),
                       breaks = seq(0.4, 1.0, 0.1),
                       expand = expansion(mult = c(0, 0.02))) +
    labs(
      title    = "Binary task performance",
      subtitle = sprintf("AUC across %d inner-CV folds | dashed line = chance",
                         n_distinct(inner_bin[, c("outer_fold","inner_fold")])),
      x = NULL, y = "AUC",
      caption  = sprintf("run = %s", run_id)
    )
  ggsave(file.path(out, "01_main_binary_AUC.pdf"), p1,
         width = 8, height = 4.5)

  # 02. Same layout, all three binary metrics 
  metric_long <- inner_bin %>%
    pivot_longer(c(auc, ap, brier), names_to = "metric", values_to = "value") %>%
    mutate(metric = factor(metric,
                           levels = c("auc", "ap", "brier"),
                           labels = c("AUC", "Average Precision", "Brier (lower = better)")))

  p2 <- ggplot(metric_long, aes(x = task_label, y = value, fill = model)) +
    geom_boxplot(position = position_dodge(width = 0.75),
                 width = 0.6, outlier.size = 0.7, outlier.alpha = 0.5,
                 alpha = 0.85, color = "grey25", linewidth = 0.3) +
    facet_wrap(~ metric, scales = "free_y", nrow = 1) +
    scale_fill_manual(values = pal, name = "model") +
    labs(
      title    = "Binary metrics across tasks and models",
      subtitle = "Inner-CV distribution per task × model",
      x = NULL, y = NULL,
      caption  = sprintf("run = %s", run_id)
    )
  ggsave(file.path(out, "02_main_binary_all_metrics.pdf"), p2,
         width = 11, height = 4.5)

  # 03. Per-outer-fold AUC 
  p3 <- ggplot(inner_bin,
               aes(x = factor(outer_fold), y = auc, color = model)) +
    geom_jitter(width = 0.18, height = 0, alpha = 0.55, size = 1.4) +
    stat_summary(fun = mean, geom = "crossbar",
                 width = 0.65, alpha = 0.7, linewidth = 0.3,
                 fatten = 1.5) +
    geom_hline(yintercept = 0.5, linetype = "dashed",
               color = "grey50", linewidth = 0.4) +
    facet_wrap(~ task_label, nrow = 1) +
    scale_color_manual(values = pal, name = "model") +
    scale_y_continuous(limits = c(0.4, 1.0)) +
    labs(
      title    = "AUC per outer fold",
      subtitle = "Inner-CV values shown as dots; crossbar = fold mean",
      x = "outer fold", y = "AUC",
      caption  = sprintf("run = %s", run_id)
    ) +
    theme(panel.grid.major.x = element_line(color = "grey95", linewidth = 0.2))
  ggsave(file.path(out, "03_per_fold_AUC.pdf"), p3,
         width = 10, height = 4)

  #  04. Cutoff sweep
  if (!is.null(res$cutoff) && nrow(res$cutoff) > 0) {
    cut_summary <- res$cutoff %>%
      filter(inner_fold != -1) %>%
      mutate(task_label = prettify_task(task)) %>%
      pivot_longer(c(precision, recall, f1, accuracy),
                   names_to = "metric", values_to = "value") %>%
      mutate(metric = factor(metric,
                             levels = c("precision", "recall", "f1", "accuracy"),
                             labels = c("Precision", "Recall", "F1", "Accuracy"))) %>%
      group_by(task_label, model, metric, cutoff) %>%
      summarise(mean = mean(value, na.rm = TRUE),
                sd   = sd(value, na.rm = TRUE),
                .groups = "drop")

    p4 <- ggplot(cut_summary,
                 aes(x = cutoff, y = mean, color = metric, fill = metric)) +
      geom_ribbon(aes(ymin = pmax(mean - sd, 0),
                      ymax = pmin(mean + sd, 1)),
                  alpha = 0.15, color = NA) +
      geom_line(linewidth = 0.7) +
      facet_grid(task_label ~ model) +
      scale_color_brewer(palette = "Dark2", name = NULL) +
      scale_fill_brewer(palette = "Dark2", name = NULL) +
      coord_cartesian(ylim = c(0, 1)) +
      labs(
        title    = "Performance vs classification cutoff",
        subtitle = "Mean ± 1 SD across inner CV folds",
        x = "probability cutoff", y = "metric value",
        caption  = sprintf("run = %s", run_id)
      )
    ggsave(file.path(out, "04_cutoff_sweep.pdf"), p4,
           width = 10, height = 5)

    #  05. ROC curves
    roc_df <- res$cutoff %>%
      filter(inner_fold != -1) %>%
      mutate(task_label = prettify_task(task)) %>%
      group_by(task_label, model, cutoff) %>%
      summarise(tpr = mean(tpr, na.rm = TRUE),
                fpr = mean(fpr, na.rm = TRUE),
                .groups = "drop") %>%
      arrange(task_label, model, fpr)

    p5 <- ggplot(roc_df, aes(x = fpr, y = tpr, color = model)) +
      geom_abline(intercept = 0, slope = 1, color = "grey70",
                  linetype = "dashed", linewidth = 0.4) +
      geom_path(linewidth = 0.9, alpha = 0.85) +
      facet_wrap(~ task_label, nrow = 1) +
      scale_color_manual(values = pal, name = "model") +
      coord_equal() +
      scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25)) +
      scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25)) +
      labs(
        title    = "ROC curves",
        subtitle = "TPR and FPR averaged across inner folds; dashed = chance",
        x = "false positive rate", y = "true positive rate",
        caption  = sprintf("run = %s", run_id)
      )
    ggsave(file.path(out, "05_ROC_curves.pdf"), p5,
           width = 10, height = 4)
  }

  # 06. Calibration / reliability diagram
  preds_bin <- res$predictions %>%
    filter(task_type == "binary", inner_fold != -1) %>%
    mutate(task_label = prettify_task(task))

  if (nrow(preds_bin) > 0) {
    calib <- preds_bin %>%
      mutate(bin = cut(prob, breaks = seq(0, 1, 0.1),
                       include.lowest = TRUE, labels = FALSE)) %>%
      group_by(task_label, model, bin) %>%
      summarise(predicted = mean(prob),
                observed  = mean(y_true),
                n         = n(),
                .groups = "drop")

    p6 <- ggplot(calib, aes(x = predicted, y = observed, color = model)) +
      geom_abline(intercept = 0, slope = 1, color = "grey70",
                  linetype = "dashed", linewidth = 0.4) +
      geom_line(linewidth = 0.7, alpha = 0.85) +
      geom_point(aes(size = n), alpha = 0.85) +
      facet_wrap(~ task_label, nrow = 1) +
      scale_color_manual(values = pal, name = "model") +
      scale_size_continuous(range = c(1, 4), name = "n in bin",
                            labels = scales::label_comma()) +
      coord_equal() +
      scale_x_continuous(limits = c(0, 1)) +
      scale_y_continuous(limits = c(0, 1)) +
      labs(
        title    = "Calibration (reliability diagram)",
        subtitle = "Predicted vs observed positive rate, binned by predicted prob",
        x = "mean predicted probability", y = "observed positive fraction",
        caption  = sprintf("run = %s | bins = deciles of predicted prob", run_id)
      )
    ggsave(file.path(out, "06_calibration.pdf"), p6,
           width = 10, height = 4.5)
  }
}

# REGRESSION

if (nrow(inner_reg) > 0) {

  inner_reg <- inner_reg %>% mutate(task_label = prettify_task(task))
  models_present_r <- unique(inner_reg$model)
  pal_r <- setNames(model_palette[seq_along(models_present_r)], models_present_r)

  #  07. Regression metrics box plot 
  metric_long <- inner_reg %>%
    pivot_longer(c(r2, rmse, mae), names_to = "metric", values_to = "value") %>%
    mutate(metric = factor(metric,
                           levels = c("r2", "rmse", "mae"),
                           labels = c("R²", "RMSE", "MAE")))

  p7 <- ggplot(metric_long, aes(x = task_label, y = value, fill = model)) +
    geom_boxplot(position = position_dodge(width = 0.75), width = 0.6,
                 outlier.size = 0.7, outlier.alpha = 0.5,
                 alpha = 0.85, color = "grey25", linewidth = 0.3) +
    facet_wrap(~ metric, scales = "free_y", nrow = 1) +
    scale_fill_manual(values = pal_r, name = "model") +
    labs(
      title    = "Regression performance",
      subtitle = "Inner-CV distribution per task × model",
      x = NULL, y = NULL,
      caption  = sprintf("run = %s", run_id)
    )
  ggsave(file.path(out, "07_regression_metrics.pdf"), p7,
         width = 11, height = 4)

  # 08. Predicted vs actual scatter 
  preds_reg <- res$predictions %>%
    filter(task_type == "regression", inner_fold != -1) %>%
    mutate(task_label = prettify_task(task))

  if (nrow(preds_reg) > 0) {
    n_per_facet <- preds_reg %>% count(task_label, model) %>% pull(n) %>% max()
    use_hex <- n_per_facet > 5000

    if (use_hex) {
      p8 <- ggplot(preds_reg, aes(x = y_true, y = y_pred)) +
        geom_hex(bins = 40, alpha = 0.95) +
        geom_abline(slope = 1, intercept = 0, color = "red",
                    linetype = "dashed", linewidth = 0.5) +
        facet_grid(task_label ~ model) +
        scale_fill_viridis_c(option = "magma", trans = "log10",
                             labels = scales::label_number(),
                             name = "count") +
        coord_equal() +
        labs(title    = "Predicted vs actual",
             subtitle = "Pooled across inner folds; dashed = identity",
             x = "true", y = "predicted",
             caption  = sprintf("run = %s", run_id))
    } else {
      p8 <- ggplot(preds_reg, aes(x = y_true, y = y_pred)) +
        geom_point(alpha = 0.2, size = 0.6, color = "grey30") +
        geom_abline(slope = 1, intercept = 0, color = "red",
                    linetype = "dashed", linewidth = 0.5) +
        geom_smooth(method = "lm", se = FALSE, color = "#4C78A8",
                    linewidth = 0.5, formula = y ~ x) +
        facet_grid(task_label ~ model) +
        coord_equal() +
        labs(title    = "Predicted vs actual",
             subtitle = "Pooled across inner folds; red dashed = identity, blue = linear fit",
             x = "true", y = "predicted",
             caption  = sprintf("run = %s", run_id))
    }
    ggsave(file.path(out, "08_pred_vs_actual.pdf"), p8,
           width = 8, height = 5)

    # 09. Residuals 
    preds_reg <- preds_reg %>% mutate(residual = y_pred - y_true)
    p9 <- ggplot(preds_reg, aes(x = y_pred, y = residual)) +
      geom_point(alpha = 0.2, size = 0.6, color = "grey30") +
      geom_hline(yintercept = 0, color = "red",
                 linetype = "dashed", linewidth = 0.5) +
      geom_smooth(method = "loess", se = TRUE, color = "#4C78A8",
                  linewidth = 0.5, alpha = 0.2, formula = y ~ x) +
      facet_grid(task_label ~ model) +
      labs(title    = "Residuals (predicted − actual)",
           subtitle = "Blue = LOESS trend; flat ≈ unbiased",
           x = "predicted", y = "residual",
           caption  = sprintf("run = %s", run_id))
    ggsave(file.path(out, "09_residuals.pdf"), p9,
           width = 8, height = 5)
  }
}

# GENERALIZATION GAP
# Compares train set performance to val-set performance per (task, model, fold).

if ("train" %in% unique(res$metrics$role)) {

  #  11. Binary generalization gap
  gap_bin <- res$metrics %>%
    filter(task_type == "binary",
           inner_fold != -1,
           role %in% c("train", "val")) %>%
    select(outer_fold, inner_fold, task, model, role, auc) %>%
    pivot_wider(names_from = role, values_from = auc) %>%
    filter(!is.na(train), !is.na(val)) %>%
    mutate(gap = train - val,
           task_label = prettify_task(task))

  if (nrow(gap_bin) > 0) {
    # ordering: same as plot 01
    task_order <- c("Y", "W⁻", "W⁺", "W > 0", "W")
    gap_bin$task_label <- factor(
      gap_bin$task_label,
      levels = c(intersect(task_order, unique(gap_bin$task_label)),
                 setdiff(unique(gap_bin$task_label), task_order))
    )
    models_present <- unique(gap_bin$model)
    pal_gap <- setNames(model_palette[seq_along(models_present)], models_present)

    # summary annotations: mean gap per model × task
    gap_means <- gap_bin %>%
      group_by(task_label, model) %>%
      summarise(mean_gap = mean(gap), .groups = "drop")

    p11 <- ggplot(gap_bin, aes(x = task_label, y = gap, fill = model)) +
      geom_hline(yintercept = 0, linetype = "dashed",
                 color = "grey50", linewidth = 0.4) +
      geom_boxplot(position = position_dodge(width = 0.75),
                   width = 0.6, outlier.size = 0.7, outlier.alpha = 0.5,
                   alpha = 0.85, color = "grey25", linewidth = 0.3) +
      scale_fill_manual(values = pal_gap, name = "model") +
      labs(
        title    = "Generalization gap — binary tasks",
        subtitle = "AUC(train) − AUC(val) per inner fold | larger = more overfitting",
        x = NULL, y = "AUC gap",
        caption  = sprintf("run = %s | dashed line = no gap", run_id)
      )
    ggsave(file.path(out, "11_generalization_gap_binary.pdf"), p11,
           width = 8, height = 4.5)
  }

  # 12. Regression generalization gap 
  gap_reg <- res$metrics %>%
    filter(task_type == "regression",
           inner_fold != -1,
           role %in% c("train", "val")) %>%
    select(outer_fold, inner_fold, task, model, role, r2) %>%
    pivot_wider(names_from = role, values_from = r2) %>%
    filter(!is.na(train), !is.na(val)) %>%
    mutate(gap = train - val,
           task_label = prettify_task(task))

  if (nrow(gap_reg) > 0) {
    models_present_r <- unique(gap_reg$model)
    pal_gap_r <- setNames(model_palette[seq_along(models_present_r)], models_present_r)

    p12 <- ggplot(gap_reg, aes(x = task_label, y = gap, fill = model)) +
      geom_hline(yintercept = 0, linetype = "dashed",
                 color = "grey50", linewidth = 0.4) +
      geom_boxplot(position = position_dodge(width = 0.75),
                   width = 0.6, outlier.size = 0.7, outlier.alpha = 0.5,
                   alpha = 0.85, color = "grey25", linewidth = 0.3) +
      scale_fill_manual(values = pal_gap_r, name = "model") +
      labs(
        title    = "Generalization gap — regression tasks",
        subtitle = "R²(train) − R²(val) per inner fold | larger = more overfitting",
        x = NULL, y = "R² gap",
        caption  = sprintf("run = %s | dashed line = no gap", run_id)
      )
    ggsave(file.path(out, "12_generalization_gap_regression.pdf"), p12,
           width = 8, height = 4.5)
  }
}

# OUTER TEST SUMMARY (single fit per outer split)

if (nrow(outer) > 0) {
  outer_summary <- outer %>%
    mutate(task_label = prettify_task(task)) %>%
    group_by(task, task_label, task_type, model) %>%
    summarise(
      across(any_of(c("auc", "ap", "brier", "r2", "rmse", "mae")),
             list(mean = ~mean(.x, na.rm = TRUE),
                  sd   = ~sd(.x, na.rm = TRUE)),
             .names = "{.col}_{.fn}"),
      n_folds = n(),
      .groups = "drop"
    )
  write.csv(outer_summary,
            file.path(out, "10_outer_test_summary.csv"),
            row.names = FALSE)
}

# ===========================================================================
# 13. W-threshold sweep (Daniele's slide 21)
# ===========================================================================
w_sweep_path <- run_path(run_id, "metrics", "w_threshold_sweep.csv")

if (file.exists(w_sweep_path)) {
  ws <- read_csv(w_sweep_path, show_col_types = FALSE)

  ws_summary <- ws %>%
    group_by(panel, threshold) %>%
    summarise(
      auc_mean = mean(auc, na.rm = TRUE),
      auc_sd   = sd(auc,   na.rm = TRUE),
      n_folds  = sum(!is.na(auc)),
      .groups  = "drop"
    ) %>%
    filter(n_folds >= 3)

  panel_labels <- c("W>0" = "(W>0)|X",
                    "W-"  = "W\u207B |X",
                    "W+"  = "W\u207A |X")
  ws_summary$panel <- factor(ws_summary$panel,
                             levels = c("W>0", "W-", "W+"),
                             labels = panel_labels)

  p_wsweep <- ggplot(ws_summary,
                     aes(x = threshold, y = auc_mean)) +
    geom_errorbar(aes(ymin = auc_mean - auc_sd, ymax = auc_mean + auc_sd),
                  width = 0, color = "#4C78A8", alpha = 0.7) +
    geom_point(color = "#4C78A8", size = 1.8) +
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey60") +
    facet_wrap(~ panel, nrow = 1) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
    labs(
      title = "Logistic regression over edge probability thresholds of W",
      subtitle = paste0("10 train/test splits  \u2014  mean \u00b1 sd  \u2014  run ", run_id),
      x = "Threshold over W",
      y = "AUC"
    )

  ggsave(file.path(out, "13_w_threshold_sweep.pdf"),
         p_wsweep, width = 11, height = 4)
  message("[plot_results] wrote 13_w_threshold_sweep.pdf")
} else {
  message("[plot_results] w_threshold_sweep.csv not found, skipping slide-21 plot")
}

# ===========================================================================
# 14-16. Lasso stability selection
# ===========================================================================
lasso_summary_path <- run_path(run_id, "metrics", "lasso_path_summary.csv")

if (file.exists(lasso_summary_path)) {
  ls <- read_csv(lasso_summary_path, show_col_types = FALSE)

  src_palette <- c("Xb" = "#4C78A8", "Xv" = "#F58518")
  src_labels  <- c("Xb" = "Bacterial ProC", "Xv" = "Viral ProC")

  # --- 14. summary curves: # features selected vs lambda, by source -------
  curve_df <- ls %>%
    filter(selection_freq > 0) %>%
    group_by(log_lambda, feature_source) %>%
    summarise(
      n_any     = sum(selection_freq > 0),
      n_stable  = sum(selection_freq >= 0.8),
      .groups   = "drop"
    ) %>%
    pivot_longer(c(n_any, n_stable),
                 names_to = "kind", values_to = "count") %>%
    mutate(kind = recode(kind,
                         n_any    = "Any (\u2265 1 fold)",
                         n_stable = "Stable (\u2265 80% of folds)"))

  p_curve <- ggplot(curve_df,
                    aes(x = log_lambda, y = count,
                        color = feature_source, linetype = kind)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 1.5) +
    scale_color_manual(values = src_palette, labels = src_labels,
                       name = "Source") +
    scale_linetype_manual(values = c("Any (\u2265 1 fold)" = "dashed",
                                      "Stable (\u2265 80% of folds)" = "solid"),
                           name = "Selection") +
    labs(
      title = "Lasso stability selection \u2014 features kept vs penalty",
      subtitle = paste0("run ", run_id),
      x = expression(log[10](lambda)),
      y = "Number of features selected"
    )
  ggsave(file.path(out, "14_lasso_path_summary.pdf"),
         p_curve, width = 9, height = 5)

  # --- 15. heatmap: feature x lambda, selection frequency -----------------
  selected_ever <- ls %>%
    group_by(feature_id) %>%
    summarise(max_freq = max(selection_freq), .groups = "drop") %>%
    filter(max_freq > 0) %>%
    pull(feature_id)

  if (length(selected_ever) > 0) {
    hm <- ls %>%
      filter(feature_id %in% selected_ever) %>%
      group_by(feature_id) %>%
      mutate(max_freq = max(selection_freq)) %>%
      ungroup() %>%
      arrange(feature_source, desc(max_freq)) %>%
      mutate(feature_label = factor(
        paste0(feature_source, ": ", feature_name),
        levels = unique(paste0(feature_source, ": ", feature_name))
      ))

    p_hm <- ggplot(hm,
                   aes(x = log_lambda, y = feature_label,
                       fill = selection_freq)) +
      geom_tile() +
      scale_fill_gradient(low = "white", high = "#E45756",
                          limits = c(0, 1),
                          name = "Selection\nfrequency") +
      facet_grid(feature_source ~ ., scales = "free_y", space = "free_y",
                 labeller = labeller(feature_source = src_labels)) +
      labs(
        title = "Lasso selection frequency across penalty grid",
        subtitle = paste0(length(selected_ever),
                          " features ever selected, across 10 outer folds",
                          "  \u2014  run ", run_id),
        x = expression(log[10](lambda)),
        y = NULL
      ) +
      theme(
        axis.text.y  = element_text(size = 6),
        strip.text.y = element_text(angle = 0)
      )

    hm_height <- min(20, max(5, 0.12 * length(selected_ever)))
    ggsave(file.path(out, "15_lasso_heatmap.pdf"),
           p_hm, width = 10, height = hm_height, limitsize = FALSE)

    # --- 16. top features by max selection frequency ----------------------
    top_n <- 30
    top_feats <- ls %>%
      group_by(feature_id, feature_source, feature_name) %>%
      summarise(
        max_freq      = max(selection_freq),
        lambda_at_max = log_lambda[which.max(selection_freq)],
        mean_coef     = mean_coef[which.max(selection_freq)],
        .groups = "drop"
      ) %>%
      arrange(desc(max_freq)) %>%
      slice_head(n = top_n) %>%
      mutate(label = paste0(feature_source, ": ", feature_name),
             label = factor(label, levels = rev(label)))

    p_top <- ggplot(top_feats,
                    aes(x = max_freq, y = label, fill = feature_source)) +
      geom_col() +
      geom_text(aes(label = sprintf("%.2f", max_freq)),
                hjust = -0.1, size = 3) +
      scale_fill_manual(values = src_palette, labels = src_labels,
                        name = "Source") +
      scale_x_continuous(limits = c(0, 1.1),
                         breaks = seq(0, 1, 0.25)) +
      labs(
        title = paste0("Top ", min(top_n, nrow(top_feats)),
                       " features by max selection frequency"),
        subtitle = paste0("across the entire \u03bb grid  \u2014  run ", run_id),
        x = "Maximum selection frequency",
        y = NULL
      )
    ggsave(file.path(out, "16_lasso_top_features.pdf"),
           p_top, width = 9, height = 0.3 * nrow(top_feats) + 2)

    message("[plot_results] wrote 14_lasso_path_summary.pdf, ",
            "15_lasso_heatmap.pdf, 16_lasso_top_features.pdf")
  } else {
    message("[plot_results] No features ever selected by lasso; ",
            "skipping plots 15 + 16")
  }
} else {
  message("[plot_results] lasso_path_summary.csv not found, ",
          "skipping plots 14-16")
}

# ===========================================================================
# 17. XGBoost (max_depth x learning_rate) grid
# ===========================================================================
xgb_long_path    <- run_path(run_id, "metrics", "xgb_grid.csv")
xgb_summary_path <- run_path(run_id, "metrics", "xgb_grid_summary.csv")

if (file.exists(xgb_long_path) && file.exists(xgb_summary_path)) {
  xg  <- read_csv(xgb_long_path,    show_col_types = FALSE)
  xgs <- read_csv(xgb_summary_path, show_col_types = FALSE)

  xgs <- xgs %>%
    mutate(
      max_depth_f     = factor(max_depth,     levels = sort(unique(max_depth))),
      learning_rate_f = factor(learning_rate, levels = sort(unique(learning_rate)))
    )
  xg <- xg %>%
    mutate(
      max_depth_f     = factor(max_depth,     levels = sort(unique(max_depth))),
      learning_rate_f = factor(learning_rate, levels = sort(unique(learning_rate)))
    )

  best <- xgs %>% arrange(desc(auc_mean)) %>% slice_head(n = 1)

  p_hm <- ggplot(xgs,
                 aes(x = learning_rate_f, y = max_depth_f, fill = auc_mean)) +
    geom_tile(color = "white", linewidth = 1) +
    geom_text(aes(label = sprintf("%.3f\n\u00b1 %.3f",
                                   auc_mean, auc_sd)),
              size = 4, color = "black") +
    geom_tile(data = best,
              aes(x = learning_rate_f, y = max_depth_f),
              fill = NA, color = "#222222", linewidth = 1.5) +
    scale_fill_gradient(low = "#FFF5F0", high = "#E45756",
                        name = "Mean AUC") +
    labs(
      title = "XGBoost hyperparameter grid \u2014 mean AUC across 10 folds",
      subtitle = paste0("Best: max_depth=", best$max_depth,
                        ", learning_rate=", best$learning_rate,
                        "  (AUC=", sprintf("%.3f", best$auc_mean), ")",
                        "  \u2014  run ", run_id),
      x = "learning_rate",
      y = "max_depth"
    )

  xg <- xg %>%
    mutate(cell = paste0("d=", max_depth, "\nlr=", learning_rate))
  cell_order <- xgs %>%
    mutate(cell = paste0("d=", max_depth, "\nlr=", learning_rate)) %>%
    arrange(auc_mean) %>%
    pull(cell)
  xg <- xg %>% mutate(cell = factor(cell, levels = cell_order))

  p_strip <- ggplot(xg, aes(x = cell, y = auc)) +
    geom_jitter(width = 0.15, height = 0, alpha = 0.6,
                color = "#4C78A8", size = 1.8) +
    stat_summary(fun = mean, geom = "crossbar", width = 0.5,
                 color = "#222222", linewidth = 0.5) +
    labs(
      title = "Per-fold AUC by grid cell (ordered by mean)",
      x = NULL, y = "AUC (one point per outer fold)"
    ) +
    theme(axis.text.x = element_text(size = 8))

  p_combined <- p_hm / p_strip + plot_layout(heights = c(2, 1))
  ggsave(file.path(out, "17_xgb_grid.pdf"),
         p_combined, width = 9, height = 9)
  message("[plot_results] wrote 17_xgb_grid.pdf")
} else {
  message("[plot_results] xgb_grid files not found, skipping XGBoost grid plot")
}

cat(sprintf("[plot_results] wrote %d files to %s\n",
            length(list.files(out, pattern = "\\.(pdf|csv)$")), out))
