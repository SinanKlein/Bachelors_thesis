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
message(sprintf("[plot_results] dataset=%s run=%s — starting results plots",
                DATASET_NAME, run_id))

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

# simplified baseline: only outer train/test splits exist
eval_metrics <- res$metrics %>% filter(role != "train")
outer <- eval_metrics
inner <- eval_metrics  # legacy object name used below; now means test-split metrics

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
      subtitle = sprintf("AUC across %d train/test splits | dashed line = chance",
                         n_distinct(inner_bin$outer_fold)),
      x = NULL, y = "AUC",
      caption  = sprintf("run = %s", run_id)
    )
  ggsave(file.path(out, "01_main_binary_AUC.png"), p1,
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
      subtitle = "Test-split distribution per task × model",
      x = NULL, y = NULL,
      caption  = sprintf("run = %s", run_id)
    )
  ggsave(file.path(out, "02_main_binary_all_metrics.png"), p2,
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
      subtitle = "Test-split values shown as dots; crossbar = fold mean",
      x = "outer fold", y = "AUC",
      caption  = sprintf("run = %s", run_id)
    ) +
    theme(panel.grid.major.x = element_line(color = "grey95", linewidth = 0.2))
  ggsave(file.path(out, "03_per_fold_AUC.png"), p3,
         width = 10, height = 4)

  #  04. Cutoff sweep
  if (!is.null(res$cutoff) && nrow(res$cutoff) > 0) {
    cut_summary <- res$cutoff %>%
      filter(role == "test") %>%
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
        subtitle = "Mean ± 1 SD across test splits",
        x = "probability cutoff", y = "metric value",
        caption  = sprintf("run = %s", run_id)
      )
    ggsave(file.path(out, "04_cutoff_sweep.png"), p4,
           width = 10, height = 5)

    #  05. ROC curves (mean curve + ±1 SD band on TPR across test splits)
    roc_df <- res$cutoff %>%
      filter(role == "test") %>%
      mutate(task_label = prettify_task(task)) %>%
      group_by(task_label, model, cutoff) %>%
      summarise(tpr_mean = mean(tpr, na.rm = TRUE),
                tpr_sd   = sd(tpr,  na.rm = TRUE),
                fpr      = mean(fpr, na.rm = TRUE),
                .groups = "drop") %>%
      mutate(tpr_sd = ifelse(is.na(tpr_sd), 0, tpr_sd)) %>%
      arrange(task_label, model, fpr)

    p5 <- ggplot(roc_df, aes(x = fpr, y = tpr_mean, color = model, fill = model)) +
      geom_abline(intercept = 0, slope = 1, color = "grey70",
                  linetype = "dashed", linewidth = 0.4) +
      geom_ribbon(aes(ymin = pmax(tpr_mean - tpr_sd, 0),
                      ymax = pmin(tpr_mean + tpr_sd, 1)),
                  alpha = 0.15, color = NA) +
      geom_path(linewidth = 0.9, alpha = 0.85) +
      facet_wrap(~ task_label, nrow = 1) +
      scale_color_manual(values = pal, name = "model") +
      scale_fill_manual(values = pal, name = "model") +
      coord_equal() +
      scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25)) +
      scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25)) +
      labs(
        title    = "ROC curves",
        subtitle = "Mean TPR vs mean FPR across test splits; shaded band = ±1 SD of TPR; dashed = chance",
        x = "false positive rate", y = "true positive rate",
        caption  = sprintf("run = %s", run_id)
      )
    ggsave(file.path(out, "05_ROC_curves.png"), p5,
           width = 10, height = 4)
    message("[plot_results] wrote 05_ROC_curves.png (with +/-1 SD band)")
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
      subtitle = "Test-split distribution per task × model",
      x = NULL, y = NULL,
      caption  = sprintf("run = %s", run_id)
    )
  ggsave(file.path(out, "07_regression_metrics.png"), p7,
         width = 11, height = 4)

  # 08. Predicted vs actual scatter 
  preds_reg <- res$predictions %>%
    filter(task_type == "regression", role == "test") %>%
    mutate(task_label = prettify_task(task))

  if (nrow(preds_reg) > 0) {
    n_per_facet <- preds_reg %>% count(task_label, model) %>% pull(n) %>% max()
    lim8 <- range(c(preds_reg$y_true, preds_reg$y_pred), na.rm = TRUE)
    use_hex <- n_per_facet > 5000 && requireNamespace("hexbin", quietly = TRUE)

    if (use_hex) {
      p8 <- ggplot(preds_reg, aes(x = y_true, y = y_pred)) +
        geom_hex(bins = 40, alpha = 0.95) +
        geom_abline(slope = 1, intercept = 0, color = "red",
                    linetype = "dashed", linewidth = 0.5) +
        facet_grid(task_label ~ model) +
        scale_fill_viridis_c(option = "magma", trans = "log10",
                             labels = scales::label_number(),
                             name = "count") +
        coord_fixed(ratio = 1, xlim = lim8, ylim = lim8) +
        labs(title    = "Predicted vs actual",
             subtitle = "Pooled across test splits; dashed = identity",
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
        coord_fixed(ratio = 1, xlim = lim8, ylim = lim8) +
        labs(title    = "Predicted vs actual",
             subtitle = "Pooled across test splits; red dashed = identity, blue = linear fit",
             x = "true", y = "predicted",
             caption  = sprintf("run = %s", run_id))
    }
    ggsave(file.path(out, "08_pred_vs_actual.png"), p8,
           width = 8, height = 5)

  }
}

# GENERALIZATION GAP
# Compares train set performance to test-set performance per (task, model, fold).

if ("train" %in% unique(res$metrics$role)) {

  #  11. Binary generalization gap
  gap_bin <- res$metrics %>%
    filter(task_type == "binary",
           role %in% c("train", "test")) %>%
    select(outer_fold, inner_fold, task, model, role, auc) %>%
    pivot_wider(names_from = role, values_from = auc) %>%
    filter(!is.na(train), !is.na(test)) %>%
    mutate(gap = train - test,
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
        subtitle = "AUC(train) − AUC(test) per test split | larger = more optimistic train fit",
        x = NULL, y = "AUC gap",
        caption  = sprintf("run = %s | dashed line = no gap", run_id)
      )
    ggsave(file.path(out, "11_generalization_gap_binary.png"), p11,
           width = 8, height = 4.5)
  }

  # 12. Regression generalization gap 
  gap_reg <- res$metrics %>%
    filter(task_type == "regression",
           role %in% c("train", "test")) %>%
    select(outer_fold, inner_fold, task, model, role, r2) %>%
    pivot_wider(names_from = role, values_from = r2) %>%
    filter(!is.na(train), !is.na(test)) %>%
    mutate(gap = train - test,
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
        subtitle = "R²(train) − R²(test) per test split | larger = more optimistic train fit",
        x = NULL, y = "R² gap",
        caption  = sprintf("run = %s | dashed line = no gap", run_id)
      )
    ggsave(file.path(out, "12_generalization_gap_regression.png"), p12,
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

# Legacy W-threshold sweep, lasso stability-selection, and XGBoost-grid plots
# are intentionally removed for the simplified baseline.

cat(sprintf("[plot_results] wrote %d files to %s\n",
            length(list.files(out, pattern = "\\.(png|csv)$")), out))
