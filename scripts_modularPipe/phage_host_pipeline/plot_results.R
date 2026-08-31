# plot_results.R  (performance plots)
#
# One image per (task, metric). Every plot is written TWICE:
#   *_all.png       — all models (baselines + latent-space MLPs), as before
#   *_baseline.png  — baseline models only (latent MLPs removed)
# Titles only (no explanatory subtitles/captions). Where a boxplot summarizes a
# distribution, the per-model median value is printed on the plot so the reader
# sees both the shape and the number. Colors are fixed per model via the shared
# palette in load_results.R, so a model keeps the same color across every image.

# Bootstrap: locate this file so load_results.R can be sourced beside it. This
# cannot live in load_results.R itself (we need it to find that file), so it is
# kept to two lines. get_script_dir() is defined there for anything downstream.
source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE),
       value = TRUE)[1])), "load_results.R"))

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(scales)
})

`%||%` <- function(a, b) if (is.null(a)) b else a

run_id <- get_run_id()
out    <- plots_results_dir(run_id)
res    <- load_results(run_id)
message(sprintf("[plot_results] dataset=%s run=%s - starting results plots",
                DATASET_NAME, run_id))

# Publication theme comes from load_results.R so every figure matches.
# Model names on x, and the colour is redundant with the axis -> no legend.
theme_set(pub_theme + theme_categorical_x() + theme(legend.position = "none"))

# pretty task labels (display only; filenames use the raw task name)
prettify_task <- function(x) {
  m <- c("y" = "Y", "y_bin" = "Y", "w_pos" = "W+", "w_neg" = "W-",
         "w_any" = "W > 0", "w_reg" = "W", "w_class" = "W class (1[W>0])")
  out <- m[x]; out[is.na(out)] <- x[is.na(out)]; unname(out)
}

# Two model views for every plot: baselines only, and everything.
# Returns a named list of data frames (a variant is dropped if it is empty or
# identical is fine — we still emit both so the file set is predictable).
model_variants <- function(df) {
  list(
    all      = df,
    baseline = df %>% filter(!is_latent_model(model))
  )
}

# experiment.py writes only role == "test" rows now — the training-split rows
# existed solely for the generalization-gap figures, which were removed. The
# filter is kept so an OLD run's metrics_by_fold.csv still plots correctly.
eval_metrics <- res$metrics %>% filter(role != "train")

# One boxplot for a single (task, metric), for a single model-variant.
save_metric_plot <- function(df_task, metric_col, metric_label, task_name, task_label,
                             n_folds, higher_better = TRUE, auc_scale = FALSE) {
  if (!(metric_col %in% names(df_task))) return(invisible(NULL))
  base_d <- df_task %>% filter(!is.na(.data[[metric_col]]))
  if (nrow(base_d) == 0) return(invisible(NULL))

  for (variant in names(model_variants(base_d))) {
    d <- model_variants(base_d)[[variant]]
    if (nrow(d) == 0) next
    d$model <- order_models(d$model)          # fixed axis order
    pal <- pal_for(d$model)
    med <- d %>% group_by(model) %>%
      summarise(v = median(.data[[metric_col]], na.rm = TRUE), .groups = "drop")

    p <- ggplot(d, aes(x = model, y = .data[[metric_col]], fill = model)) +
      geom_boxplot(width = 0.6, outlier.size = 0.7, outlier.alpha = 0.5,
                   alpha = 0.85, color = "grey25", linewidth = 0.3) +
      geom_text(data = med, aes(x = model, y = v, label = sprintf("%.3f", v)),
                inherit.aes = FALSE, vjust = -0.6, size = 3, color = "grey15") +
      scale_fill_manual(values = pal) +
      labs(title = sprintf("%s - %s", task_label, metric_label), x = NULL, y = metric_label)
    if (auc_scale) {
      p <- p +
        geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey50", linewidth = 0.4) +
        scale_y_continuous(limits = c(0.4, 1.0), breaks = seq(0.4, 1.0, 0.1))
    }
    fname <- sprintf("%s_%s_%s.png", task_name, metric_col, variant)
    ggsave(file.path(out, fname), p, width = 5.2, height = 4.2)
    message("[plot_results] wrote ", fname)
  }
}

# ---------------------------------------------------------------------------
# BINARY tasks: one image per (task, metric)
# ---------------------------------------------------------------------------
inner_bin <- eval_metrics %>% filter(task_type == "binary")
if (nrow(inner_bin) > 0) {
  n_folds <- n_distinct(inner_bin$outer_fold)

  task_order <- c("y", "w_class")
  bin_tasks  <- c(intersect(task_order, unique(inner_bin$task)),
                  setdiff(unique(inner_bin$task), task_order))

  # AUC and AP only. Brier, and the per-fold and ROC images, were dropped:
  # the single summary image per (task, metric) carries the same information.
  bin_metrics <- list(
    auc = list(label = "AUC",               auc = TRUE),
    ap  = list(label = "Average Precision", auc = FALSE)
  )

  for (tk in bin_tasks) {
    df_tk <- inner_bin %>% filter(task == tk)
    tlab  <- prettify_task(tk)
    for (mc in names(bin_metrics)) {
      spec <- bin_metrics[[mc]]
      save_metric_plot(df_tk, mc, spec$label, tk, tlab, n_folds, auc_scale = spec$auc)
    }

  }
}

# ---------------------------------------------------------------------------
# REGRESSION tasks: one image per (task, metric). No predicted-vs-actual plot.
# ---------------------------------------------------------------------------
inner_reg <- eval_metrics %>% filter(task_type == "regression")
if (nrow(inner_reg) > 0) {
  n_folds <- n_distinct(inner_reg$outer_fold)

  # R2 only; RMSE and MAE were dropped as redundant with it.
  reg_metrics <- list(r2 = "R2")
  for (tk in unique(inner_reg$task)) {
    df_tk <- inner_reg %>% filter(task == tk)
    tlab  <- prettify_task(tk)
    for (mc in names(reg_metrics)) {
      save_metric_plot(df_tk, mc, reg_metrics[[mc]], tk, tlab, n_folds)
    }
  }
}

# ---------------------------------------------------------------------------
# OUTER TEST SUMMARY (single fit per outer split) - CSV
# ---------------------------------------------------------------------------
if (nrow(eval_metrics) > 0) {
  outer_summary <- eval_metrics %>%
    mutate(task_label = prettify_task(task)) %>%
    group_by(task, task_label, task_type, model) %>%
    summarise(
      across(any_of(c("auc", "ap", "r2")),
             list(mean = ~mean(.x, na.rm = TRUE), sd = ~sd(.x, na.rm = TRUE)),
             .names = "{.col}_{.fn}"),
      n_folds = n(),
      .groups = "drop"
    )
  write.csv(outer_summary, file.path(out, "outer_test_summary.csv"), row.names = FALSE)
}

cat(sprintf("[plot_results] wrote %d files to %s\n",
            length(list.files(out, pattern = "\\.(png|csv)$")), out))
