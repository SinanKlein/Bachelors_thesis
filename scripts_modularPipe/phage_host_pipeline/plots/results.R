# results.R  —  prediction performance (experiment.py) -> plots/results/
# One boxplot per (task, metric), written twice: *_all.png and *_baseline.png
# (without mlp_latent_*), plus outer_test_summary.csv.

source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE),
       value = TRUE)[1])), "common.R"))
suppressPackageStartupMessages(library(scales))

run_id  <- get_run_id()
out     <- out_dir(run_id, "plots", "results")
metrics <- read_run_csv(run_id, "metrics", "metrics_by_fold.csv")
message(sprintf("[results] dataset=%s run=%s", DATASET_NAME, run_id))

theme_set(pub_theme + theme_categorical_x() + theme(legend.position = "none"))

save_metric_plot <- function(df, metric, label, task, auc_scale = FALSE) {
  if (!(metric %in% names(df))) return(invisible(NULL))
  df <- df %>% filter(!is.na(.data[[metric]]))
  variants <- list(all = df, baseline = df %>% filter(!is_latent_model(model)))
  for (variant in names(variants)) {
    d <- variants[[variant]]
    if (nrow(d) == 0) next
    d$model <- order_models(d$model)
    med <- d %>% group_by(model) %>%
      summarise(v = median(.data[[metric]], na.rm = TRUE), .groups = "drop")
    p <- ggplot(d, aes(x = model, y = .data[[metric]], fill = model)) +
      geom_boxplot(width = 0.6, outlier.size = 0.7, outlier.alpha = 0.5,
                   alpha = 0.85, color = "grey25", linewidth = 0.3) +
      geom_text(data = med, aes(x = model, y = v, label = sprintf("%.3f", v)),
                inherit.aes = FALSE, vjust = -0.6, size = 3, color = "grey15") +
      scale_fill_manual(values = pal_for(d$model)) +
      labs(x = NULL, y = label)
    if (auc_scale) {
      p <- p +
        geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey50", linewidth = 0.4) +
        scale_y_continuous(limits = c(0.4, 1.0), breaks = seq(0.4, 1.0, 0.1))
    }
    fname <- sprintf("%s_%s_%s.png", task, metric, variant)
    ggsave(file.path(out, fname), p, width = 4.6, height = 5.2)
    message("[results] wrote ", fname)
  }
}

# Binary tasks: AUC and AP.
bin <- metrics %>% filter(task_type == "binary")
for (tk in c(intersect(c("y", "w_class"), unique(bin$task)), setdiff(unique(bin$task), c("y", "w_class")))) {
  save_metric_plot(bin %>% filter(task == tk), "auc", "AUC", tk, auc_scale = TRUE)
  save_metric_plot(bin %>% filter(task == tk), "ap", "Average Precision", tk)
}

# Regression tasks: R2.
reg <- metrics %>% filter(task_type == "regression")
for (tk in unique(reg$task)) {
  save_metric_plot(reg %>% filter(task == tk), "r2", "R2", tk)
}

# Mean / sd per task x model.
if (nrow(metrics) > 0) {
  summary <- metrics %>%
    mutate(task_label = response_label(task)) %>%
    group_by(task, task_label, task_type, model) %>%
    summarise(across(any_of(c("auc", "ap", "r2")),
                     list(mean = ~mean(.x, na.rm = TRUE), sd = ~sd(.x, na.rm = TRUE)),
                     .names = "{.col}_{.fn}"),
              n_folds = n(), .groups = "drop")
  write.csv(summary, file.path(out, "outer_test_summary.csv"), row.names = FALSE)
}
message(sprintf("[results] %d files in %s", length(list.files(out, pattern = "\\.(png|csv)$")), out))
