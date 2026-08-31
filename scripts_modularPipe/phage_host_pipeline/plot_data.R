# plot_data.R  —  everything descriptive about the data, before any model
# Two sections, previously two scripts. Neither depends on a model having been
# fit; both describe the inputs.
#
#   1. features + labels   reads preprocessing/*.csv written by preprocess.py
#                          Mean-variance feature selection, the score
#                          distribution, how many ProCs survive per source, label
#                          balance, the W distribution and split balance.
#
#   2. interaction structure  reads graph_matrices/*.csv written by graph_export.py
#                          interaction_Y_matrix.png        CRISPR edges
#                          interaction_W_matrix.png        glasso W, grey = unobserved
#                          interaction_combined_matrix.png both / CRISPR / glasso
#                          nestedness_Y.png, nestedness_W.png   degree-sorted, NODF
#                          interaction_{W,Y}_matrix_observed.png  the W-observed
#                                                         block only, no grey
#                          The three full-grid heatmaps share ONE clustering
#                          (computed once on observed W) so cells line up and can
#                          be compared side by side.
#
# Both write into plots/data/. Colors and the theme come from load_results.R.
#
# Usage:  Rscript plot_data.R <run_id> <DATASET_NAME>

source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE),
       value = TRUE)[1])), "load_results.R"))

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
  library(scales)
  library(tidyr)
})

`%||%` <- function(a, b) if (is.null(a)) b else a

run_id <- get_run_id()
message(sprintf("[plot_data] dataset=%s run=%s", DATASET_NAME, run_id))

# 1. Features, labels and splits
# Reads preprocessing/*.csv written by preprocess.py.
local({
  out    <- plots_data_dir(run_id)
  pre    <- load_preprocessing(run_id)
  message(sprintf("[plot_data] dataset=%s run=%s — starting data plots",
                  DATASET_NAME, run_id))

  # Publication theme (applied to every plot)

  # Theme: the shared base from load_results.R (no overrides needed here).
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
    ggsave(file.path(out, sprintf("0%d_mean_variance_%s.png",
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
      x = "mean (log)", y = "variance (log)",
      caption = sprintf("run = %s", run_id)
    )
  ggsave(file.path(out, "03_mean_variance_combined.png"),
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
      title    = "Raw-variance score distribution",
      x = "raw-variance score", y = "feature count",
      caption  = sprintf("run = %s", run_id)
    )
  ggsave(file.path(out, "04_score_distribution.png"), p4,
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
      x = NULL, y = "feature count",
      caption  = sprintf("run = %s", run_id)
    )
  ggsave(file.path(out, "05_feature_counts.png"), p5,
         width = 6, height = 4)

  # 6. Feature sparsity — fraction of zeros per feature  (NEW)
  # Raw feature means are heavy-tailed with almost all mass near zero, and a
  activity_pc <- {
    pos <- fs$mean[fs$mean > 0]
    if (length(pos) > 0) max(min(pos, na.rm = TRUE) / 10, 1e-6) else 1e-6
  }
  fs_activity <- fs %>% mutate(mean_log = log10(mean + activity_pc))

  p6 <- ggplot(fs_activity, aes(x = source, y = mean_log, fill = source)) +
    geom_violin(alpha = 0.6, color = NA, scale = "width", trim = FALSE) +
    geom_boxplot(width = 0.12, alpha = 0.9, outlier.size = 0.4,
                 outlier.alpha = 0.3, color = "grey20") +
    scale_fill_manual(values = col_source, guide = "none") +
    labs(
      title    = "Per-feature mean activity by source",
      x = NULL, y = sprintf("log10(feature mean + %.2g)", activity_pc),
      caption  = sprintf("run = %s", run_id)
    )
  ggsave(file.path(out, "06_feature_activity.png"), p6,
         width = 6, height = 4)

  # 6b. Top protein clusters by raw total abundance
  if (all(c("feature_name", "total_abundance") %in% names(fs))) {
    top_pc <- fs %>%
      group_by(source) %>%
      slice_max(order_by = total_abundance, n = 25, with_ties = FALSE) %>%
      ungroup() %>%
      mutate(
        short_name = ifelse(
          nchar(feature_name) > 38,
          paste0(substr(feature_name, 1, 35), "..."),
          feature_name
        ),
        label_raw = paste(source, short_name, sep = ": "),
        label = factor(
          make.unique(label_raw),
          levels = rev(make.unique(label_raw))
        )
      )
    p6b <- ggplot(top_pc, aes(x = total_abundance, y = label, fill = source)) +
      geom_col(width = 0.75) +
      scale_fill_manual(values = col_source, name = "source") +
      scale_x_continuous(trans = scales::pseudo_log_trans(base = 10),
                         labels = scales::label_comma()) +
      labs(
        title = "Top protein clusters by total raw abundance",
        x = "total raw abundance/count across entities (pseudo-log)",
        y = NULL,
        caption = sprintf("run = %s", run_id)
      )
    ggsave(file.path(out, "06b_top_protein_clusters.png"), p6b,
           width = 9, height = 8)
  }

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
      x = NULL, y = "count (observed rows)",
      caption  = sprintf("run = %s", run_id)
    )
  ggsave(file.path(out, "07_label_balance.png"), p7,
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
        x = "W value", y = "count",
        caption = sprintf("run = %s", run_id)
      )
    ggsave(file.path(out, "08_W_distribution.png"), p8,
           width = 5.0, height = 3.2)
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
      x = "outer fold", y = "P(y = 1)",
      caption  = sprintf("run = %s", run_id)
    )
  ggsave(file.path(out, "09_split_balance.png"), p9,
         width = 8, height = 4)

  cat(sprintf("[plot_data] wrote %d plots to %s\n",
              length(list.files(out, pattern = "\\.png$")), out))
})

# 2. Interaction matrices and nestedness
# Reads graph_matrices/*.csv written by graph_export.py.
local({
  gm     <- graph_matrices_dir(run_id)
  out    <- plots_data_dir(run_id)

  # load exported matrices
  need <- file.path(gm, c("Y_adjacency.csv", "W_adjacency.csv", "W_mask_adjacency.csv"))
  if (!all(file.exists(need))) {
    stop("Missing exported matrices in ", gm,
         "\nRun graph_export.py first.")
  }
  Y <- as.matrix(read.csv(need[1], header = FALSE))
  W <- as.matrix(read.csv(need[2], header = FALSE))
  M <- as.matrix(read.csv(need[3], header = FALSE))   # 1 = W observed
  storage.mode(Y) <- "numeric"; storage.mode(W) <- "numeric"; storage.mode(M) <- "numeric"
  message(sprintf("[plot_data:interactions] %d bacteria x %d viruses", nrow(Y), ncol(Y)))

  # Theme: shared base, minus the grid and axis text (these are heatmaps).
  theme_set(pub_theme +
    theme(panel.grid = element_blank(), axis.text = element_blank(),
          axis.ticks = element_blank()))

  # Ordering helpers
  # Cluster ONCE on the (observed) W matrix; reuse the order for all three heatmaps
  # so cells are comparable across Y, W and the combined view.
  cluster_order <- function(mat) {
    ord_dim <- function(m) {
      if (nrow(m) < 3) return(seq_len(nrow(m)))
      d <- tryCatch(dist(m), error = function(e) NULL)
      if (is.null(d) || any(!is.finite(d))) return(seq_len(nrow(m)))
      hc <- tryCatch(hclust(d, method = "ward.D2"), error = function(e) NULL)
      if (is.null(hc)) seq_len(nrow(m)) else hc$order
    }
    Wc <- mat; Wc[!is.finite(Wc)] <- 0
    list(rows = ord_dim(Wc), cols = ord_dim(t(Wc)))
  }

  # Degree ordering for nestedness: most-connected first on both axes.
  degree_order <- function(mat) {
    list(rows = order(rowSums(mat), decreasing = TRUE),
         cols = order(colSums(mat), decreasing = TRUE))
  }

  to_long <- function(mat, row_ord, col_ord, value_name = "value") {
    m <- mat[row_ord, col_ord, drop = FALSE]
    df <- expand.grid(row = seq_len(nrow(m)), col = seq_len(ncol(m)))
    df[[value_name]] <- as.vector(m)   # column-major: matches expand.grid (row varies fastest)
    df
  }

  # Pure-R NODF nestedness (Almeida-Neto et al. 2008), binary matrix.
  # Returns NODF in [0, 100]. No external packages so the pipeline always runs.
  nodf <- function(binmat) {
    A <- (binmat > 0) * 1
    pair_sum <- function(M, marg) {
      n <- nrow(M)
      if (n < 2) return(list(s = 0, np = 0))
      O     <- M %*% t(M)
      gt    <- outer(marg, marg, ">")               # [i,j] TRUE if marg[i] > marg[j]
      denom <- matrix(marg, n, n, byrow = TRUE)      # [i,j] = marg[j]
      contrib <- ifelse(gt & denom > 0, 100 * O / denom, 0)
      list(s = sum(contrib), np = choose(n, 2))
    }
    rp <- pair_sum(A,    rowSums(A))
    cp <- pair_sum(t(A), colSums(A))
    denom <- rp$np + cp$np
    if (denom == 0) return(NA_real_)
    (rp$s + cp$s) / denom
  }

  # 1. CRISPR (Y) interaction matrix — binary, clustered on W
  ord <- cluster_order(W * M)                       # cluster on observed W
  dfY <- to_long(Y, ord$rows, ord$cols, "y")
  dfY$edge <- factor(ifelse(dfY$y >= 1, "edge", "none"), levels = c("none", "edge"))
  n_edge <- sum(Y >= 1); dens <- 100 * n_edge / length(Y)

  pY <- ggplot(dfY, aes(x = col, y = row, fill = edge)) +
    geom_raster() +
    scale_y_reverse(expand = c(0, 0)) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_fill_manual(values = c(none = "black", edge = "#E8542F"), name = NULL) +
    labs(title = sprintf("CRISPR interaction matrix  (edges = %d, density = %.2f%%)",
                         n_edge, dens),
         x = "viruses", y = "bacteria")
  ggsave(file.path(out, "interaction_Y_matrix.png"), pY, width = 5.4, height = 4.4, dpi = 150)
  message("[plot_data:interactions] wrote interaction_Y_matrix.png")

  # 2. glasso (W) interaction matrix — intensity, unobserved = grey
  Wm <- W; Wm[M != 1] <- NA                         # NA where unobserved -> grey
  dfW <- to_long(Wm, ord$rows, ord$cols, "w")
  n_obs <- sum(M == 1); w_mean <- mean(W[M == 1])

  pW <- ggplot(dfW, aes(x = col, y = row, fill = w)) +
    geom_raster() +
    scale_y_reverse(expand = c(0, 0)) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_fill_gradient(low = "#132B43", high = "#F5C518", na.value = "grey55",
                        name = "W", limits = c(0, 1)) +
    labs(title = sprintf("glasso interaction matrix  (observed = %d, mean W = %.3f; grey = unobserved)",
                         n_obs, w_mean),
         x = "viruses", y = "bacteria")
  ggsave(file.path(out, "interaction_W_matrix.png"), pW, width = 5.4, height = 4.4, dpi = 150)
  message("[plot_data:interactions] wrote interaction_W_matrix.png")

  # 3. Combined agreement matrix — both / CRISPR only / glasso only
 
  w_thr <- local({
    cfg_path <- run_path(run_id, "config.yaml")
    if (file.exists(cfg_path) && requireNamespace("yaml", quietly = TRUE)) {
      cfg <- try(yaml::read_yaml(cfg_path), silent = TRUE)
      if (!inherits(cfg, "try-error")) {
        for (tk in cfg$tasks) {
          if (!is.null(tk$name) && tk$name == "w_class" && !is.null(tk$binarize_threshold)) {
            return(as.numeric(tk$binarize_threshold))
          }
        }
      }
    }
    # No yaml package (or no frozen config): fall back to a plain text scan, then 0.
    if (file.exists(cfg_path)) {
      txt <- readLines(cfg_path, warn = FALSE)
      hit <- grep("binarize_threshold", txt, value = TRUE)
      if (length(hit)) {
        v <- suppressWarnings(as.numeric(sub(".*binarize_threshold:\\s*([0-9.eE+-]+).*", "\\1",
                                             hit[length(hit)])))
        if (!is.na(v)) return(v)
      }
    }
    message("[plot_data:interactions] could not read w_class threshold from config; using 0.")
    0
  })
  message(sprintf("[plot_data:interactions] glasso edge defined as W > %g (from config)", w_thr))
  crispr <- Y >= 1
  glasso <- (M == 1) & (W > w_thr)
  cat_mat <- matrix("none", nrow = nrow(Y), ncol = ncol(Y))
  cat_mat[crispr & glasso]  <- "both"
  cat_mat[crispr & !glasso] <- "CRISPR only"
  cat_mat[!crispr & glasso] <- "glasso only"
  dfC <- to_long(cat_mat, ord$rows, ord$cols, "cat")
  dfC$cat <- factor(dfC$cat, levels = c("none", "glasso only", "CRISPR only", "both"))
  n_both <- sum(cat_mat == "both")
  n_c    <- sum(cat_mat %in% c("both", "CRISPR only"))
  n_g    <- sum(cat_mat %in% c("both", "glasso only"))

  pC <- ggplot(dfC, aes(x = col, y = row, fill = cat)) +
    geom_raster() +
    scale_y_reverse(expand = c(0, 0)) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_fill_manual(values = c(none = "black", `glasso only` = "#1F78B4",
                                 `CRISPR only` = "#E31A1C", both = "#6A3D9A"),
                      name = NULL) +
    labs(title = sprintf("CRISPR vs glasso (W>%.1f)  (CRISPR = %d, glasso = %d, overlap = %d)",
                         w_thr, n_c, n_g, n_both),
         x = "viruses", y = "bacteria")
  ggsave(file.path(out, "interaction_combined_matrix.png"), pC, width = 5.4, height = 4.4, dpi = 150)
  message("[plot_data:interactions] wrote interaction_combined_matrix.png")

  # 4. Nestedness — CRISPR (binary), degree-sorted
  Yb <- (Y >= 1) * 1
  ordY <- degree_order(Yb)
  dfNY <- to_long(Yb, ordY$rows, ordY$cols, "y")
  dfNY$edge <- factor(ifelse(dfNY$y >= 1, "edge", "none"), levels = c("none", "edge"))
  nodf_Y <- nodf(Yb)

  pNY <- ggplot(dfNY, aes(x = col, y = row, fill = edge)) +
    geom_raster() +
    scale_y_reverse(expand = c(0, 0)) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_fill_manual(values = c(none = "black", edge = "#E8542F"), name = NULL) +
    labs(title = sprintf("CRISPR nestedness  (NODF = %.1f)", nodf_Y),
         x = "viruses (sorted by degree)", y = "bacteria (sorted by degree)")
  ggsave(file.path(out, "nestedness_Y.png"), pNY, width = 5.4, height = 4.4, dpi = 150)
  message("[plot_data:interactions] wrote nestedness_Y.png")

  # 5. Nestedness — glasso (intensity), degree-sorted on presence
  # Presence for structure = observed edge with W > 0; shade by W intensity.
  # Restrict to the W-OBSERVED block first (drop bacteria/viruses with no W data):
  # unobserved rows/cols are not degree-0 species, they are unmeasured, and
  # including them as empty padding distorts NODF. This matches the intersection
  # heatmaps (no grey / no unobserved region).
  obs_r <- which(rowSums(M == 1) > 0)
  obs_c <- which(colSums(M == 1) > 0)
  Wob <- W[obs_r, obs_c, drop = FALSE]
  Mob <- M[obs_r, obs_c, drop = FALSE]
  Wpres <- ((Mob == 1) & (Wob > 0)) * 1
  Wval  <- Wob; Wval[!(Mob == 1 & Wob > 0)] <- NA
  ordW  <- degree_order(Wpres)
  dfNW  <- to_long(Wval, ordW$rows, ordW$cols, "w")
  nodf_W <- nodf(Wpres)

  pNW <- ggplot(dfNW, aes(x = col, y = row, fill = w)) +
    geom_raster() +
    scale_y_reverse(expand = c(0, 0)) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_fill_gradient(low = "#132B43", high = "#F5C518", na.value = "black",
                        name = "W", limits = c(0, 1)) +
    labs(title = sprintf("glasso nestedness — observed block  (NODF = %.1f, presence = W>0)", nodf_W),
         x = "viruses (sorted by degree)", y = "bacteria (sorted by degree)")
  ggsave(file.path(out, "nestedness_W.png"), pNW, width = 5.4, height = 4.4, dpi = 150)
  message("[plot_data:interactions] wrote nestedness_W.png")

  # 6. Intersection-only heatmaps — W and Y on the W-observed block (no grey)
  obs_rows <- which(rowSums(M == 1) > 0)
  obs_cols <- which(colSums(M == 1) > 0)
  if (length(obs_rows) >= 2 && length(obs_cols) >= 2) {
    Wsub <- W[obs_rows, obs_cols, drop = FALSE]
    Ysub <- Y[obs_rows, obs_cols, drop = FALSE]
    ordW <- degree_order(Wsub)          # W degree = continuous row/col sums
    ordY <- degree_order(Ysub >= 1)     # CRISPR degree = number of edges

    # 6a. glasso W intensity on the intersection, degree-sorted (no grey)
    dfWs <- to_long(Wsub, ordW$rows, ordW$cols, "w")
    pWs <- ggplot(dfWs, aes(x = col, y = row, fill = w)) +
      geom_raster() +
      scale_y_reverse(expand = c(0, 0)) +
      scale_x_continuous(expand = c(0, 0)) +
      scale_fill_gradient(low = "#132B43", high = "#F5C518",
                          name = "W", limits = c(0, 1)) +
      labs(title = sprintf("glasso W — observed block  (mean W = %.3f)", mean(Wsub)),
           x = sprintf("viruses   (m = %d)", ncol(Wsub)),
           y = sprintf("bacteria   (n = %d)", nrow(Wsub))) +
      theme(axis.title = element_text(size = 20, face = "bold"))
    ggsave(file.path(out, "interaction_W_matrix_observed.png"), pWs, width = 5.4, height = 4.4, dpi = 150)
    message("[plot_data:interactions] wrote interaction_W_matrix_observed.png")

    # 6b. CRISPR Y on the intersection, degree-sorted by its own connectivity
    dfYs <- to_long(Ysub, ordY$rows, ordY$cols, "y")
    dfYs$edge <- factor(ifelse(dfYs$y >= 1, "edge", "none"), levels = c("none", "edge"))
    n_edge_s <- sum(Ysub >= 1); dens_s <- 100 * n_edge_s / length(Ysub)
    pYs <- ggplot(dfYs, aes(x = col, y = row, fill = edge)) +
      geom_raster() +
      scale_y_reverse(expand = c(0, 0)) +
      scale_x_continuous(expand = c(0, 0)) +
      scale_fill_manual(values = c(none = "black", edge = "#E8542F"), name = NULL) +
      labs(title = sprintf("CRISPR Y — observed block  (edges = %d, density = %.2f%%)",
                           n_edge_s, dens_s),
           x = sprintf("viruses   (m = %d)", ncol(Ysub)),
           y = sprintf("bacteria   (n = %d)", nrow(Ysub))) +
      theme(axis.title = element_text(size = 20, face = "bold"))
    ggsave(file.path(out, "interaction_Y_matrix_observed.png"), pYs, width = 5.4, height = 4.4, dpi = 150)
    message("[plot_data:interactions] wrote interaction_Y_matrix_observed.png")
  } else {
    message("[plot_data:interactions] not enough W-observed rows/cols for intersection plots; skipped.")
  }

  cat(sprintf("[plot_data:interactions] wrote interaction/nestedness plots to %s\n", out))
})

message("[plot_data] done.")
