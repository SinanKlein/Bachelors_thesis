# data.R  —  descriptive plots of the inputs -> plots/data/
#   1. features, labels, splits   preprocessing/*.csv           (data.py preprocess)
#   2. interaction matrices       graph_matrices/*.csv          (data.py describe)
#   3. protein-cluster presence   graph_matrices/X*_presence.csv
source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE),
       value = TRUE)[1])), "common.R"))

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
  library(scales)
  library(tidyr)
})

run_id <- get_run_id()
message(sprintf("[plot_data] dataset=%s run=%s", DATASET_NAME, run_id))

# 1. Features, labels and splits
local({
  out <- out_dir(run_id, "plots", "data")
  pre <- list(
    feature_stats = read_run_csv(run_id, "preprocessing", "feature_stats.csv"),
    labels        = read_run_csv(run_id, "preprocessing", "labels.csv"),
    splits        = read_run_csv(run_id, "preprocessing", "splits.csv"),
    summary       = jsonlite::fromJSON(run_path(run_id, "preprocessing", "preprocess_log.json"))
  )
  theme_set(pub_theme)

  col_kept    <- "#D62728"
  col_dropped <- "#BDBDBD"
  col_source  <- c(Xb = "#1F77B4", Xv = "#FF7F0E")

  # Mean-variance scatter with marginal histograms, one source.
  plot_mean_var_single <- function(df_one, accent_color) {
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

    p_top <- ggplot(df_one, aes(x = mean)) +
      geom_histogram(bins = 50, fill = accent_color, alpha = 0.7, color = NA) +
      scale_x_continuous(trans = scales::pseudo_log_trans(base = 10)) +
      theme_void() +
      theme(plot.margin = margin(0, 0, 0, 0))

    p_right <- ggplot(df_one, aes(x = var)) +
      geom_histogram(bins = 50, fill = accent_color, alpha = 0.7, color = NA) +
      scale_x_continuous(trans = scales::pseudo_log_trans(base = 10)) +
      coord_flip() +
      theme_void() +
      theme(plot.margin = margin(0, 0, 0, 0))

    p_top + plot_spacer() +
                 p_main + p_right +
      plot_layout(
        ncol = 2, nrow = 2,
        widths  = c(4, 1),
        heights = c(1, 4)
      ) +
      plot_annotation(theme = pub_theme)
  }

  # 01-02. Per-source mean-variance
  fs <- pre$feature_stats
  fs$selected <- as.logical(fs$selected)

  sources <- unique(fs$source)
  for (s in sources) {
    p <- plot_mean_var_single(fs %>% filter(source == s), col_source[[s]] %||% col_kept)
    ggsave(file.path(out, sprintf("0%d_mean_variance_%s.png",
                                  which(sources == s), s)),
           p, width = 7, height = 6.5)
  }

  # 03. Combined mean-variance
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
    labs(x = "mean (log)", y = "variance (log)"
    )
  ggsave(file.path(out, "03_mean_variance_combined.png"),
         p3, width = 9, height = 4.5)

  # 04. Score distribution
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
    labs(x = "raw-variance score", y = "feature count"
    )
  ggsave(file.path(out, "04_score_distribution.png"), p4,
         width = 9, height = 4)


  # 05. Feature counts
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
    labs(x = NULL, y = "feature count"
    )
  ggsave(file.path(out, "05_feature_counts.png"), p5,
         width = 6, height = 4)

  # 06. Feature activity (log mean, pseudocount = smallest positive mean / 10)
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
    labs(x = NULL, y = sprintf("log10(feature mean + %.2g)", activity_pc)
    )
  ggsave(file.path(out, "06_feature_activity.png"), p6,
         width = 6, height = 4)

  # 06b. Top protein clusters by total abundance
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
      labs(x = "total raw abundance/count across entities (pseudo-log)",
        y = NULL
      )
    ggsave(file.path(out, "06b_top_protein_clusters.png"), p6b,
           width = 9, height = 8)
  }

  # 07. Label balance of binary tasks (observed rows)
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

  p7 <- ggplot(lb_long, aes(x = response_label(task), y = n, fill = class)) +
    geom_col(width = 0.6) +
    geom_text(data = lb_bin,
              aes(x = response_label(task), y = n_total,
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
    labs(x = NULL, y = "count (observed rows)"
    )
  ggsave(file.path(out, "07_label_balance.png"), p7,
         width = 6.5, height = 4.5)

  # 08. W distribution
  reg_labels <- pre$labels %>% filter(task_type == "regression", mask == 1)
  if (nrow(reg_labels) > 0) {
    p8 <- ggplot(reg_labels, aes(x = y)) +
      geom_histogram(bins = 50, fill = col_source[["Xb"]],
                     alpha = 0.8, color = NA) +
      geom_vline(xintercept = 0, linetype = "dashed", color = "grey30") +
      labs(x = "edge selection probability", y = "count"
      )
    ggsave(file.path(out, "08_W_distribution.png"), p8,
           width = 6.5, height = 4)
  }

  # 09. Positive fraction per fold and role
  prim_task <- pre$summary$tasks$name[1]
  y_prim <- pre$labels %>% filter(task == prim_task) %>%
    select(sample_id, y)

  bal <- pre$splits %>%
    inner_join(y_prim, by = "sample_id") %>%
    group_by(outer_fold, role) %>%
    summarise(pos_frac = mean(y == 1), n = n(), .groups = "drop")

  baseline <- mean(y_prim$y == 1)

  p9 <- ggplot(bal, aes(x = role, y = pos_frac, fill = role)) +
    geom_boxplot(width = 0.5, alpha = 0.85, outlier.shape = NA,
                 colour = "grey25", linewidth = 0.3) +
    geom_point(position = position_jitter(width = 0.08, height = 0, seed = 1),
               size = 1.6, colour = "grey20", alpha = 0.8) +
    geom_hline(yintercept = baseline, linetype = "dashed",
               color = "grey30", linewidth = 0.4) +
    annotate("text", x = 0.7, y = baseline,
             label = sprintf("overall = %.2f%%", 100 * baseline),
             vjust = -0.5, hjust = 0, size = 3, color = "grey30") +
    scale_fill_manual(values = c(train = "grey60", test = col_kept),
                      name = "role") +
    scale_y_continuous(labels = scales::label_percent(accuracy = 0.1),
                       expand = expansion(mult = c(0, 0.15))) +
    labs(x = NULL, y = sprintf("P(%s = 1)", response_label(prim_task))
    )
  ggsave(file.path(out, "09_split_balance.png"), p9,
         width = 8, height = 4)

  cat(sprintf("[plot_data] wrote %d plots to %s\n",
              length(list.files(out, pattern = "\\.png$")), out))
})

# 2. Interaction matrices and nestedness
local({
  gm  <- run_path(run_id, "graph_matrices")
  out <- out_dir(run_id, "plots", "data")
  need <- file.path(gm, c("Y_adjacency.csv", "W_adjacency.csv", "W_mask_adjacency.csv"))
  if (!all(file.exists(need))) stop("Missing matrices in ", gm, "; run data.py describe first.")
  Y <- as.matrix(read.csv(need[1], header = FALSE))
  W <- as.matrix(read.csv(need[2], header = FALSE))
  M <- as.matrix(read.csv(need[3], header = FALSE))   # 1 = W observed
  storage.mode(Y) <- "numeric"; storage.mode(W) <- "numeric"; storage.mode(M) <- "numeric"
  message(sprintf("[plot_data:interactions] %d bacteria x %d viruses", nrow(Y), ncol(Y)))

  # Heatmap theme: no grid, no axis text.
  theme_set(pub_theme +
    theme(panel.grid = element_blank(), axis.text = element_blank(),
          axis.ticks = element_blank()))

  # Ward order of rows and columns.
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

  # Most-connected first on both axes.
  degree_order <- function(mat) {
    list(rows = order(rowSums(mat), decreasing = TRUE),
         cols = order(colSums(mat), decreasing = TRUE))
  }

  to_long <- function(mat, row_ord, col_ord, value_name = "value") {
    m <- mat[row_ord, col_ord, drop = FALSE]
    df <- expand.grid(row = seq_len(nrow(m)), col = seq_len(ncol(m)))
    df[[value_name]] <- as.vector(m)   # column-major, matches expand.grid
    df
  }

  # NODF nestedness (Almeida-Neto et al. 2008) of a binary matrix, in [0, 100].
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

  # Y matrix, ordered by clustering observed W
  ord <- cluster_order(W * M)
  dfY <- to_long(Y, ord$rows, ord$cols, "y")
  dfY$edge <- factor(ifelse(dfY$y >= 1, "edge", "none"), levels = c("none", "edge"))

  pY <- ggplot(dfY, aes(x = col, y = row, fill = edge)) +
    geom_raster() +
    scale_y_reverse(expand = c(0, 0)) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_fill_manual(values = c(none = "black", edge = "#E8542F"), name = NULL) +
    labs(x = sprintf("viruses   (m = %d, clustered on W)", ncol(Y)),
         y = sprintf("bacteria   (n = %d, clustered on W)", nrow(Y)))
  ggsave(file.path(out, "interaction_Y_matrix.png"), pY, width = 7.5, height = 6, dpi = 150)
  message("[plot_data:interactions] wrote interaction_Y_matrix.png")

  # W matrix, unobserved = grey
  Wm <- W; Wm[M != 1] <- NA
  dfW <- to_long(Wm, ord$rows, ord$cols, "w")

  pW <- ggplot(dfW, aes(x = col, y = row, fill = w)) +
    geom_raster() +
    scale_y_reverse(expand = c(0, 0)) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_fill_gradient(low = "#132B43", high = "#F5C518", na.value = "grey55",
                        name = "edge selection\nprobability", limits = c(0, 1)) +
    labs(x = sprintf("viruses   (m = %d, clustered on W)", ncol(Wm)),
         y = sprintf("bacteria   (n = %d, clustered on W)", nrow(Wm)))
  ggsave(file.path(out, "interaction_W_matrix.png"), pW, width = 7.5, height = 6, dpi = 150)
  message("[plot_data:interactions] wrote interaction_W_matrix.png")

  # Combined view on the W-observed block: W as colour, CRISPR edges outlined
  keep_r <- which(rowSums(M == 1) > 0)
  keep_c <- which(colSums(M == 1) > 0)
  Wo <- W[keep_r, keep_c, drop = FALSE]
  Yo <- (Y[keep_r, keep_c, drop = FALSE] >= 1) * 1
  n_o <- nrow(Wo); m_o <- ncol(Wo)

  ordO <- cluster_order(Wo)
  dfWo <- to_long(Wo, ordO$rows, ordO$cols, "w")
  dfYo <- to_long(Yo, ordO$rows, ordO$cols, "y")
  dfYo <- dfYo[dfYo$y == 1, , drop = FALSE]

  message(sprintf("[plot_data:interactions] observed block %d x %d = %d pairs; %d CRISPR edges",
                  n_o, m_o, n_o * m_o, nrow(dfYo)))

  pC <- ggplot() +
    geom_raster(data = dfWo, aes(x = col, y = row, fill = w)) +
    geom_tile(data = dfYo, aes(x = col, y = row),
              fill = NA, colour = "#EB6834", linewidth = 0.22) +
    scale_y_reverse(expand = c(0, 0)) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_fill_gradientn(
      colours = c("#F0EFEC", "#C3D9F3", "#7FB0E6", "#2A78D6", "#143C6B"),
      limits = c(0, 1), name = "edge selection\nprobability") +
    labs(x = sprintf("vOTUs   (m = %d, clustered on W)", m_o),
         y = sprintf("bacterial genera   (n = %d, clustered on W)", n_o),
         subtitle = "orange outline: CRISPR edge")
  ggsave(file.path(out, "interaction_combined_matrix.png"), pC, width = 7.5, height = 6, dpi = 150)
  message("[plot_data:interactions] wrote interaction_combined_matrix.png")

  # Nestedness of Y, degree-sorted
  Yb <- (Y >= 1) * 1
  ordY <- degree_order(Yb)
  dfNY <- to_long(Yb, ordY$rows, ordY$cols, "y")
  dfNY$edge <- factor(ifelse(dfNY$y >= 1, "edge", "none"), levels = c("none", "edge"))
  nodf_Y <- nodf(Yb)
  message(sprintf("[plot_data] CRISPR linkage: edges = %d, density = %.2f%%, NODF = %.1f",
                  sum(Yb), 100 * mean(Yb), nodf_Y))

  pNY <- ggplot(dfNY, aes(x = col, y = row, fill = edge)) +
    geom_raster() +
    scale_y_reverse(expand = c(0, 0)) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_fill_manual(values = c(none = "black", edge = "#E8542F"), name = NULL) +
    labs(x = sprintf("viruses   (m = %d, sorted by degree)", ncol(Yb)),
         y = sprintf("bacteria   (n = %d, sorted by degree)", nrow(Yb)))
  ggsave(file.path(out, "nestedness_Y.png"), pNY, width = 7.5, height = 6, dpi = 150)
  message("[plot_data:interactions] wrote nestedness_Y.png")

  # Nestedness of W > 0 on the W-observed block, shaded by W
  obs_r <- which(rowSums(M == 1) > 0)
  obs_c <- which(colSums(M == 1) > 0)
  Wob <- W[obs_r, obs_c, drop = FALSE]
  Mob <- M[obs_r, obs_c, drop = FALSE]
  Wpres <- ((Mob == 1) & (Wob > 0)) * 1
  Wval  <- Wob; Wval[!(Mob == 1 & Wob > 0)] <- NA
  ordW  <- degree_order(Wpres)
  dfNW  <- to_long(Wval, ordW$rows, ordW$cols, "w")
  nodf_W <- nodf(Wpres)
  message(sprintf("[plot_data] glasso edge, observed block: edges = %d, density = %.2f%%, mean prob = %.3f, NODF = %.1f",
                  sum(Wpres), 100 * mean(Wpres), mean(Wob[Mob == 1]), nodf_W))

  pNW <- ggplot(dfNW, aes(x = col, y = row, fill = w)) +
    geom_raster() +
    scale_y_reverse(expand = c(0, 0)) +
    scale_x_continuous(expand = c(0, 0)) +
    scale_fill_gradient(low = "#132B43", high = "#F5C518", na.value = "black",
                        name = "edge selection\nprobability", limits = c(0, 1)) +
    labs(x = sprintf("viruses   (m = %d, sorted by degree)", ncol(Wpres)),
         y = sprintf("bacteria   (n = %d, sorted by degree)", nrow(Wpres)))
  ggsave(file.path(out, "nestedness_W.png"), pNW, width = 7.5, height = 6, dpi = 150)
  message("[plot_data:interactions] wrote nestedness_W.png")

  # W and Y on the W-observed block, degree-sorted
  obs_rows <- which(rowSums(M == 1) > 0)
  obs_cols <- which(colSums(M == 1) > 0)
  if (length(obs_rows) >= 2 && length(obs_cols) >= 2) {
    Wsub <- W[obs_rows, obs_cols, drop = FALSE]
    Ysub <- Y[obs_rows, obs_cols, drop = FALSE]
    ordW <- degree_order(Wsub)
    ordY <- degree_order(Ysub >= 1)
    dfWs <- to_long(Wsub, ordW$rows, ordW$cols, "w")
    pWs <- ggplot(dfWs, aes(x = col, y = row, fill = w)) +
      geom_raster() +
      scale_y_reverse(expand = c(0, 0)) +
      scale_x_continuous(expand = c(0, 0)) +
      scale_fill_gradient(low = "#132B43", high = "#F5C518",
                          name = "edge selection\nprobability", limits = c(0, 1)) +
      labs(x = sprintf("viruses   (m = %d, sorted by degree)", ncol(Wsub)),
           y = sprintf("bacteria   (n = %d, sorted by degree)", nrow(Wsub))) +
      theme(axis.title = element_text(size = 20, face = "bold"))
    ggsave(file.path(out, "interaction_W_matrix_observed.png"), pWs, width = 7.5, height = 6, dpi = 150)
    message("[plot_data:interactions] wrote interaction_W_matrix_observed.png")

    dfYs <- to_long(Ysub, ordY$rows, ordY$cols, "y")
    dfYs$edge <- factor(ifelse(dfYs$y >= 1, "edge", "none"), levels = c("none", "edge"))
    pYs <- ggplot(dfYs, aes(x = col, y = row, fill = edge)) +
      geom_raster() +
      scale_y_reverse(expand = c(0, 0)) +
      scale_x_continuous(expand = c(0, 0)) +
      scale_fill_manual(values = c(none = "black", edge = "#E8542F"), name = NULL) +
      labs(x = sprintf("viruses   (m = %d, sorted by degree)", ncol(Ysub)),
           y = sprintf("bacteria   (n = %d, sorted by degree)", nrow(Ysub))) +
      theme(axis.title = element_text(size = 20, face = "bold"))
    ggsave(file.path(out, "interaction_Y_matrix_observed.png"), pYs, width = 7.5, height = 6, dpi = 150)
    message("[plot_data:interactions] wrote interaction_Y_matrix_observed.png")
  } else {
    message("[plot_data:interactions] not enough W-observed rows/cols for intersection plots; skipped.")
  }

  cat(sprintf("[plot_data:interactions] wrote interaction/nestedness plots to %s\n", out))
})

# 3. Protein-cluster presence, degree-sorted
local({
  out <- out_dir(run_id, "plots", "data")
  fb  <- run_path(run_id, "graph_matrices", "Xb_presence.csv")
  fv  <- run_path(run_id, "graph_matrices", "Xv_presence.csv")

  if (!file.exists(fb) || !file.exists(fv)) {
    message("[plot_data:procs] presence CSVs not found; skipped")
    return(invisible(NULL))
  }

  theme_set(pub_theme)

  proc_panel <- function(path, entity_lab) {
    M <- as.matrix(read.csv(path, header = FALSE))
    storage.mode(M) <- "numeric"
    ro <- order(rowSums(M), decreasing = TRUE)
    co <- order(colSums(M), decreasing = TRUE)
    m  <- M[ro, co, drop = FALSE]
    df <- expand.grid(row = seq_len(nrow(m)), col = seq_len(ncol(m)))
    df$present <- factor(ifelse(as.vector(m) > 0, "present", "absent"),
                         levels = c("absent", "present"))
    ggplot(df, aes(x = col, y = row, fill = present)) +
      geom_raster() +
      scale_y_reverse(expand = c(0, 0)) +
      scale_x_continuous(expand = c(0, 0)) +
      scale_fill_manual(values = c(absent = "black", present = "#E8542F"), name = NULL) +
      labs(x = "protein clusters (sorted by degree)",
           y = sprintf("%s (sorted by degree)", entity_lab))
  }

  p <- proc_panel(fb, "genera") / proc_panel(fv, "vOTUs") +
       plot_layout(guides = "collect")

  ggsave(file.path(out, "10_proc_sparsity.png"), p, width = 10, height = 7, dpi = 300)
  message("[plot_data:procs] wrote 10_proc_sparsity.png")
})

message("[plot_data] done.")
