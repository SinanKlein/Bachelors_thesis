# MLP Experiment plots

library(dplyr)
library(tidyr)
library(ggplot2)
library(janitor)
library(viridis)

out_dir <- "C:/Sinan_Klein/LMU/lmu_thesis/multimodal_network/outputs_vib_dual/plots"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

data_dir <- "C:/Sinan_Klein/LMU/lmu_thesis/multimodal_network/outputs_vib_dual"

# Load Data
df_lat <- read.csv(file.path(data_dir, "metrics_all_latdims.csv")) %>% clean_names()
df_trace <- read.csv(file.path(data_dir, "info_theory_trace_latdims.csv")) %>% clean_names()

# Set Theme
theme_pub <- theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 16),
    plot.subtitle = element_text(hjust = 0.5, color = "gray30"),
    legend.position = "bottom",
    legend.title = element_text(face = "bold"),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "gray80", fill = NA, size = 0.5),
    strip.background = element_rect(fill = "gray90", color = NA),
    strip.text = element_text(face = "bold")
  )

# Set a color palette
color_palette <- scale_color_viridis_d(option = "plasma", end = 0.8)
fill_palette <- scale_fill_viridis_d(option = "plasma", end = 0.8)

# Calculate average warmup boundary for plotting vertical lines
warmup_boundary <- df_trace %>%
  filter(phase == "warmup") %>%
  group_by(latent_dim, split_idx) %>%
  summarise(max_warmup_epoch = max(epoch), .groups = 'drop') %>%
  summarise(mean_warmup = mean(max_warmup_epoch)) %>%
  pull(mean_warmup)

# 1. Performance Metrics vs Latent Dimensions

# Reshape for easier plotting
df_lat_long <- df_lat %>%
  pivot_longer(
    cols = starts_with("auc_y_") | starts_with("auc_w0_"),
    names_to = c("metric", "model"),
    names_pattern = "(auc_y|auc_w0)_(nn|vib|lr)",
    values_drop_na = TRUE
  ) %>%
  mutate(
    model = toupper(model),
    metric = ifelse(metric == "auc_y", "Task Performance (AUC Y)", "Co abundance Prediction (AUC W0)")
  ) %>%
  group_by(latent_dim, metric, model) %>%
  summarise(
    mean_val = mean(value, na.rm = TRUE),
    sd_val = sd(value, na.rm = TRUE),
    .groups = 'drop'
  )

p_perf_lat <- ggplot(df_lat_long, aes(x = as.factor(latent_dim), y = mean_val, color = model, group = model)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = mean_val - sd_val, ymax = mean_val + sd_val), width = 0.2, alpha = 0.7) +
  facet_wrap(~metric, scales = "free_y") +
  labs(title = "Performance vs Latent Dimension", x = "Latent Dimension", y = "AUC") +
  theme_pub + color_palette

ggsave(file.path(out_dir, "performance_vs_latent_dim.png"), p_perf_lat, width = 10, height = 6, dpi = 300)

# 2. S-Task Performance vs Latent Dimension

df_s_task <- df_lat %>%
  pivot_longer(
    cols = c("auc_spos_nn", "auc_sneg_nn", "auc_spos_vib", "auc_sneg_vib"),
    names_to = c("task", "model"),
    names_pattern = "auc_s(pos|neg)_(nn|vib)"
  ) %>%
  mutate(model = toupper(model), task = paste("S-Task:", toupper(task))) %>%
  group_by(latent_dim, task, model) %>%
  summarise(mean_val = mean(value), sd_val = sd(value), .groups = 'drop')

p_stask <- ggplot(df_s_task, aes(x = as.factor(latent_dim), y = mean_val, color = model, group = model)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = mean_val - sd_val, ymax = mean_val + sd_val), width = 0.2, alpha = 0.7) +
  facet_wrap(~task) +
  labs(title = "S-Task Performance vs Latent Dimension", x = "Latent Dimension", y = "AUC") +
  theme_pub + color_palette

ggsave(file.path(out_dir, "stask_vs_latent_dim.png"), p_stask, width = 10, height = 6, dpi = 300)

# 3. Generalization Plots (VIB Only)

df_gen <- df_lat %>%
  group_by(latent_dim) %>%
  summarise(
    mean_train = mean(train_loss_eval_vib), sd_train = sd(train_loss_eval_vib),
    mean_test = mean(test_loss_eval_vib), sd_test = sd(test_loss_eval_vib),
    mean_gap = mean(gen_gap_vib), sd_gap = sd(gen_gap_vib),
    mean_auc = mean(auc_y_vib),
    .groups = 'drop'
  )

# Loss vs Latent Dim
p_loss <- ggplot(df_gen, aes(x = as.factor(latent_dim))) +
  geom_line(aes(y = mean_train, color = "Train Loss", group = 1), size = 1.2) +
  geom_point(aes(y = mean_train, color = "Train Loss"), size = 3) +
  geom_line(aes(y = mean_test, color = "Test Loss", group = 1), size = 1.2) +
  geom_point(aes(y = mean_test, color = "Test Loss"), size = 3) +
  labs(title = "Eval Loss vs Latent Dimension (VIB)", x = "Latent Dimension", y = "Loss", color = "Split") +
  theme_pub + scale_color_viridis_d(option = "mako", end = 0.7)

ggsave(file.path(out_dir, "loss_vs_latent_dim.png"), p_loss, width = 8, height = 5, dpi = 300)

# Gen Gap vs Latent Dim
p_gap <- ggplot(df_gen, aes(x = as.factor(latent_dim), y = mean_gap, group = 1)) +
  geom_line(size = 1.2, color = "darkred") +
  geom_point(size = 3, color = "darkred") +
  geom_errorbar(aes(ymin = mean_gap - sd_gap, ymax = mean_gap + sd_gap), width = 0.2, color = "darkred") +
  labs(title = "Generalization Gap vs Latent Dimension (VIB)", x = "Latent Dimension", y = "Generalization Gap (Test - Train Loss)") +
  theme_pub

ggsave(file.path(out_dir, "gengap_vs_latent_dim.png"), p_gap, width = 8, height = 5, dpi = 300)

# Scatter: Train vs Test Loss
p_scatter_loss <- ggplot(df_lat, aes(x = train_loss_eval_vib, y = test_loss_eval_vib, color = as.factor(latent_dim))) +
  geom_point(size = 3, alpha = 0.8) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  labs(title = "Train vs Test Loss (VIB)", subtitle = "Dashed line is 45° (x=y)", x = "Train Loss", y = "Test Loss", color = "Latent Dim") +
  theme_pub + color_palette

ggsave(file.path(out_dir, "scatter_train_test_loss.png"), p_scatter_loss, width = 8, height = 6, dpi = 300)

# Scatter: AUC Y vs Gen Gap
p_scatter_gap <- ggplot(df_lat, aes(x = gen_gap_vib, y = auc_y_vib, color = as.factor(latent_dim))) +
  geom_point(size = 3, alpha = 0.8) +
  labs(title = "AUC Y vs Generalization Gap (VIB)", x = "Generalization Gap", y = "AUC (Y)", color = "Latent Dim") +
  theme_pub + color_palette

ggsave(file.path(out_dir, "scatter_auc_vs_gengap.png"), p_scatter_gap, width = 8, height = 6, dpi = 300)

# 4. Info Theory Trace Plots 

# Summarize trace data over cross-validation splits
df_trace_sum <- df_trace %>%
  group_by(latent_dim, epoch) %>%
  summarise(
    across(c(kl_per_dim_mean, c, beta, val_auc_y, loss_task_mean, loss_y_mean, loss_s_mean, loss_w_mean), 
           list(mean = ~mean(.x, na.rm = TRUE), sd = ~sd(.x, na.rm = TRUE))),
    .groups = 'drop'
  )

# KL Enforcement
p_kl <- ggplot(df_trace_sum, aes(x = epoch)) +
  geom_line(aes(y = kl_per_dim_mean_mean), color = "steelblue", size = 1) +
  geom_ribbon(aes(ymin = kl_per_dim_mean_mean - kl_per_dim_mean_sd, ymax = kl_per_dim_mean_mean + kl_per_dim_mean_sd), alpha = 0.2, fill = "steelblue") +
  geom_line(aes(y = c_mean), color = "darkred", linetype = "dashed", size = 1) +
  geom_vline(xintercept = warmup_boundary, linetype = "dotted", color = "black", size = 0.8) +
  facet_wrap(~latent_dim, scales = "free_y", ncol = 5) +
  labs(title = "KL Enforcement per Epoch", subtitle = "Dashed red = Budget (C), Dotted black = Warmup Boundary",
       x = "Epoch", y = "KL per Dimension") +
  theme_pub

ggsave(file.path(out_dir, "trace_kl_enforcement.png"), p_kl, width = 14, height = 8, dpi = 300)

# Dual Variable Dynamics (Beta)
p_beta <- ggplot(df_trace_sum, aes(x = epoch, y = beta_mean)) +
  geom_line(color = "darkorange", size = 1) +
  geom_ribbon(aes(ymin = pmax(0, beta_mean - beta_sd), ymax = beta_mean + beta_sd), alpha = 0.2, fill = "darkorange") +
  geom_vline(xintercept = warmup_boundary, linetype = "dotted", color = "black", size = 0.8) +
  facet_wrap(~latent_dim, scales = "free_y", ncol = 5) +
  labs(title = "Dual Variable (Beta) Dynamics vs Epoch", x = "Epoch", y = "Beta") +
  theme_pub

ggsave(file.path(out_dir, "trace_beta_dynamics.png"), p_beta, width = 14, height = 8, dpi = 300)

# Validation Performance Over Training
p_val_auc <- ggplot(df_trace_sum, aes(x = epoch, y = val_auc_y_mean)) +
  geom_line(color = "forestgreen", size = 1) +
  geom_ribbon(aes(ymin = val_auc_y_mean - val_auc_y_sd, ymax = val_auc_y_mean + val_auc_y_sd), alpha = 0.2, fill = "forestgreen") +
  geom_vline(xintercept = warmup_boundary, linetype = "dotted", color = "black", size = 0.8) +
  facet_wrap(~latent_dim, ncol = 5) +
  labs(title = "Validation AUC Y vs Epoch", x = "Epoch", y = "Val AUC Y") +
  theme_pub

ggsave(file.path(out_dir, "trace_val_auc_vs_epoch.png"), p_val_auc, width = 14, height = 8, dpi = 300)

# 5. All Losses Over Epoch

# Pivot long for multiple losses
df_trace_losses <- df_trace_sum %>%
  select(latent_dim, epoch, ends_with("_mean_mean")) %>%
  rename_with(~gsub("_mean_mean$", "", .), ends_with("_mean_mean")) %>%
  pivot_longer(
    cols = c(loss_task, loss_y, loss_s, loss_w),
    names_to = "loss_type",
    values_to = "loss_value"
  ) %>%
  mutate(loss_type = toupper(gsub("loss_", "Loss ", loss_type)))

p_losses_epoch <- ggplot(df_trace_losses, aes(x = epoch, y = loss_value, color = loss_type)) +
  geom_line(size = 0.8) +
  geom_vline(xintercept = warmup_boundary, linetype = "dotted", color = "black", size = 0.6) +
  facet_wrap(~latent_dim, scales = "free_y", ncol = 5) +
  labs(title = "Loss Components vs Epoch by Latent Dimension", x = "Epoch", y = "Loss Value", color = "Loss Component") +
  theme_pub + color_palette

# 6. Rate Distortion Trade off

# Calculate Distortion (1 - AUC)
df_rd <- df_lat %>%
  mutate(
    distortion_y = 1 - auc_y_vib,
    distortion_w = rmse_w_vib # RMSE is already a distortion metric
  ) %>%
  group_by(latent_dim) %>%
  summarise(
    mean_kl = mean(kl_total_vib, na.rm=TRUE),
    mean_dist_y = mean(distortion_y, na.rm=TRUE),
    mean_dist_w = mean(distortion_w, na.rm=TRUE),
    .groups = 'drop'
  )

# Plot Total KL vs Distortion Y
p_rd_y <- ggplot(df_rd, aes(x = mean_kl, y = mean_dist_y, label = latent_dim)) +
  geom_line(color = "darkblue", size = 1) +
  geom_point(color = "darkblue", size = 3) +
  geom_text(vjust = -1, size = 4) +
  labs(title = "Rate-Distortion Trade-off (Task Y)", subtitle = "Labels indicate Latent Dimension",
       x = "Information Rate (Total KL)", y = "Task Distortion (1 - AUC Y)") +
  theme_pub

ggsave(file.path(out_dir, "rate_distortion_y.png"), p_rd_y, width = 8, height = 6, dpi = 300)

# 7. Multi-Task Distortion Geometry 

# Error on Y vs Error on W
df_plane <- df_lat %>%
  select(split_idx, latent_dim, 
         auc_y_nn, rmse_w_nn, 
         auc_y_vib, rmse_w_vib, 
         auc_y_lr, rmse_w_lr) %>%
  pivot_longer(
    cols = -c(split_idx, latent_dim),
    names_to = c(".value", "model"),
    names_pattern = "(.*)_(nn|vib|lr)"
  ) %>%
  mutate(
    distortion_y = 1 - auc_y,
    distortion_w = rmse_w,
    model = toupper(model)
  ) %>%
  group_by(model, latent_dim) %>%
  summarise(
    mean_dist_y = mean(distortion_y, na.rm=TRUE),
    mean_dist_w = mean(distortion_w, na.rm=TRUE),
    .groups = 'drop'
  )

# Plot Distortion Plane
p_plane <- ggplot(df_plane, aes(x = mean_dist_y, y = mean_dist_w, color = model, shape = model)) +
  geom_point(size = 4, alpha = 0.8) +
  geom_path(aes(group = model), alpha = 0.5, linetype = "dashed") + 
  labs(title = "Multi-Task Distortion Plane", 
       subtitle = "Bottom-Left is the optimal Pareto Frontier (Lower Error on Both Tasks)",
       x = "Main Task Distortion (1 - AUC Y)", 
       y = "Co abundance Task Distortion (RMSE W)") +
  theme_pub + color_palette

ggsave(file.path(out_dir, "distortion_plane.png"), p_plane, width = 8, height = 6, dpi = 300)

# 8. Efficiency of Information Allocation 

# Show that total KL does not blindly explode with latent_dim
p_efficiency <- ggplot(df_lat, aes(x = as.factor(latent_dim), y = kl_total_vib)) +
  geom_boxplot(fill = "lightblue", color = "darkblue", alpha = 0.7) +
  labs(title = "Total Information Capacity vs Latent Dimension", 
       subtitle = "Shows how VIB controls total KL regardless of physical bottleneck size",
       x = "Latent Dimension", y = "Total KL (Rate)") +
  theme_pub

ggsave(file.path(out_dir, "efficiency_kl_vs_dim.png"), p_efficiency, width = 8, height = 5, dpi = 300)

ggsave(file.path(out_dir, "trace_all_losses_epoch.png"), p_losses_epoch, width = 16, height = 8, dpi = 300)

cat("Done. Plots saved to:", out_dir, "\n")