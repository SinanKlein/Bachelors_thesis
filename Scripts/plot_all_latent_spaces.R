# Plot PCA/correlation diagnostics for all exported latent spaces.
# Usage:
#   Rscript plot_all_latent_spaces.R <run_id> <dataset_name>

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

# Resolve the *Rscript* front-end, not commandArgs(FALSE)[1] (which is Rterm/R
# and would open an interactive session instead of running the script).
get_rscript <- function() {
  exe <- if (.Platform$OS.type == "windows") "Rscript.exe" else "Rscript"
  cand <- file.path(R.home("bin"), exe)
  if (file.exists(cand)) return(cand)
  cand_x64 <- file.path(R.home("bin"), "x64", exe)   # older Windows layouts
  if (file.exists(cand_x64)) return(cand_x64)
  found <- Sys.which("Rscript")                       # fall back to PATH
  if (nzchar(found)) return(unname(found))
  stop("Could not locate the Rscript executable.")
}

script_dir <- get_script_dir()
rscript    <- get_rscript()
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: Rscript plot_all_latent_spaces.R <run_id> <dataset_name>")
run_id <- args[[1]]
dataset_name <- if (length(args) >= 2) args[[2]] else ""
models <- c("twotower_y_only", "twotower_w_only", "twotower_yw_joint", "mlp_yw_joint")
for (m in models) {
  message("[plot_all_latent_spaces] plotting ", m)
  cmd_args <- shQuote(c(file.path(script_dir, "plot_latent_space.R"),
                        run_id, dataset_name, m))
  status <- system2(command = rscript, args = cmd_args)
  if (!identical(status, 0L)) stop("plot_latent_space.R failed for ", m)
}
