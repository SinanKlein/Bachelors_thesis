"""
Central path configuration.

Fill in the two paths below for your environment. Every script imports
from here. Nothing else in the pipeline hardcodes paths.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# USER: fill these in
# ---------------------------------------------------------------------------
RAW_DATA_DIR = Path(r"C:\Sinan_Klein\LMU\lmu_thesis\multimodal_network\data\splits_unified")    # folder containing graph_data.npz
RESULTS_DIR  = Path(r"C:\Sinan_Klein\LMU\lmu_thesis\results_modularPipe")     # folder where per-run results are saved

# ---------------------------------------------------------------------------
# Auto-derived 
# ---------------------------------------------------------------------------
PIPELINE_DIR    = Path(__file__).resolve().parent
CONFIG_DIR      = PIPELINE_DIR
R_DIR           = PIPELINE_DIR
GRAPH_DATA_FILE = RAW_DATA_DIR / "graph_data.npz"


# ---------------------------------------------------------------------------
# Per run subfolders (auto created on access)
# ---------------------------------------------------------------------------
def run_dir(run_id: str) -> Path:
    d = RESULTS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def preprocessing_dir(run_id: str) -> Path:
    d = run_dir(run_id) / "preprocessing"
    d.mkdir(parents=True, exist_ok=True)
    return d

def predictions_dir(run_id: str) -> Path:
    d = run_dir(run_id) / "predictions"
    d.mkdir(parents=True, exist_ok=True)
    return d

def metrics_dir(run_id: str) -> Path:
    d = run_dir(run_id) / "metrics"
    d.mkdir(parents=True, exist_ok=True)
    return d

def plots_dir(run_id: str) -> Path:
    d = run_dir(run_id) / "plots"
    (d / "data").mkdir(parents=True, exist_ok=True)
    (d / "results").mkdir(parents=True, exist_ok=True)
    return d
