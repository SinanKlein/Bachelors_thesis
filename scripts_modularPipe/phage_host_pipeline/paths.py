"""
Central path configuration.

Fill in the two paths below for your environment. Every script imports
from here. Nothing else in the pipeline hardcodes paths.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# USER: fill these in
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# THE ONLY THING TO CHANGE WHEN SWITCHING COHORT
# ---------------------------------------------------------------------------
# One token drives the raw data folder, the graph_data folder and the results
# folder. It used to live in four places across two files, which made it
# possible to build one cohort's data into another cohort's results directory
# without any error — set it here and nowhere else.
COHORT = "GvHD"                 # "CRC" | "GvHD" | "IBD"

# Roots that do not change between cohorts.
DATA_ROOT   = Path(r"C:\Sinan_Klein\LMU\lmu_thesis\datas_final\SINAN_datasets\data")
SPLITS_ROOT = Path(r"C:\Sinan_Klein\LMU\lmu_thesis\multimodal_network\data")
RESULTS_DIR = Path(r"C:\Sinan_Klein\LMU\lmu_thesis\results_modularPipe")

# Derived. RAW_DATA_DIR holds graph_data.npz; COHORT_DIR holds the raw tables.
RAW_DATA_DIR = SPLITS_ROOT / f"splits_unified_{COHORT}"
COHORT_DIR   = DATA_ROOT / COHORT

# Name of the dataset currently being run. ALL outputs for this run are nested
# under RESULTS_DIR / DATASET_NAME / <run_id>, so different datasets never mix.
# Change this when you point the pipeline at a different dataset.
DATASET_NAME = f"{COHORT}_outputs"

# ---------------------------------------------------------------------------
# Auto-derived 
# ---------------------------------------------------------------------------
PIPELINE_DIR    = Path(__file__).resolve().parent
CONFIG_DIR      = PIPELINE_DIR
R_DIR           = PIPELINE_DIR
GRAPH_DATA_FILE = RAW_DATA_DIR / "graph_data.npz"


# Genus -> family taxonomy table, used by the family/genus aggregation step.
# The filename suffix differs by cohort (GvHD/IBD ship taxonomy_table_F.csv,
# CRC ships taxonomy_table_P.csv), so this is resolved by glob rather than
# hardcoded. Set TAXONOMY_FILE explicitly to override.
TAXONOMY_DIR  = DATA_ROOT
TAXONOMY_FILE = None   # None -> auto-resolve via find_taxonomy_file()


def find_taxonomy_file(dataset_name: str = DATASET_NAME) -> Path | None:
    """Locate the DADA2 taxonomy table for a cohort.

    DATASET_NAME carries an "_outputs" suffix in this pipeline (e.g.
    "CRC_outputs"), while the raw data folder is just the cohort ("CRC"), so the
    suffix is stripped before looking. Returns None when nothing matches, which
    makes the family step skip itself rather than crash the run.
    """
    if TAXONOMY_FILE is not None:
        return Path(TAXONOMY_FILE)
    cohort = str(dataset_name).replace("_outputs", "").strip()
    d = TAXONOMY_DIR / cohort / "DADA2"
    if not d.is_dir():
        return None
    # prefer the plain table over the *_progenomeLikeNCBIIDs variant
    cands = [p for p in sorted(d.glob("taxonomy_table_*.csv"))
             if "progenome" not in p.name.lower()]
    return cands[0] if cands else None


# ---------------------------------------------------------------------------
# Per run subfolders (auto created on access)
# ---------------------------------------------------------------------------
def run_dir(run_id: str) -> Path:
    # everything for this run lives under RESULTS_DIR / DATASET_NAME / run_id
    d = RESULTS_DIR / DATASET_NAME / run_id
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

def ablation_dir(run_id: str) -> Path:
    """Everything the ablation produces lives here, in one folder."""
    d = run_dir(run_id) / "ablation"
    d.mkdir(parents=True, exist_ok=True)
    return d

