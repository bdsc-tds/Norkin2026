import pandas as pd
import re
import pandas as pd
from itertools import product
from pathlib import Path

def _generate_from_dict(params_dict):
    """Creates regex constraints from a dictionary of lists."""
    constraints = {}
    for wildcard, values in params_dict.items():
        if isinstance(values, list) and values:
            escaped_values = [re.escape(str(v)) for v in values]
            constraints[wildcard] = r"|".join(escaped_values)
    return constraints

def _generate_from_dataframe(params_df):
    """Creates regex constraints from the unique values in DataFrame columns."""
    constraints = {}
    for wildcard in params_df.columns:
        values = params_df[wildcard].unique().tolist()
        if values:
            escaped_values = [re.escape(str(v)) for v in values]
            constraints[wildcard] = r"|".join(escaped_values)
    return constraints

def generate_wildcard_constraints(*param_objects):
    """
    Generates a Snakemake wildcard_constraints dictionary by inspecting
    various parameter objects.

    Accepts any number of dictionaries or pandas DataFrames. It automatically
    detects the type of each object and generates the appropriate regex
    constraints, combining them into a single dictionary.

    Args:
        *param_objects: A variable number of dicts or pandas.DataFrames.

    Returns:
        A dictionary suitable for use with Snakemake's wildcard_constraints.
    """
    all_constraints = {}
    for obj in param_objects:
        if isinstance(obj, dict):
            # Dispatch to the dictionary helper
            constraints = _generate_from_dict(obj)
            all_constraints.update(constraints)
        elif isinstance(obj, pd.DataFrame):
            # Dispatch to the DataFrame helper
            constraints = _generate_from_dataframe(obj)
            all_constraints.update(constraints)
        # Silently ignore any other types
    return all_constraints


# =================================================================================
# I. CORE CONFIGURATION AND PATH DISCOVERY
# =================================================================================

# --- Paths from main config ---
PIXI_ENV = 'norkin-organoid'
PIXI_ENV_CELLCHARTER = 'cellcharter'
LOG_DIR = Path(config['log_dir'])
RESULTS_DIR = Path(config['results_dir'])
FIGURES_DIR = Path(config['figures_dir'])
PALETTE_DIR = Path(config['xenium_metadata_dir'])
XENIUM_PROCESSED_DIR = Path(config['xenium_processed_dir'])
STD_SEURAT_ANALYSIS_DIR = Path(config['xenium_std_seurat_analysis_dir'])
COUNT_CORRECTION_DIR = Path(config['xenium_count_correction_dir'])
CELL_TYPE_ANNOTATION_DIR = Path(config['xenium_cell_type_annotation_dir'])
CELL_TYPE_PALETTE = Path(config['cell_type_palette'])
PANEL_PALETTE = Path(config['panel_palette'])
SAMPLE_PALETTE = Path(config['sample_palette'])

# --- Master Wildcard DataFrame from File System ---
path_parts = [p.parent.parts[-5:] for p in STD_SEURAT_ANALYSIS_DIR.glob("*/*/*/*/*/*")]
PATHS_PARAMS = pd.DataFrame(path_parts, columns=["segmentation", "condition", "panel", "donor", "sample"])

# # Filter out any unwanted top-level directories
# PATHS_PARAMS = PATHS_PARAMS[
#     ~PATHS_PARAMS['segmentation'].isin(['proseg_mode', 'bats_normalised', 'bats_expected'])
# ]

# =================================================================================
# II. GENERAL PARAMETER DEFINITIONS
# These are parameters used across multiple rules.
# =================================================================================

# --- QC Parameters ---
QC_PARAMS = {
    'min_counts': 10,
    'min_features': 5,
    'max_counts': float("inf"),
    'max_features': float("inf"),
    'min_cells': 5
}

# --- UMAP Parameters ---
UMAP_PARAMS = pd.DataFrame(
    product([50], [50], [0.5], ['euclidean']),
    columns=['n_comps', 'n_neighbors', 'min_dist', 'metric']
)

# --- Plotting Style Parameters ---
PLOT_PARAMS = {
    's': 1,
    'alpha': 0.5,
    'dpi': 100,
    'points_only': False,
    'extension': 'png',
    'cell_type_palette': PALETTE_DIR / 'col_palette_cell_types.csv',
    'panel_palette': PALETTE_DIR / 'col_palette_panel.csv',
    'sample_palette': PALETTE_DIR / 'col_palette_sample.csv',

}

# --- Plotting Logic Parameters ---
LIST_PARAMS = {
    'normalisation': ['lognorm'],
    'layer': ['data', 'scale_data'],
    'method': ['rctd_class_aware'],
    'plot_type': ['umap', 'facet_umap'],
    'correction_method':['raw','split_fully_purified']
}

REF_LEVELS = {
    'GEO_GSE178341': ['Level1', 'Level2', 'Level3', 'sample'],
    'GEO_GSE236581': ['Level1', 'Level2', 'sample'],
    # 'Marteau2024':   ['Level1', 'Level2', 'Level3', 'Level4', 'sample']
}
LIST_PARAMS['reference'] = list(REF_LEVELS.keys())
LIST_PARAMS['level'] = list(set().union(*REF_LEVELS.values()))


# =================================================================================
# III. AUTOMATIC WILDCARD CONSTRAINT GENERATION
# =================================================================================
WILDCARD_CONSTRAINTS = generate_wildcard_constraints(
    QC_PARAMS,
    PATHS_PARAMS,
    UMAP_PARAMS,
    PLOT_PARAMS,
    LIST_PARAMS,
    REF_LEVELS,

    # Add any other future param objects here
)



