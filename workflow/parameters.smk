# config/parameters.smk

from pathlib import Path
import pandas as pd
from itertools import product

import re
import pandas as pd

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
RESULTS_DIR = Path(config['results_dir'])
FIGURES_DIR = Path(config['figures_dir'])
PALETTE_DIR = Path(config['xenium_metadata_dir'])
STD_SEURAT_ANALYSIS_DIR = Path(config['xenium_std_seurat_analysis_dir'])
XENIUM_PROCESSED_DIR = Path(config['xenium_processed_dir'])
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
# Storing as a DataFrame is good practice for sets of parameters
UMAP_PARAMS = pd.DataFrame(
    product([50], [50], [0.5], ['euclidean']),
    columns=['n_comps', 'n_neighbors', 'min_dist', 'metric']
)

# --- Plotting Style Parameters ---
PLOT_PARAMS = {
    's': 0.5,
    'alpha': 0.5,
    'dpi': 300,
    'points_only': False,
    'extension': 'png',
    'cell_type_palette': PALETTE_DIR / 'col_palette_cell_types.csv',
    'panel_palette': PALETTE_DIR / 'col_palette_panel.csv',
    'sample_palette': PALETTE_DIR / 'col_palette_sample.csv',

}

# --- Plotting Logic Parameters ---
# These control *what* gets plotted
LIST_PARAMS = {
    'normalisation': ['lognorm'],
    'layer': ['data', 'scale_data'],
    # 'reference': ['GEO_GSE178341', 'GEO_GSE236581', 'Marteau2024'],
    'method': ['rctd_class_aware'],
    # 'level': ['sample', 'Level1', 'Level2', 'Level3', 'Level4'],
    'plot_type': ['umap', 'facet_umap']
}

REF_LEVELS = {
    'GEO_GSE178341': ['Level1', 'Level2', 'sample'],
    'GEO_GSE236581': ['Level1', 'Level2', 'Level3', 'sample'],
    'Marteau2024':   ['Level1', 'Level2', 'Level3', 'Level4', 'sample']
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


# =================================================================================
# IV. RULE-SPECIFIC TARGET GENERATION
# Encapsulated logic to generate file lists for 'all' rules.
# =================================================================================

def get_outputs_embed_panel():
    """Builds the full list of output files by expanding params for each valid path."""
    
    # The full parameter space to expand for each panel
    param_combinations = expand(
        "/{normalisation}/umap_{layer}_n_comps={n_comps}_n_neighbors={n_neighbors}_min_dist={min_dist}_metric={metric}.parquet",
        normalisation=LIST_PARAMS['normalisation'],
        layer=LIST_PARAMS['layer'],
        n_comps=UMAP_PARAMS['n_comps'],
        n_neighbors=UMAP_PARAMS['n_neighbors'],
        min_dist=UMAP_PARAMS['min_dist'],
        metric=UMAP_PARAMS['metric']
    )
    
    # The valid base directories for each panel
    base_dirs = expand(
        str(RESULTS_DIR / "xenium/embed_panel/{segmentation}/{condition}/{panel}"),
        zip,
        segmentation=PATHS_PARAMS['segmentation'],
        condition=PATHS_PARAMS['condition'],
        panel=PATHS_PARAMS['panel']
    )
    
    # Create the final list using a list comprehension
    # For each base directory, append each parameter combination.
    target_files = [
        f"{base_dir}{param_combo}"
        for base_dir in base_dirs
        for param_combo in param_combinations
    ]
    return target_files


def get_outputs_embed_panel_plot():
    """Generates the list of output files for the embed_panel_plot rule."""

    # ref levels to df
    ref_level_pairs = [
        {'reference': ref, 'level': lvl}
        for ref, levels in REF_LEVELS.items()
        for lvl in levels
    ]
    ref_level_df = pd.DataFrame(ref_level_pairs)

    # list params to df
    list_params_df = pd.DataFrame(
        product(*LIST_PARAMS.values()), columns=LIST_PARAMS.keys()
    ).drop(columns=['reference', 'level'])

    # combine all params
    base_df = PATHS_PARAMS[['segmentation',	'condition','panel']].drop_duplicates().merge(list_params_df, how='cross')
    combined_df = base_df.merge(ref_level_df, how='cross')
    final_df_with_umap = combined_df.merge(UMAP_PARAMS, how='cross')

    query_filters = ["not (plot_type == 'facet_umap' and level == 'sample')"]
    final_df_filtered = final_df_with_umap.query(" and ".join(query_filters))

    # Path building logic 
    def build_path(row):
        dir_path = FIGURES_DIR / "xenium/embed_panel" / row['segmentation'] / row['condition'] / row['panel'] / row['normalisation']
        filename = (
            f"{row['plot_type']}_{row['layer']}_"
            f"n_comps={row['n_comps']}_n_neighbors={row['n_neighbors']}_min_dist={row['min_dist']}_metric={row['metric']}_"
            f"{row['reference']}_{row['method']}_{row['level']}.{PLOT_PARAMS['extension']}"
        )
        return dir_path / filename

    return final_df_filtered.apply(build_path, axis=1).tolist()

# Generate the list of target files for the `all` rule to consume
OUTPUTS_EMBED_PANEL = get_outputs_embed_panel()
OUTPUTS_EMBED_PANEL_PLOT = get_outputs_embed_panel_plot()



