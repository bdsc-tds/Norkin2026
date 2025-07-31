rule concatenate_seurat_by_segmentation:
    input:
        data_dir=STD_SEURAT_ANALYSIS_DIR,
        results_dir=RESULTS_DIR / "xenium/embed_and_cluster_panel/raw"
    output:
        # The output file, with wildcards for segmentation and condition
        rds=RESULTS_DIR/"xenium/concatenate_seurat_by_segmentation/raw/{segmentation}_{condition}_{panel}_{normalisation}.rds"
    log:
        LOG_DIR / "xenium/concatenate_seurat_by_segmentation/raw/{segmentation}_{condition}_{panel}_{normalisation}.log"
    params:
        # We pass the wildcards as parameters to the script
        segmentation="{segmentation}",
        condition="{condition}",
        panel="{panel}",
        normalisation="{normalisation}",
        pixi_env=PIXI_ENV
    threads: 1
    resources:
        mem='100G',
        runtime='2h',
    shell:
        """
        pixi run -e {params.pixi_env} Rscript workflow/scripts/xenium/concatenate_samples_by_segmentation.R \
            --input-dir {input.data_dir} \
            --results-dir {input.results_dir} \
            --output-file {output.rds} \
            --segmentation {params.segmentation} \
            --condition {params.condition} \
            --panel {params.panel} \
            --normalisation {params.normalisation} \
            > {log} 2>&1
        """

# outputs
def get_outputs_concatenate_seurat_by_segmentation():
    """Builds the full list of output files by expanding params for each valid path."""
    
    # The full parameter space to expand for each panel
    param_combinations = expand(
        "_{normalisation}.rds",
        normalisation=LIST_PARAMS['normalisation'],
    )
    
    # The valid base directories for each panel
    base_dirs = expand(
        str(RESULTS_DIR / "xenium/concatenate_seurat_by_segmentation/raw/{segmentation}_{condition}_{panel}"),
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

rule concatenate_seurat_by_segmentation_all:
    input:
        get_outputs_concatenate_seurat_by_segmentation()

