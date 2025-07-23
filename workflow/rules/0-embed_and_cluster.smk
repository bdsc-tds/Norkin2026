rule embed_and_cluster_panel:
    input:
        # The input panel path is constructed dynamically from the wildcards
        panel=STD_SEURAT_ANALYSIS_DIR / "{segmentation}/{condition}/{panel}"
    output:
        # The output file path captures all the wildcards from the requested file
        out_file=RESULTS_DIR / "xenium/embed_and_cluster_panel/raw/{segmentation}/{condition}/{panel}/{normalisation}/umap_{layer}_n_comps={n_comps}_n_neighbors={n_neighbors}_min_dist={min_dist}_metric={metric}.parquet"
    params:
        # Parameters that are constant for all runs
        xenium_processed_dir=XENIUM_PROCESSED_DIR,
        min_counts=QC_PARAMS['min_counts'],
        min_features=QC_PARAMS['min_features'],
        max_counts=QC_PARAMS['max_counts'],
        max_features=QC_PARAMS['max_features'],
        min_cells=QC_PARAMS['min_cells'],
        
        # Pass wildcards to params so they can be used in the shell command
        normalisation="{normalisation}",
        layer="{layer}",
        n_comps="{n_comps}",
        n_neighbors="{n_neighbors}",
        metric="{metric}",
        min_dist="{min_dist}",
        pixi_env=PIXI_ENV
    log:
        "logs/xenium/embed_and_cluster_panel/raw/{segmentation}/{condition}/{panel}/{normalisation}/umap_{layer}_n_comps={n_comps}_n_neighbors={n_neighbors}_min_dist={min_dist}_metric={metric}.log"
    resources:
        mem=lambda wildcards: '100G' if wildcards.panel == '5k' else '50G',
        runtime=lambda wildcards: '8h' if wildcards.panel == '5k' else '3h',
        # slurm_partition="gpu",
        # slurm_extra='--gres=gpu:1',
    shell:
        """
        pixi run -e {params.pixi_env} python workflow/scripts/xenium/embed_and_cluster_panel.py \
            --panel {input.panel} \
            --out_file {output.out_file} \
            --xenium_processed_data_dir {params.xenium_processed_dir} \
            --normalisation {params.normalisation} \
            --layer {params.layer} \
            --n_comps {params.n_comps} \
            --n_neighbors {params.n_neighbors} \
            --metric {params.metric} \
            --min_dist {params.min_dist} \
            --min_counts {params.min_counts} \
            --min_features {params.min_features} \
            --max_counts {params.max_counts} \
            --max_features {params.max_features} \
            --min_cells {params.min_cells} \
            > {log} 2>&1
        """


# outputs
def get_outputs_embed_and_cluster_panel():
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
        str(RESULTS_DIR / "xenium/embed_and_cluster_panel/raw/{segmentation}/{condition}/{panel}"),
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

rule embed_and_cluster_panel_all:
    input:
        get_outputs_embed_and_cluster_panel()

