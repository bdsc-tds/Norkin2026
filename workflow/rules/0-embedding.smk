rule embed_panel:
    input:
        # The input panel path is constructed dynamically from the wildcards
        panel=STD_SEURAT_ANALYSIS_DIR / "{segmentation}/{condition}/{panel}"
    output:
        # The output file path captures all the wildcards from the requested file
        out_file=RESULTS_DIR / "xenium/embed_panel/{segmentation}/{condition}/{panel}/{normalisation}/umap_{layer}_n_comps={n_comps}_n_neighbors={n_neighbors}_min_dist={min_dist}_metric={metric}.parquet"
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
    threads: 1
    resources:
        # Use a function to dynamically set resources based on wildcards
        mem=lambda wildcards: '100G' if wildcards.panel == '5k' else '50G',
        runtime=lambda wildcards: '8h' if wildcards.panel == '5k' else '3h',
        # slurm_partition="gpu",
        # slurm_extra='--gres=gpu:1',
    shell:
        """
        pixi run -e {params.pixi_env} python workflow/scripts/xenium/embed_panel.py \
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
            --min_cells {params.min_cells}
        """


rule embed_panel_all:
    input:
        OUTPUTS_EMBED_PANEL
