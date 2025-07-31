rule run_cellcharter:
    input:
        xenium_dir=XENIUM_PROCESSED_DIR,
        count_correction_dir=COUNT_CORRECTION_DIR,
        seurat_analysis_dir=STD_SEURAT_ANALYSIS_DIR,
        cell_type_annotation_dir=CELL_TYPE_ANNOTATION_DIR,
    output:
        plot=          RESULTS_DIR/"xenium/cellcharter/{correction_method}/{segmentation}/{condition}/{panel}/autok_stability.png",
        labels=        RESULTS_DIR/"xenium/cellcharter/{correction_method}/{segmentation}/{condition}/{panel}/labels.parquet",
        X_scvi=        RESULTS_DIR/"xenium/cellcharter/{correction_method}/{segmentation}/{condition}/{panel}/X_scvi.parquet",
        X_cellcharter= RESULTS_DIR/"xenium/cellcharter/{correction_method}/{segmentation}/{condition}/{panel}/X_cellcharter.parquet",
        scvi_model=    directory(RESULTS_DIR/"xenium/cellcharter/{correction_method}/{segmentation}/{condition}/{panel}/scvi_model"),
        cc_models=     directory(RESULTS_DIR/"xenium/cellcharter/{correction_method}/{segmentation}/{condition}/{panel}/cellcharter_models"),
        output_dir=    directory(RESULTS_DIR/"xenium/cellcharter/{correction_method}/{segmentation}/{condition}/{panel}")
    params:
        normalisation='lognorm',
        reference='GEO_GSE178341',
        method='rctd_class_aware',
        level='Level1',
        # The script will create this directory; we just need to pass its path.
        pixi_env=PIXI_ENV_CELLCHARTER
    log:
        LOG_DIR / "xenium/cellcharter/{correction_method}/{segmentation}/{condition}/{panel}.log"
    resources:
        mem='50G',
        runtime='2h',
        slurm_partition = "gpu",
        slurm_extra = '--gres=gpu:1',
    shell:
        """
        pixi run -e {params.pixi_env} python workflow/scripts/xenium/run_cellcharter.py \
            --xenium-dir {input.xenium_dir} \
            --count-correction-dir {input.count_correction_dir} \
            --seurat-analysis-dir {input.seurat_analysis_dir} \
            --cell-type-annotation-dir {input.cell_type_annotation_dir} \
            --output-dir {output.output_dir} \
            --correction-method {wildcards.correction_method} \
            --segmentation {wildcards.segmentation} \
            --condition {wildcards.condition} \
            --panel {wildcards.panel} \
            --normalisation {params.normalisation} \
            --reference {params.reference} \
            --method {params.method} \
            --level {params.level} \
            --max-workers {threads} \
            > {log} 2>&1
        """


rule run_cellcharter_cohort:
    input:
        xenium_dir=XENIUM_PROCESSED_DIR,
        count_correction_dir=COUNT_CORRECTION_DIR,
        seurat_analysis_dir=STD_SEURAT_ANALYSIS_DIR,
        cell_type_annotation_dir=CELL_TYPE_ANNOTATION_DIR,
    output:
        plot=          RESULTS_DIR/"xenium/cellcharter_cohort/{correction_method}/{segmentation}/autok_stability.png",
        labels=        RESULTS_DIR/"xenium/cellcharter_cohort/{correction_method}/{segmentation}/labels.parquet",
        X_scvi=        RESULTS_DIR/"xenium/cellcharter_cohort/{correction_method}/{segmentation}/X_scvi.parquet",
        X_cellcharter= RESULTS_DIR/"xenium/cellcharter_cohort/{correction_method}/{segmentation}/X_cellcharter.parquet",
        scvi_model=    directory(RESULTS_DIR/"xenium/cellcharter_cohort/{correction_method}/{segmentation}/scvi_model"),
        cc_models=     directory(RESULTS_DIR/"xenium/cellcharter_cohort/{correction_method}/{segmentation}/cellcharter_models"),
        output_dir=    directory(RESULTS_DIR/"xenium/cellcharter_cohort/{correction_method}/{segmentation}")
    params:
        normalisation='lognorm',
        reference='GEO_GSE178341',
        method='rctd_class_aware',
        level='Level1',
        
        # The script will create this directory; we just need to pass its path.
        pixi_env=PIXI_ENV_CELLCHARTER
    log:
        LOG_DIR / "xenium/cellcharter_cohort/{correction_method}/{segmentation}.log"
    resources:
        mem='100G',
        runtime='8h',
        slurm_partition = "gpu",
        slurm_extra = '--gres=gpu:1',
    shell:
        """
        pixi run -e {params.pixi_env} python workflow/scripts/xenium/run_cellcharter.py \
            --xenium-dir {input.xenium_dir} \
            --count-correction-dir {input.count_correction_dir} \
            --seurat-analysis-dir {input.seurat_analysis_dir} \
            --cell-type-annotation-dir {input.cell_type_annotation_dir} \
            --output-dir {output.output_dir} \
            --correction-method {wildcards.correction_method} \
            --segmentation {wildcards.segmentation} \
            --condition 'all' \
            --panel 'all' \
            --normalisation {params.normalisation} \
            --reference {params.reference} \
            --method {params.method} \
            --level {params.level} \
            --max-workers {threads} \
            > {log} 2>&1
        """

#outputs

def get_outputs_cellcharter():
    """Builds the full list of output files by expanding params for each valid path."""
    
    zipped_paths = list(zip(
        PATHS_PARAMS['segmentation'],
        PATHS_PARAMS['condition'],
        PATHS_PARAMS['panel']
    ))

    output_files_panel = [
        RESULTS_DIR / f"xenium/cellcharter/{cm}/{seg}/{cond}/{pan}/labels.parquet"
        for cm in LIST_PARAMS['correction_method']
        for seg, cond, pan in zipped_paths
    ]

    output_files_cohort = [
        RESULTS_DIR / f"xenium/cellcharter_cohort/{cm}/{seg}/labels.parquet"
        for cm in LIST_PARAMS['correction_method']
        for seg in PATHS_PARAMS['segmentation'].unique()
    ]
   
    return output_files_panel, output_files_cohort

OUTPUTS_CELLCHARTER_PANEL, OUTPUTS_CELLCHARTER_COHORT = get_outputs_cellcharter()

rule run_cellcharter_all:
    input:
        OUTPUTS_CELLCHARTER_PANEL

rule run_cellcharter_cohort_all:
    input:
        OUTPUTS_CELLCHARTER_COHORT