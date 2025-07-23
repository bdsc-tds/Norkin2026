rule embed_panel_plot:
    input:
        panel = STD_SEURAT_ANALYSIS_DIR / '{segmentation}/{condition}/{panel}',
        embed_file = RESULTS_DIR / "xenium/embed_and_cluster_panel/raw/{segmentation}/{condition}/{panel}/{normalisation}/umap_{layer}_n_comps={n_comps}_n_neighbors={n_neighbors}_min_dist={min_dist}_metric={metric}.parquet"
    output:
        out_file = FIGURES_DIR / "xenium/embed_panel_plot/raw/{segmentation}/{condition}/{panel}/{normalisation}/{plot_type}_{layer}_n_comps={n_comps}_n_neighbors={n_neighbors}_min_dist={min_dist}_metric={metric}_{reference}_{method}_{color}.{extension}"
    params:
        # QC Params
        min_counts=QC_PARAMS['min_counts'],
        min_features=QC_PARAMS['min_features'],
        max_counts=QC_PARAMS['max_counts'],
        max_features=QC_PARAMS['max_features'],
        min_cells=QC_PARAMS['min_cells'],
        
        # Plotting Style Params
        s=PLOT_PARAMS['s'],
        alpha=PLOT_PARAMS['alpha'],
        dpi=PLOT_PARAMS['dpi'],
        points_only=lambda wildcards: '--points_only' if PLOT_PARAMS['points_only'] else '',
        
        # Dynamic Params from wildcards
        facet=lambda wildcards: '--facet' if wildcards.plot_type == 'facet_umap' else '',
        reference="{reference}",
        method="{method}",
        color="{color}",
        normalisation="{normalisation}",

        # Static Params
        cell_type_annotation_dir=CELL_TYPE_ANNOTATION_DIR,
        cell_type_palette=CELL_TYPE_PALETTE,
        panel_palette=PANEL_PALETTE,
        sample_palette=SAMPLE_PALETTE,
        pixi_env=PIXI_ENV
    log:
        "logs/xenium/embed_panel_plot/raw/{segmentation}/{condition}/{panel}/{normalisation}/{plot_type}_{layer}_n_comps={n_comps}_n_neighbors={n_neighbors}_min_dist={min_dist}_metric={metric}_{reference}_{method}_{color}.{extension}.log"
    resources:
        mem='30G',
        runtime='10m'
    shell:
        """
        pixi run -e {params.pixi_env} python workflow/scripts/xenium/embed_panel_plot.py \
            --panel {input.panel} \
            --embed_file {input.embed_file} \
            --out_file {output.out_file} \
            --cell_type_annotation_dir {params.cell_type_annotation_dir} \
            --normalisation {params.normalisation} \
            --reference {params.reference} \
            --method {params.method} \
            --color {params.color} \
            --cell_type_palette {params.cell_type_palette} \
            --panel_palette {params.panel_palette} \
            --sample_palette {params.sample_palette} \
            --s {params.s} \
            --alpha {params.alpha} \
            --dpi {params.dpi} \
            {params.points_only} \
            {params.facet} \
            > {log} 2>&1
        """



# outputs
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
        dir_path = FIGURES_DIR / "xenium/embed_panel_plot/raw/" / row['segmentation'] / row['condition'] / row['panel'] / row['normalisation']
        filename = (
            f"{row['plot_type']}_{row['layer']}_"
            f"n_comps={row['n_comps']}_n_neighbors={row['n_neighbors']}_min_dist={row['min_dist']}_metric={row['metric']}_"
            f"{row['reference']}_{row['method']}_{row['level']}.{PLOT_PARAMS['extension']}"
        )
        return dir_path / filename

    return final_df_filtered.apply(build_path, axis=1).tolist()

rule embed_panel_plot_all:
    input:
        get_outputs_embed_panel_plot()


# rule embed_condition_plot_all:
#     input:
#         out_files_condition