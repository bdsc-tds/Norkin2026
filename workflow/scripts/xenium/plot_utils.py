from pathlib import Path

import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import anndata as ad
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, rankdata
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import pairwise_distances

import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sys

sys.path.append("../../scripts")
import readwrite

cfg = readwrite.config()


def prepare_clustered_data(
    df_comp,
    annotations,
    n_clusters=10,
    exclude_all_malignant=True,
    malignant_column="Epi",
    cmaps={},
    order_by="centroid",
):
    """
    Performs KMeans clustering and orders the data for consistent plotting.

    Args:
        df_comp (pd.DataFrame): The data to cluster and plot.
        annotations (dict): A dictionary where keys are annotation names (e.g., 'Donors')
                                 and values are pandas Series of the annotations.
        n_clusters (int): The number of clusters for KMeans.
        exclude_all_malignant (bool): Whether to filter out rows with 100% or 0% malignant cells.
        malignant_column (str): The name of the column representing malignant cells.
        cmaps (dict): A dictionary where keys are annotation names and values are matplotlib colormaps names.
        order_by (str): order cluster centroids only or order all points

    Returns:
        tuple: A tuple containing:
            - df_comp_ordered (pd.DataFrame): The data, ordered by cluster.
            - annotations_ordered (dict): A dict of ordered annotation Series. The cluster
                                              annotation is always the last element.
            - palettes (dict): A dict of color palettes corresponding to the annotations.
    """
    # Exclude all malignant if specified
    if exclude_all_malignant and malignant_column in df_comp.columns:
        idx_malignant = np.where(df_comp.columns == malignant_column)[0]
        idx_some_malignant = (
            (df_comp.iloc[:, idx_malignant] < 1) & (df_comp.iloc[:, idx_malignant] > 0)
        ).values.flatten()
        df_comp = df_comp.loc[idx_some_malignant]
        df_comp.index = np.arange(df_comp.shape[0])
        for key in annotations:
            annotations[key] = annotations[key].loc[idx_some_malignant]
            annotations[key].index = np.arange(annotations[key].shape[0])

    # Run KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(df_comp)
    clusters = kmeans.labels_

    # Order clusters by centroid positions
    centroids = kmeans.cluster_centers_
    pc = PCA(n_components=1).fit(centroids)
    cluster_order = np.argsort(pc.transform(centroids).ravel())

    if order_by != "centroid":
        all_points_projection = pc.transform(df_comp)

        # This links each original index to its cluster and its projection score.
        sorting_df = pd.DataFrame(
            {"cluster": clusters, "projection": all_points_projection.ravel()}, index=df_comp.index
        )

        # 4. Enforce the desired cluster order by converting to a categorical type
        #    This is crucial for the primary sort level.
        sorting_df["cluster"] = pd.Categorical(sorting_df["cluster"], categories=cluster_order, ordered=True)

        # 5. Perform a stable, two-level sort:
        #    - Level 1: Sort by the ordered cluster category.
        #    - Level 2: Sort by the projection score within each cluster.
        sorted_df = sorting_df.sort_values(by=["cluster", "projection"])

        # 6. The final ordered_indices is the index of this sorted DataFrame
        ordered_indices = sorted_df.index.tolist()

    else:
        # Create ordered index
        ordered_indices = []
        for c in cluster_order:
            ordered_indices.extend(np.where(clusters == c)[0])

    df_comp_ordered = df_comp.iloc[ordered_indices]

    # Order annotations
    annotations_ordered = {}

    clusters_ordered = pd.Series(clusters[ordered_indices], index=df_comp_ordered.index, name="Cluster")
    annotations_ordered["Cluster"] = clusters_ordered
    for k, anno in annotations.items():
        annotations_ordered[k] = anno.iloc[ordered_indices]

    # Create palettes for user annotations
    palettes = {}
    for k, anno in annotations.items():
        unique_vals = anno.unique().tolist()
        n_vals = len(unique_vals)
        has_nan = False

        if np.nan in unique_vals:
            unique_vals.remove(np.nan)
            n_vals -= 1
            has_nan = True

        # Use a predefined palette if available or generate one
        if k not in cmaps:
            palette_colors = sns.color_palette("tab20", n_vals)
        else:
            palette_colors = sns.color_palette(cmaps[k], n_vals)
        palettes[k] = dict(zip(unique_vals, palette_colors.as_hex()))

        if has_nan:
            palettes[k][np.nan] = "#FFFFFF"

    # Create and add the cluster palette
    u_clusters = np.unique(clusters_ordered)
    clusters_cmap = dict(zip(u_clusters, sns.color_palette("pastel", n_clusters).as_hex()))
    palettes["Cluster"] = clusters_cmap

    return df_comp_ordered, annotations_ordered, clusters_ordered, cluster_order, palettes


def clustermap(
    df_comp_ordered,
    annotations_ordered,
    palettes,
    legend_positions=None,
    output_path=None,
    figure_top_margin=0.95,
    separator_line_width=1.5,
    separator_line_color="white",
    colorbar_position=[0.02, 0.65, 0.03, 0.15],
    **kwargs,
):
    """
    Generates a clustermap with annotation labels, separators, and layout control.

    Args:
        df_comp_ordered (pd.DataFrame): The ordered data to plot.
        annotations_ordered (dict): Dict of ordered annotation Series. The last item
                                    in the dict will be plotted closest to the heatmap.
        palettes (dict): Dict of color palettes for the annotations.
        legend_positions (dict): A dictionary mapping annotation names to their
                                 legend's bbox_to_anchor coordinates.
        output_path (str): The path to save the figure.
        figure_top_margin (float): Adjusts the top margin to reduce whitespace.
        separator_line_width (float): The width of the lines between annotation bars.
        separator_line_color (str): The color of the separator lines.
        colorbar_position (list): The position of the colorbar as [left, bottom, width, height].
    """
    # Prepare row colors.
    row_colors = [anno.map(palettes[name]) for name, anno in annotations_ordered.items()]

    # Prepare legend positions
    if legend_positions is None:
        legend_positions = {}
        pos_ = 0.45
        for k in annotations_ordered.keys():
            legend_positions[k] = (0.02, pos_)
            pos_ -= 0.15

    g = sns.clustermap(
        df_comp_ordered,
        row_cluster=False,
        col_cluster=False,
        row_colors=row_colors,
        cmap="Purples",
        figsize=(12, 16),
        **kwargs,
    )

    g.fig.subplots_adjust(top=figure_top_margin)
    g.ax_heatmap.set_yticks([])
    g.ax_heatmap.set_ylabel("")
    g.ax_row_dendrogram.set_visible(False)
    g.ax_heatmap.tick_params(axis="x", labelsize=12)
    g.cax.set_position(colorbar_position)

    # Add labels and separator lines to the annotation color bars
    annotation_names = list(annotations_ordered.keys())
    g.ax_row_colors.set_xticks(np.arange(len(annotation_names)) + 0.5)
    g.ax_row_colors.set_xticklabels(annotation_names, rotation=90, ha="center")
    g.ax_row_colors.tick_params(axis="x", length=0, labelsize=10)

    for i in range(len(annotation_names) - 1):
        g.ax_row_colors.axvline(x=i + 1, color=separator_line_color, linewidth=separator_line_width)

    # Create legends with specified positions
    for name, anno in annotations_ordered.items():
        if name not in legend_positions:
            continue

        palette = palettes[name]
        patches = [mpatches.Patch(color=color, label=label) for label, color in palette.items()]
        g.fig.legend(
            handles=patches, title=name, loc="center left", frameon=False, bbox_to_anchor=legend_positions[name]
        )

    # Save the figure
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight", dpi=300)
        plt.close()


def stackplot(
    df_comp_ordered, annotations_ordered, palettes, proportions_palette_key, grouping_col_name, output_path=None
):
    """
    Generates stack plots with the cluster bar closest to the plot.

    Args:
        df_comp_ordered (pd.DataFrame): The ordered data to plot.
        annotations_ordered (dict): Dict of ordered annotation Series.
        palettes (dict): Dict of color palettes for the annotations.
        proportions_palette_key (dict): Key of palettes dictionary mapping stackplot bars to colors.
        grouping_col_name (str): The name of the annotation in annotations_ordered to group by.
        output_path (str): The path to save the figure.
    """

    proportions_palette = palettes[proportions_palette_key]

    if grouping_col_name not in annotations_ordered:
        raise ValueError(f"grouping_col_name '{grouping_col_name}' not found in annotations_ordered keys.")
    groups = annotations_ordered[grouping_col_name]

    unique_groups = groups.unique()
    n_groups = len(unique_groups)
    n_cols = int(np.ceil(np.sqrt(n_groups)))
    n_rows = int(np.ceil(n_groups / n_cols))

    f, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3), squeeze=False)
    axs = axs.flat

    # Define a height for each annotation bar
    bar_height = 0.1

    for i, group in enumerate(unique_groups):
        ax = axs[i]
        idx_group = np.where(groups == group)[0]
        if len(idx_group) == 0:
            continue

        df_group = df_comp_ordered.iloc[idx_group].T
        df_group.columns = np.arange(df_group.shape[1])

        # Stackplot
        colors = [proportions_palette.get(label, "gray") for label in df_group.index]
        stack = ax.stackplot(df_group.columns, *df_group.values, labels=df_group.index, colors=colors)

        # Store the original top limit of the stackplot
        _, top_limit = ax.get_ylim()

        ax.set_title(group)

        # Drawing Annotation Bars
        y_offset = 0
        # Exclude the grouping column itself from the annotation bars
        annotations_to_draw = {k: v for k, v in annotations_ordered.items() if k != grouping_col_name}

        for name, annot in annotations_to_draw.items():
            palette = palettes[name]
            colors_hex = annot.iloc[idx_group].map(palette)
            colors_rgb = np.array([mcolors.to_rgb(c) for c in colors_hex])[np.newaxis, :, :]

            # Position the bar below the previous one
            y_pos = -bar_height * (y_offset + 1)

            ax.imshow(colors_rgb, aspect="auto", extent=(0, df_group.shape[1] - 1, y_pos, y_pos + bar_height), zorder=2)
            y_offset += 1

        ax.margins(x=0)  # Remove horizontal padding

        # (FIX 2) Calculate the correct bottom limit
        num_annotations = len(annotations_to_draw)
        bottom_limit = -bar_height * num_annotations

        # (FIX 3) Set both top and bottom limits explicitly
        ax.set_ylim(bottom=bottom_limit, top=top_limit)

        ax.set_xticks([])  # Ensure x-ticks are off

    # Hide any unused subplots
    for i in range(n_groups, len(axs)):
        axs[i].set_visible(False)

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Create the legend
    f.legend(
        [p for p in stack],
        [lab.get_label() for lab in stack],
        title="Cell Type",
        loc="center right",
        bbox_to_anchor=(0.95, 0.5),
        frameon=False,
    )

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight", dpi=300)
        plt.close()
    plt.show()


def barplot(
    clusters_ordered,
    cluster_order,
    annotations_ordered,
    palettes,
    grouping_col_name,
    output_path=None,
):
    """
    Generates bar plots with the cluster bar closest to the plot.

    Args:
        clusters_ordered (pd.DataFrame): The ordered cluster data to plot.
        cluster_order (list): The order of the clusters.
        annotations_ordered (dict): Dict of ordered annotation Series.
        palettes (dict): Dict of color palettes for the annotations.
        grouping_col_name (str): The name of the annotation in annotations_ordered to group by.
        output_path (str): The path to save the figure.
    """

    # Compute donor × cluster composition table
    df_clusters_comp = (
        pd.DataFrame(np.vstack((clusters_ordered, annotations_ordered[grouping_col_name])), index=["cluster", "donor"])
        .T.groupby("donor")["cluster"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)[cluster_order]
    )
    # Define donor order
    pc = PCA(n_components=1).fit(df_clusters_comp)
    donor_order = np.argsort(pc.transform(df_clusters_comp).ravel())
    df_clusters_comp = df_clusters_comp.iloc[donor_order]

    # Define colors in cluster order
    colors = [palettes["Cluster"][i] for i in cluster_order]

    # Custom legend
    custom_handles = [mpatches.Patch(color=palettes["Cluster"][i], label=i) for i in range(len(cluster_order))]

    # Figure size based on number of donors
    num_samples = len(df_clusters_comp.index)
    fig_width = max(8, num_samples * 0.5)

    fig, ax = plt.subplots(figsize=(fig_width, 6))

    # Vertical stacked bar plot
    df_clusters_comp.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        width=1.0,  # 1.0 = no space between bars
        color=colors,
        edgecolor="white",
        linewidth=0.3,
    )

    # Customize axes
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_xlabel("Donor", fontsize=12)
    ax.set_xticklabels(df_clusters_comp.index, rotation=45, ha="right")
    ax.legend(handles=custom_handles, title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, len(df_clusters_comp.index) - 0.5)

    plt.tight_layout()
    # Save figure
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight", dpi=300)
        plt.close()
    plt.show()


def compute_correlation(
    adata: ad.AnnData, morph_key: str, morph_features: list, metric: str = "spearman"
) -> pd.DataFrame:
    """
    Computes the correlation between gene expression and morphological features.

    Args:
        adata (anndata.AnnData):
            The AnnData object containing the gene expression data in .X
            and morphological data in .obsm.
        morph_key (str):
            The key in adata.obsm where the morphological data is stored.
        morph_features (list):
            A list of strings with the names of the morphological features.
        metric (str, optional):
            The correlation metric to use, either 'spearman' or 'pearson'.
            Defaults to 'spearman'.

    Returns:
        pd.DataFrame:
            A DataFrame containing the correlation matrix with genes as rows
            and morphological features as columns.
    """
    # 1. Filter out cells with any NaN values in the morphology data
    valid_cells_mask = ~np.isnan(adata.obsm[morph_key]).any(axis=1)
    adata_filt = adata[valid_cells_mask, :].copy()

    # 2. Extract gene expression matrix (and convert if sparse)
    X = adata_filt.X.copy()
    if not isinstance(X, np.ndarray):
        X = X.toarray()

    # 3. Extract morphological features matrix
    Y = adata_filt.obsm[morph_key].copy()

    # 4. Apply rank transformation for Spearman correlation
    if metric == "spearman":
        X_processed = rankdata(X, axis=0)
        Y_processed = rankdata(Y, axis=0)
    elif metric == "pearson":
        # For Pearson, we typically standardize the data
        X_processed = StandardScaler().fit_transform(X)
        Y_processed = StandardScaler().fit_transform(Y)
    else:
        raise ValueError(f"Metric '{metric}' is not handled. Use 'spearman' or 'pearson'.")

    # 5. Compute the correlation matrix
    # pairwise_distances with 'correlation' metric computes 1 - Pearson's r.
    # So, we subtract from 1 to get the actual correlation coefficient.
    corr_matrix = 1 - pairwise_distances(X_processed.T, Y_processed.T, metric="correlation")

    # 6. Create a well-labeled DataFrame
    df_corr = pd.DataFrame(corr_matrix, index=adata_filt.var_names, columns=morph_features)

    return df_corr


def correlation_clustermap(
    df_corr: pd.DataFrame,
    output_path: str = None,
    top_n_genes: int = None,
    figsize: tuple = (6, 12),
    cmap: str = "vlag",
    show=True,
    **kwargs,
) -> sns.matrix.ClusterGrid:
    """
    Plots a clustermap of the correlation matrix.

    Args:
        df_corr (pd.DataFrame):
            The correlation matrix (genes x morphology features).
        output_path (str, optional):
            Path to save the figure. If None, the figure is not saved.
            Defaults to None.
        top_n_genes (int, optional):
            If specified, only the top N genes with the highest absolute
            correlation will be plotted. Defaults to None (all genes).
        figsize (tuple, optional):
            The size of the figure. Defaults to (6, 12).
        cmap (str, optional):
            The colormap for the heatmap. Defaults to "vlag".
        **kwargs:
            Additional keyword arguments passed to sns.clustermap.

    Returns:
        sns.matrix.ClusterGrid:
            The ClusterGrid object for further customization if needed.
    """
    # 1. Optionally, filter for the most variable genes
    if top_n_genes:
        top_genes = df_corr.abs().max(axis=1).sort_values(ascending=False).index[:top_n_genes]
        plot_data = df_corr.loc[top_genes]
    else:
        plot_data = df_corr

    # 2. Plot the clustermap
    sns.set(font_scale=0.8)
    cg = sns.clustermap(
        plot_data,
        cmap=cmap,
        center=0,
        linewidths=0.001,
        figsize=figsize,
        row_cluster=True,
        col_cluster=True,
        **kwargs,  # Pass any other clustermap arguments
    )

    # 3. Adjust layout and aesthetics
    cg.ax_row_dendrogram.set_visible(False)
    cg.ax_col_dendrogram.set_visible(False)

    # Reposition colorbar to be less intrusive
    # x0, _y0, _w, _h = cg.cbar_pos
    # cg.ax_cbar.set_position([x0 - 0.1, _y0 - 0.2, cg.ax_row_dendrogram.get_position().width - 0.05, 0.02])
    # cg.ax_cbar.set_title("")
    # cg.ax_cbar.tick_params(axis='x', length=2)

    # 4. Save the figure if a path is provided
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight", dpi=300)
        print(f"Clustermap saved to: {p}")
    if show:
        plt.show()
    else:
        plt.close()
    return cg


def gex_morphology_facet_plot(adata, morph_features, feats, metric="spearman"):
    # Morphology features
    morph_df = pd.DataFrame(adata.obsm["morphology"], index=adata.obs_names, columns=morph_features)

    # Gene expression
    expr_df = pd.DataFrame(adata[:, feats].X.toarray(), columns=feats, index=adata.obs_names)

    # Combine
    plot_df = pd.concat([expr_df, morph_df], axis=1)

    # Grid dimensions
    n_morphs = len(morph_df.columns)
    n_genes = len(feats)

    fig, axes = plt.subplots(
        nrows=n_morphs, ncols=n_genes, figsize=(3 * n_genes, 3 * n_morphs), sharex=False, sharey=False
    )

    # Ensure axes is 2D
    if n_morphs == 1:
        axes = axes[np.newaxis, :]
    if n_genes == 1:
        axes = axes[:, np.newaxis]

    # Plot with regression line and Spearman r
    for i, morph_feat in enumerate(morph_df.columns):  # rows
        for j, gene in enumerate(feats):  # columns
            ax = axes[i, j]

            # Scatter + regression line
            sns.regplot(
                data=plot_df,
                x=gene,
                y=morph_feat,
                scatter_kws={"alpha": 0.6, "s": 15},
                line_kws={"color": "red"},
                ax=ax,
            )

            # Compute Spearman correlation
            rho, pval = spearmanr(plot_df[gene], plot_df[morph_feat])
            ax.text(0.05, 0.9, f"ρ={rho:.2f}", transform=ax.transAxes, fontsize=8, color="blue")

            # Axis labels
            if i == n_morphs - 1:
                ax.set_xlabel(gene)
            else:
                ax.set_xlabel("")
            if j == 0:
                ax.set_ylabel(morph_feat)
            else:
                ax.set_ylabel("")

            ax.tick_params(labelsize=6)

    plt.tight_layout()
    plt.show()


def joint_clustermap(
    mod1: pd.DataFrame,
    mod2: pd.DataFrame,
    cluster_on: str = "joint",
    orientation: str = "horizontal",
    winsorize_limits: tuple = None,
    row_colors: [pd.DataFrame, pd.Series] = None,
    row_colors_palettes: dict = None,
    mod1_cmap: str = "viridis",
    mod2_cmap: str = "plasma",
    mod1_name: str = "Modality 1",
    mod2_name: str = "Modality 2",
    dendrogram_ratio: float = 0.15,
    colors_ratio: float = 0.03,
    cbar1_pos: tuple = (0.02, 0.75, 0.03, 0.20),
    cbar2_pos: tuple = (0.02, 0.50, 0.03, 0.20),
    legend_pos: tuple = (1.02, 0.95),  # NEW: Top-left corner for the first legend
    legend_box_dims: tuple = (0.15, 0.2),  # NEW: (width, height) for each legend box
    title: str = None,
    output_path: str = None,
    **kwargs,
):
    """
    Creates a joint clustermap of two modalities
    """
    if orientation not in ["horizontal", "vertical"]:
        raise ValueError("`orientation` must be 'horizontal' or 'vertical'.")
    if cluster_on not in ["joint_equal_weight", "joint", "mod1", "mod2"]:
        raise ValueError("`cluster_on` must be one of 'joint_equal_weight', 'joint', 'mod1', or 'mod2'.")

    # --- 1. PRE-PROCESSING AND DATA PREPARATION ---
    # A) Optional Winsorizing
    if winsorize_limits:
        mod1, mod2 = mod1.copy(), mod2.copy()
        lower_q, upper_q = winsorize_limits
        for df in [mod1, mod2]:
            for col in df.columns:
                lower_b, upper_b = df[col].quantile(lower_q), df[col].quantile(upper_q)
                df[col] = df[col].clip(lower=lower_b, upper=upper_b)

    # B) Calculate sample linkage
    scaler = StandardScaler()
    mod1_scaled_cluster = scaler.fit_transform(mod1)
    mod2_scaled_cluster = scaler.fit_transform(mod2)

    cluster_data = {
        "joint": np.concatenate([mod1_scaled_cluster, mod2_scaled_cluster], axis=1),
        "mod1": mod1_scaled_cluster,
        "mod2": mod2_scaled_cluster,
    }

    if cluster_on == "joint_equal_weight":
        dist_mod1 = pdist(cluster_data["mod1"], metric=kwargs.get("metric", "euclidean"))
        dist_mod2 = pdist(cluster_data["mod2"], metric=kwargs.get("metric", "euclidean"))
        avg_dist = (dist_mod1 + dist_mod2) / 2.0
        sample_linkage = linkage(avg_dist, method="average")
    else:
        sample_linkage = linkage(
            cluster_data[cluster_on], method=kwargs.get("method", "average"), metric=kwargs.get("metric", "euclidean")
        )

    # C) Map categorical colors and store final palettes for the legend
    mapped_sample_colors = None
    final_palettes = {}
    if row_colors is not None:
        if isinstance(row_colors, pd.Series):
            row_colors = row_colors.to_frame()
        mapped_sample_colors = pd.DataFrame(index=row_colors.index)
        for col in row_colors.columns:
            unique_cats = row_colors[col].dropna().unique()
            palette = (row_colors_palettes or {}).get(
                col, dict(zip(unique_cats, sns.color_palette("tab20", len(unique_cats))))
            )
            final_palettes[col] = palette
            mapped_sample_colors[col] = row_colors[col].map(palette)

    # D) Scale data for visualization
    range1, range2 = (0, 1), (1.5, 2.5)
    scaler1_vis = MinMaxScaler(feature_range=range1)
    scaler2_vis = MinMaxScaler(feature_range=range2)
    mod1_norm = pd.DataFrame(scaler1_vis.fit_transform(mod1), index=mod1.index, columns=mod1.columns)
    mod2_norm = pd.DataFrame(scaler2_vis.fit_transform(mod2), index=mod2.index, columns=mod2.columns)

    # 2. CREATE CUSTOM COLORMAP
    vmin, vmax, spacer_val = 0, range2[1], 1.25
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap1, cmap2 = plt.get_cmap(mod1_cmap), plt.get_cmap(mod2_cmap)
    colors_and_nodes = [
        (norm(range1[0]), cmap1(0.0)),
        (norm(range1[1]), cmap1(1.0)),
        (norm(range1[1]) + 1e-6, "white"),
        (norm(range2[0]) - 1e-6, "white"),
        (norm(range2[0]), cmap2(0.0)),
        (norm(range2[1]), cmap2(1.0)),
    ]
    custom_cmap = LinearSegmentedColormap.from_list("joint_cmap", colors_and_nodes)

    # 3. ASSEMBLE AND PLOT
    spacer = pd.DataFrame(np.full((mod1_norm.shape[0], 1), spacer_val), index=mod1_norm.index, columns=["spacer"])
    if orientation == "horizontal":
        final_df = pd.concat([mod1_norm, spacer, mod2_norm], axis=1)
        plot_args = {
            "row_linkage": sample_linkage,
            "col_cluster": False,
            "row_colors": mapped_sample_colors,
            "dendrogram_ratio": (dendrogram_ratio, 0.01),
            "colors_ratio": (colors_ratio, 0.01),
        }
    else:  # Vertical
        final_df = pd.concat([mod1_norm.T, spacer.T, mod2_norm.T], axis=0)
        plot_args = {
            "col_linkage": sample_linkage,
            "row_cluster": False,
            "col_colors": mapped_sample_colors,
            "dendrogram_ratio": (0.01, dendrogram_ratio),
            "colors_ratio": (0.01, colors_ratio),
        }

    g = sns.clustermap(final_df, cmap=custom_cmap, vmin=vmin, vmax=vmax, cbar_pos=None, **plot_args, **kwargs)

    # --- 4. MANUALLY CREATE AND PLACE ALL EXTERNAL PLOT ELEMENTS ---

    # A) Place Colorbars
    if cbar1_pos:
        ax1 = g.fig.add_axes(cbar1_pos)
        cbar1 = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap1), cax=ax1, orientation="vertical", ticks=[0, 1])
        ax1.set_title(mod1_name, fontsize=8)
        cbar1.ax.tick_params(labelsize=7, pad=2)
    if cbar2_pos:
        ax2 = g.fig.add_axes(cbar2_pos)
        cbar2 = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap2), cax=ax2, orientation="vertical", ticks=[0, 1])
        ax2.set_title(mod2_name, fontsize=8)
        cbar2.ax.tick_params(labelsize=7, pad=2)

    # B) Place Annotation Legends
    if legend_pos and row_colors is not None:
        current_y = legend_pos[1]
        # Create a legend for each annotation column, stacking them vertically
        for col, palette in final_palettes.items():
            box_left = legend_pos[0]
            box_top = current_y
            box_width, box_height = legend_box_dims

            # Create a new, invisible axes for the legend
            leg_ax = g.fig.add_axes([box_left, box_top - box_height, box_width, box_height])
            leg_ax.set_axis_off()

            # Create legend handles and labels
            handles = [Patch(facecolor=color, edgecolor=color) for color in palette.values()]
            labels = list(palette.keys())

            leg_ax.legend(
                handles,
                labels,
                title=col,
                loc="center left",
                handletextpad=0.5,
                labelspacing=0.4,
                frameon=False,
                fontsize=8,
                title_fontsize=9,
            )

            # Update the y-position for the next legend
            current_y -= box_height + 0.02  # Add a small gap

    # --- 5. FINAL CLEANUP ---
    ax_heatmap = g.ax_heatmap
    axis = ax_heatmap.xaxis if orientation == "horizontal" else ax_heatmap.yaxis
    labels = [label.get_text() for label in axis.get_ticklabels()]
    ticks = list(axis.get_ticklocs())
    spacer_idx = labels.index("spacer")
    del ticks[spacer_idx]
    del labels[spacer_idx]
    axis.set_ticks(ticks)
    axis.set_ticklabels(labels)

    ax_heatmap.set_xlabel("")
    ax_heatmap.set_ylabel("")
    if title:
        plt.suptitle(title, y=0.98, fontsize=16)

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight", dpi=300)
        print(f"Clustermap saved to: {p}")
        plt.close()

    plt.show()

    return g, sample_linkage, final_df


def dot_clustermap(
    data,
    dot_size_df=None,
    row_annot=None,
    col_annot=None,
    row_cmap=None,
    col_cmap=None,
    dot_scale=50,
    legend_width=0.8,
    legend_spacing=0.3,
    cbar_params=None,
    size_legend_params=None,
    cbar_pos=(0.55, 0.4),
    size_legend_pos=(0, 0.45),
    show_xticks=True,
    show_yticks=True,
    **kwargs,
):
    """
    Creates a dot plot with clustering, multi-track annotations, labels, and customizable legends.

    Args:
        data (pd.DataFrame): The rectangular data matrix for dot COLOR.
        dot_size_df (pd.DataFrame, optional): A separate rectangular data matrix for
                                           dot SIZE. Must have the same index and
                                           columns as `data`. If None, `data` is used
                                           for both color and size.
        row_annot (pd.Series or pd.DataFrame, optional): Categorical annotations for rows.
        col_annot (pd.Series or pd.DataFrame, optional): Categorical annotations for columns.
        row_cmap (dict, optional): Nested dictionary mapping row annotation values to colors.
        col_cmap (dict, optional): Nested dictionary for column annotation values to colors.
        dot_scale (int, optional): Scaling factor for dot size.
        legend_width (float, optional): Width ratio for the legend column.
        legend_spacing (float, optional): Width of the space between the plot and legends.
        cbar_params (dict, optional): Parameters for the colorbar legend.
        size_legend_params (dict, optional): Parameters for the dot size legend.
        cbar_pos (tuple, optional): (y, height) for the colorbar.
        size_legend_pos (tuple, optional): (y, height) for the size legend.
        **kwargs: Additional keyword arguments passed to seaborn.clustermap.
    """

    # --- 1. Process annotations (unchanged) ---
    def process_annotations(annot, cmap):
        if annot is None:
            return None
        if isinstance(annot, pd.Series):
            annot = annot.to_frame()
        annot_colors = pd.DataFrame(index=annot.index)
        for col_name in annot.columns:
            track_cmap = (cmap or {}).get(col_name, {})
            unique_items = annot[col_name].dropna().unique()
            missing_items = set(unique_items) - set(track_cmap.keys())
            if missing_items:
                palette = sns.color_palette("husl", len(missing_items))
                for item, color in zip(missing_items, palette):
                    track_cmap[item] = color
            annot_colors[col_name] = annot[col_name].map(track_cmap)
        return annot_colors

    row_colors, col_colors = process_annotations(row_annot, row_cmap), process_annotations(col_annot, col_cmap)

    # --- 2. Generate clustermap for ordering (based on `data` for color) ---
    clustergrid = sns.clustermap(data, row_colors=row_colors, col_colors=col_colors, cbar_pos=None, **kwargs)
    plt.close(clustergrid.fig)

    # --- 3. Reorder both color and size DataFrames ---
    row_indices = clustergrid.dendrogram_row.reordered_ind
    col_indices = clustergrid.dendrogram_col.reordered_ind
    reordered_color_data = data.iloc[row_indices, col_indices]

    if dot_size_df is None:
        reordered_size_data = reordered_color_data
    else:
        if not data.index.equals(dot_size_df.index) or not data.columns.equals(dot_size_df.columns):
            raise ValueError("`dot_size_df` must have the same index and columns as `data`.")
        reordered_size_data = dot_size_df.iloc[row_indices, col_indices]

    # --- 4. Dynamically create the figure layout (unchanged) ---
    n_row_tracks = row_colors.shape[1] if row_colors is not None else 0
    n_col_tracks = col_colors.shape[1] if col_colors is not None else 0
    figsize = kwargs.get("figsize", (10, 10))
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        1 + n_col_tracks,
        3 + n_row_tracks,
        height_ratios=[0.2] * n_col_tracks + [5],
        width_ratios=[5] + [0.2] * n_row_tracks + [legend_spacing] + [legend_width],
        wspace=0,
        hspace=0,
    )

    # --- 5. Plot the main dot plot using separate data for size and color ---
    ax_main = fig.add_subplot(gs[n_col_tracks, 0])
    x, y = np.meshgrid(np.arange(reordered_color_data.shape[1]), np.arange(reordered_color_data.shape[0]))

    dot_sizes = reordered_size_data.values.flatten() * dot_scale / reordered_size_data.values.max()

    scatter_artist = ax_main.scatter(
        x=x.flatten(),
        y=y.flatten(),
        s=dot_sizes,
        c=reordered_color_data.values.flatten(),
        cmap=kwargs.get("cmap", "viridis"),
        alpha=0.9,
    )
    if show_xticks:
        ax_main.set_xticks(np.arange(reordered_color_data.shape[1]))
        ax_main.set_xticklabels(reordered_color_data.columns, rotation=90)
    else:
        ax_main.set_xticks([])
    if show_yticks:
        ax_main.set_yticks(np.arange(reordered_color_data.shape[0]))
        ax_main.set_yticklabels(reordered_color_data.index)
    else:
        ax_main.set_yticks([])

    # ax_main.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)
    ax_main.invert_yaxis()
    ax_main.set_xlim(-0.5, reordered_color_data.shape[1] - 0.5)
    ax_main.tick_params(top=False, right=False, left=False, bottom=False)
    [spine.set_visible(False) for spine in ax_main.spines.values()]

    # --- 6. Plot annotation color bars (unchanged) ---
    if n_col_tracks > 0:
        col_colors_reordered = col_colors.iloc[col_indices]
        for i, name in enumerate(col_colors_reordered.columns):
            ax = fig.add_subplot(gs[i, 0], sharex=ax_main)
            rgb = np.array([matplotlib.colors.to_rgb(c) for c in col_colors_reordered[name]])
            ax.imshow(rgb[np.newaxis, :, :], aspect="auto", interpolation="nearest")
            ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
            ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            [s.set_visible(False) for s in ax.spines.values()]
    if n_row_tracks > 0:
        row_colors_reordered = row_colors.iloc[row_indices]
        for i, name in enumerate(row_colors_reordered.columns):
            ax = fig.add_subplot(gs[n_col_tracks, 1 + i], sharey=ax_main)
            rgb = np.array([matplotlib.colors.to_rgb(c) for c in row_colors_reordered[name]])
            ax.imshow(rgb[:, np.newaxis, :], aspect="auto", interpolation="nearest")
            ax.set_title(name, rotation=75, ha="left", va="bottom", fontsize=9)
            ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            [s.set_visible(False) for s in ax.spines.values()]

    # --- 7. Add Legends, with size legend based on `reordered_size_data` ---
    legend_ax_container = fig.add_subplot(gs[n_col_tracks, 2 + n_row_tracks])
    legend_ax_container.axis("off")
    bbox = legend_ax_container.get_position()

    cbar_params = cbar_params or {}
    size_params = size_legend_params or {}

    # Colorbar Legend (based on `data`)
    ax_colorbar = fig.add_axes([bbox.x0, bbox.y0 + bbox.height * cbar_pos[0], bbox.width, bbox.height * cbar_pos[1]])
    cbar = fig.colorbar(scatter_artist, cax=ax_colorbar, orientation="vertical")
    if "ticks" in cbar_params:
        cbar.set_ticks(cbar_params["ticks"])
    elif "nticks" in cbar_params:
        cbar.locator = mticker.MaxNLocator(nbins=cbar_params["nticks"])
        cbar.update_ticks()
    cbar.set_label(cbar_params.get("label", "Color Value"), rotation=-90, va="bottom")

    # Size Legend (based on `dot_size_df`)
    ax_size_legend = fig.add_axes(
        [bbox.x0, bbox.y0 + bbox.height * size_legend_pos[0], bbox.width, bbox.height * size_legend_pos[1]]
    )
    legend_values = size_params.get("values", np.percentile(reordered_size_data.values, [10, 50, 90]))
    legend_labels = size_params.get("labels", [f"{v:.2g}" for v in legend_values])
    legend_title = size_params.get("title", "Size Value")
    sizes = np.array(legend_values) * dot_scale / reordered_size_data.values.max()
    handles = [ax_size_legend.scatter([], [], s=s, color="gray", alpha=0.8) for s in sizes]
    ax_size_legend.legend(
        handles=handles, labels=legend_labels, title=legend_title, frameon=False, labelspacing=1.3, loc="center"
    )
    ax_size_legend.axis("off")

    plt.show()
