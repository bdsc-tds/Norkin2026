from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay, adjusted_rand_score
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors
import anndata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def transfer_labels_knn(
    adata: anndata.AnnData,
    ref_condition: str,
    query_condition: str,
    condition_key: str,
    label_to_transfer: str,
    use_rep: str = None,
    use_graph_key: str = "connectivities",
    n_neighbors: int = 15,
) -> pd.Series:
    """
    Transfers labels from reference to query cells via a k-Nearest Neighbors graph.

    This function is vectorized for high performance in both operating modes.

    Args:
        adata (anndata.AnnData):
            The AnnData object containing all data.
        ref_condition (str):
            The value in `adata.obs[condition_key]` for the reference cells.
        query_condition (str):
            The value in `adata.obs[condition_key]` for the query cells.
        condition_key (str):
            The column name in `adata.obs` that distinguishes query from reference.
        label_to_transfer (str):
            The column name in `adata.obs` with the labels to transfer.
        use_rep (str, optional):
            Key in `adata.obsm` to use for computing kNN on the fly. If set, this
            method is used instead of the pre-computed graph. Defaults to None.
        use_graph_key (str, optional):
            Key in `adata.obsp` for the pre-computed connectivity graph.
            Used only if `use_rep` is None. Defaults to "connectivities".
        n_neighbors (int, optional):
            Number of neighbors for kNN search if `use_rep` is specified. Defaults to 15.

    Returns:
        pd.Series:
            A Series with predicted labels for the query cells, indexed by the
            original cell barcodes/names.
    """
    print(f"--- Transferring '{label_to_transfer}' labels from '{ref_condition}' to '{query_condition}' ---")

    # Create boolean masks for reference and query cells
    ref_mask = adata.obs[condition_key] == ref_condition
    query_mask = adata.obs[condition_key] == query_condition

    # --- MODE 1: Compute kNN on the fly  ---
    if use_rep:
        print(f"Mode: Computing new kNN search using data from .obsm['{use_rep}'] with k={n_neighbors}.")
        if use_rep not in adata.obsm:
            raise KeyError(f"'{use_rep}' not found in adata.obsm.")

        ref_data = adata.obsm[use_rep][ref_mask, :]
        query_data = adata.obsm[use_rep][query_mask, :]

        # Fit kNN model and find neighbors
        nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto").fit(ref_data)
        _, neighbor_local_indices = nn.kneighbors(query_data)

        # Factorize labels to handle string-based mode calculation for modern SciPy
        ref_labels_series = adata.obs[label_to_transfer][ref_mask]
        codes, uniques = pd.factorize(ref_labels_series)
        neighbor_codes = codes[neighbor_local_indices]
        most_common_codes, _ = mode(neighbor_codes, axis=1, keepdims=False)
        most_common_labels = uniques[most_common_codes]

        predicted_labels = pd.Series(most_common_labels, index=adata.obs.index[query_mask])

    # --- MODE 2: Use pre-computed graph  ---
    else:
        print(f"Mode: Using pre-computed graph from .obsp['{use_graph_key}'].")
        if use_graph_key not in adata.obsp:
            raise KeyError(f"'{use_graph_key}' not found in adata.obsp.")

        ref_indices = np.where(ref_mask)[0]
        query_indices = np.where(query_mask)[0]

        query_to_ref_conn = adata.obsp[use_graph_key][query_indices, :][:, ref_indices]
        ref_labels = adata.obs[label_to_transfer][ref_mask]
        ref_labels_ohe = pd.get_dummies(ref_labels)

        unique_labels = ref_labels_ohe.columns.values
        vote_counts = query_to_ref_conn @ ref_labels_ohe.values

        predicted_label_indices = np.argmax(vote_counts, axis=1)
        predicted_label_names = unique_labels[predicted_label_indices].astype(object)

        # Handle query cells that had no reference neighbors by marking them 'Unassigned'
        no_neighbor_mask = query_to_ref_conn.sum(axis=1).A1 == 0
        predicted_label_names[no_neighbor_mask] = "Unassigned"

        predicted_labels = pd.Series(predicted_label_names, index=adata.obs.index[query_mask])

    print("Label transfer complete.\n")
    return predicted_labels


def evaluate_label_transfer(
    true_labels: pd.Series,
    predicted_labels: pd.Series,
    title: str = "Label Transfer Evaluation",
    normalize: str = "true",
    figsize: tuple = (10, 10),
    values_format: str = None,
    text_kw: dict = None,
    save_path: str = None,
    show=True,
):
    """
    Calculates performance metrics and plots a customizable confusion matrix.

    Args:
        true_labels: A pandas Series with the ground truth labels.
        predicted_labels: A pandas Series with the predicted labels.
        title: The title for the plot and output.
        normalize: How to normalize the matrix ('true', 'pred', 'all', or None).
        figsize (tuple, optional):
            The size of the figure (width, height) in inches.
            Increase this for matrices with many labels. Defaults to (10, 10).
        values_format (str, optional):
            The format string for the numbers in the matrix (e.g., '.2f').
            If None, a sensible default is chosen based on `normalize`.
            Defaults to None.
        text_kw (dict, optional):
            A dictionary of keyword arguments to pass to the text objects
            (e.g., {'size': 8} to set the font size). Defaults to None.
    """
    print(f"--- Evaluating Label Transfer: {title} ---")

    # Calculate Adjusted Rand Index
    ari_score = adjusted_rand_score(true_labels, predicted_labels)
    print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")

    # Get a complete list of all possible labels for the matrix axes
    all_labels = sorted(list(set(true_labels.unique()) | set(predicted_labels.unique())))

    # Set a sensible default for number formatting based on normalization
    if values_format is None:
        values_format = ".2f" if normalize in ["true", "pred", "all"] else "d"

    # Set figure size and create the plot axes
    fig, ax = plt.subplots(figsize=figsize)

    # Create the display object
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true=true_labels,
        y_pred=predicted_labels,
        labels=all_labels,
        normalize=normalize,
        cmap="Blues",
        xticks_rotation="vertical",
        ax=ax,
        values_format=values_format,
    )

    # Manually set text properties after the plot is drawn.
    if text_kw:
        for text_array in disp.text_:
            for text_obj in text_array:
                text_obj.set(**text_kw)

    ax.set_title(title)

    # Use tight_layout to help prevent labels from overlapping the figure edge
    plt.tight_layout()
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return disp


def find_neighbor_same_donor(
    adata: anndata.AnnData,
    source_cond: str,
    target_cond: str,
    donor_key: str = "donor",
    condition_key: str = "condition",
    use_rep: str = "X_pca",
) -> None:
    """
    Finds nearest neighbors and adds their info directly to adata.obs using iloc.

    For each cell in a `source_cond`, this function finds its nearest neighbor
    from the same donor in the `target_cond`. It then adds the neighbor's
    `obs_name`, its integer index in the AnnData object, and the distance
    as new columns to `adata.obs`. The function modifies the AnnData object in-place
    and uses integer-location based assignment (`iloc`) to be robust to duplicate index labels.

    Args:
        adata (anndata.AnnData):
            The AnnData object to be modified.
        source_cond (str):
            The condition of the cells to find neighbors FOR (e.g., 'treated').
        target_cond (str):
            The condition to search for neighbors WITHIN (e.g., 'control').
        donor_key (str, optional):
            The key in `adata.obs` for donor identifiers. Defaults to 'donor'.
        condition_key (str, optional):
            The key in `adata.obs` for condition labels. Defaults to 'condition'.
        use_rep (str, optional):
            The key in `adata.obsm` for the PCA embedding. Defaults to 'X_pca'.

    Returns:
        None: The function modifies `adata.obs` in-place.
    """
    # 1. Input validation
    if adata.obs_names.nunique() < len(adata.obs_names):
        raise ValueError("adata.obs_names contains duplicate cell IDs.")
    if use_rep not in adata.obsm:
        raise KeyError(f"PCA embedding not found in adata.obsm['{use_rep}'].")
    for key in [donor_key, condition_key]:
        if key not in adata.obs.columns:
            raise KeyError(f"Metadata column '{key}' not found in adata.obs.")

    all_conditions = set(adata.obs[condition_key].unique())
    if source_cond not in all_conditions:
        raise ValueError(f"Source condition '{source_cond}' not found in adata.obs['{condition_key}'].")
    if target_cond not in all_conditions:
        raise ValueError(f"Target condition '{target_cond}' not found in adata.obs['{condition_key}'].")
    if source_cond == target_cond:
        raise ValueError("`source_cond` and `target_cond` cannot be the same.")

    # 2. Initialize new columns in adata.obs
    col_name_id = f"neighbor_{target_cond}_obs_name"
    col_name_idx = f"neighbor_{target_cond}_index"
    col_name_dist = f"neighbor_{target_cond}_dist"

    # Initialize with arrays of null values matching the length of adata.obs
    adata.obs[col_name_id] = np.full(adata.n_obs, pd.NA, dtype=object)
    adata.obs[col_name_idx] = np.full(adata.n_obs, pd.NA, dtype=object)
    adata.obs[col_name_dist] = np.full(adata.n_obs, np.nan, dtype=float)

    # Get the integer positions of the new columns for efficient iloc assignment
    col_id_pos = adata.obs.columns.get_loc(col_name_id)
    col_idx_pos = adata.obs.columns.get_loc(col_name_idx)
    col_dist_pos = adata.obs.columns.get_loc(col_name_dist)

    # Create a mapping from cell ID (obs_name) to its integer position
    obs_name_to_pos = {name: i for i, name in enumerate(adata.obs_names)}

    donors = adata.obs[donor_key].unique()
    print(f"Searching for neighbors for cells in '{source_cond}' within '{target_cond}'.")
    print(f"Found {len(donors)} unique donors to process.")

    # 3. Iterate over each donor
    for i, donor in enumerate(donors):
        print(f"Processing donor {i + 1}/{len(donors)}: {donor}", end="\r")

        # Create boolean masks for the current donor in each condition
        donor_mask = adata.obs[donor_key] == donor
        source_mask = donor_mask & (adata.obs[condition_key] == source_cond)
        target_mask = donor_mask & (adata.obs[condition_key] == target_cond)

        # Skip if donor is not present in both conditions
        if not np.any(source_mask) or not np.any(target_mask):
            print(f"\t{donor} not found in both conditions")
            continue

        # Get data and integer positions using the masks
        pca_source = adata.obsm[use_rep][source_mask]
        pca_target = adata.obsm[use_rep][target_mask]
        target_obs_names = adata.obs_names[target_mask]

        # This is the key: get the integer positions of the source cells
        source_ilocs = np.where(source_mask)[0]

        # Fit KNN on the target cells and query with the source cells
        distances, indices = NearestNeighbors(n_neighbors=1).fit(pca_target).kneighbors(pca_source)

        # Map the resulting indices back to the original cell IDs and integer positions
        neighbor_obs_names = target_obs_names[indices.flatten()]
        neighbor_indices = [obs_name_to_pos[name] for name in neighbor_obs_names]

        # --- SOLUTION: Assign results using iloc ---
        # This is robust to any duplicate labels in adata.obs_names
        adata.obs.iloc[source_ilocs, col_id_pos] = neighbor_obs_names
        adata.obs.iloc[source_ilocs, col_idx_pos] = neighbor_indices
        adata.obs.iloc[source_ilocs, col_dist_pos] = distances.flatten()

    print("\nFinished processing all donors.")

    # Convert index column to nullable integer type to handle NaNs cleanly
    adata.obs[col_name_idx] = adata.obs[col_name_idx].astype("Int64")

    print(f"Successfully added columns: '{col_name_id}', '{col_name_idx}', '{col_name_dist}' to adata.obs.")
