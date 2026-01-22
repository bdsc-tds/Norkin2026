from skbio.stats import composition
import pandas as pd
import numpy as np
import sklearn

# def sparse_to_knn(A):
#     """
#     Converts a sparse adjacency matrix into a list of NumPy arrays.
#     """
#     knnidx = A.tolil().rows
#     return knnidx


def sparse_to_knn(A):
    # Extract row (samples), col (indices of neighbors), and data (distances)
    row_indices = A.indices
    row_ptrs = A.indptr
    distances = A.data

    # Prepare empty lists for knnidx and knndist
    knnidx = []
    knndist = []

    # Loop over each row (sample) to extract the neighbor indices and distances
    for i in range(A.shape[0]):
        start, end = row_ptrs[i], row_ptrs[i + 1]
        knnidx.append(row_indices[start:end])  # Neighbor indices for i-th sample
        knndist.append(distances[start:end])  # Corresponding distances for i-th sample

    # Convert lists to numpy arrays (optional)
    knnidx = np.array(knnidx, dtype=object)
    knndist = np.array(knndist, dtype=object)

    return knnidx, knndist


def get_knn_labels(knnidx, dummy_array):
    # Convert df_dummies to a numpy array for efficient indexing
    if not isinstance(dummy_array, np.ndarray):
        dummy_array = np.array(dummy_array)

    if knnidx.ndim == 2:
        # knnidx contains same length list of neighbor indices for each sample
        knnlabels = dummy_array[knnidx].sum(1)
    else:
        # Initialize an empty list to store the summed labels
        knnlabels = []

        # Loop over each row in knnidx
        for neighbors in knnidx:
            # Get the one-hot encoded labels for the current neighbors
            neighbor_labels = dummy_array[neighbors]

            # Sum the labels across the neighbors (axis=0 sums column-wise)
            summed_labels = neighbor_labels.sum(axis=0)

            # Append the summed labels to the list
            knnlabels.append(summed_labels)

        # Convert the list back to a numpy array (optional)
        knnlabels = np.array(knnlabels)

    return knnlabels


def _helmert_contrast(n):
    """Helmert contrasts.
    This is equivalent to R's `contr.helmert`.
    """
    contr = np.zeros((n, n - 1))
    contr[1:][np.diag_indices(n - 1)] = np.arange(1, n)
    contr[np.triu_indices(n - 1)] = -1
    return contr


def ilr(x, p=0):
    """
    performs an isometric log-ratio transformation on the input data x.
    It first takes the natural logarithm of x, then applies a Box-Cox transformation if the parameter p is not equal to 0.
    After that, it re-centers the values and applies a Helmert contrast transformation.
    Finally, it returns the dot product of the transformed y and the transpose of the Helmert transformation matrix.
    https://stats.stackexchange.com/questions/259208/how-to-perform-isometric-log-ratio-transformation
    """
    y = np.log(x)
    if p != 0:
        y = (np.exp(p * y) - 1) / p  # Box-Cox transformation
    y -= np.mean(y, axis=1, keepdims=True)  # Recentered values
    k = y.shape[1]
    H = _helmert_contrast(k)
    H = H.T / np.sqrt(np.arange(2, k + 1) * np.arange(1, k)).reshape(-1, 1)

    return np.dot(y, H.T)


# def ilr(X_composition):
#     X_nz = composition.multi_replace(X_composition)
#     X_ilr = ilr(X_nz)
#     return X_ilr


def get_ilr(
    adata,
    radius=20,
    p=0.0,
    label_key="cell_type",
    knn_key="X_spatial",
    knnidx=None,
    min_neighbors=5,
):
    """
    Generate the isometric log ratio (ILR) transformation.

    Args:
        adata: Anndata object containing spatial transcriptomics data coordinates in .obsm['X_spatial'].
        n_neighbors: Number of nearest neighbors to consider for the ILR transformation (default is 20).
        p: Parameter for the ILR Box-Cox transformation (default is 0.5).
        label_key: Key in .obs containing labels.
        knn_key: Key in .obsm containing data coordinates to compute neighbors (default is 'X_spatial').

    Returns:
        knnlabels: DataFrame containing the ILR-transformed spatial transcriptomics data.
        adata.obsm['X_ilr'] : ILR-transformed spatial transcriptomics data.
        adata.obsm['X_ilr_pca'] : PCA-transformed ILR-transformed spatial transcriptomics data.
        adata.uns['ilr_pca'] : PCA attributes.
    """

    if knnidx is None:
        knndis, knnidx = sklearn.neighbors.NearestNeighbors(radius=radius).fit(adata.obsm[knn_key]).radius_neighbors()
    df_dummies = pd.get_dummies(adata.obs[label_key])
    adata.obsm["X_knnlabels"] = get_knn_labels(knnidx, df_dummies)
    if min_neighbors is not None:
        adata = adata[adata.obsm["X_knnlabels"].sum(1) >= min_neighbors].copy()

    adata.obsm["X_composition"] = adata.obsm["X_knnlabels"] / adata.obsm["X_knnlabels"].sum(1, keepdims=1)

    adata.obsm["X_composition_nz"] = composition.multi_replace(adata.obsm["X_composition"])

    adata.obsm["X_ilr"] = ilr(adata.obsm["X_composition_nz"], p=p)

    pca = sklearn.decomposition.PCA().fit(adata.obsm["X_ilr"])
    adata.obsm["X_ilr_pca"] = pca.transform(adata.obsm["X_ilr"])
    adata.uns["ilr_pca"] = pca.__dict__
    adata.uns["X_knnlabels_columns"] = df_dummies.columns
    return adata
