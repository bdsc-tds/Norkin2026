import os
import sys
import scanpy as sc
import anndata2ri
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
import numpy as np

try:
    import readwrite
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts")))
    import readwrite


def subsample(adata, obs_key, n_obs, random_state=0, copy=True):
    """
    Subsamples each category in adata.obs[obs_key] to a maximum of n_obs cells.

    Args:
        adata (anndata.AnnData): The AnnData object, can be in-memory or backed.
        obs_key (str): The column in adata.obs to group by.
        n_obs (int): The maximum number of cells per category.
        random_state (int): Seed for the random number generator.
        copy (bool): If True, returns a new in-memory AnnData object.

    Returns:
        anndata.AnnData: The subsampled AnnData object.
    """
    rs = np.random.RandomState(random_state)
    counts = adata.obs[obs_key].value_counts()

    indices = []
    print(f"Subsampling groups in '{obs_key}' to max {n_obs} cells each...")
    for group, count in counts.items():
        # Get indices for the current group
        # Using .values is safer for potentially large boolean arrays
        group_indices = np.where(adata.obs[obs_key].values == group)[0]

        if count <= n_obs:
            print(f"  - Group '{group}': Keeping all {count} cells.")
            indices.extend(group_indices)
        else:
            print(f"  - Group '{group}': Subsampling {count} -> {n_obs} cells.")
            indices.extend(rs.choice(group_indices, size=n_obs, replace=False))

    # Subsetting a backed object and then calling .copy() creates a new
    # in-memory object with only the required data. This is memory-efficient.
    if copy:
        return adata[indices].copy()
    else:
        return adata[indices]


def convert_h5ad_to_seurat_rds(input_h5ad, output_rds):
    """
    Converts a large .h5ad file to a subsampled Seurat .rds file.
    """
    # --- 1. Load data in a memory-efficient way ---
    print(f"Reading anndata from: {input_h5ad}")
    # backed='r' memory-maps the file instead of loading it all into RAM.
    adata_backed = sc.read_h5ad(input_h5ad)
    print(f"Full object dimensions: {adata_backed.n_obs} obs × {adata_backed.n_vars} vars")

    # --- 2. Subsample the data correctly ---
    # This creates an IN-MEMORY object from the backed object.
    # Subsampling to 10k since this is RCTD default maximum
    adata_sub = subsample(adata_backed, obs_key="cell_type", n_obs=10_000, random_state=0)
    print(f"Subsampled object dimensions: {adata_sub.n_obs} obs × {adata_sub.n_vars} vars")

    # The large backed object is no longer needed
    del adata_backed

    # --- 3. Convert to Seurat and Save ---
    # Import required R packages using rpy2
    base = importr("base")
    seurat = importr("Seurat")
    # The as.Seurat function is in SeuratObject
    seurat_object_pkg = importr("SeuratObject")

    # Activate the automatic conversion between AnnData and SingleCellExperiment
    anndata2ri.activate()

    print("\nStarting conversion to R objects...")
    with localconverter(ro.default_converter + anndata2ri.converter):
        # Step 3a: Convert Python AnnData to R SingleCellExperiment (SCE)
        # This is the default behavior of anndata2ri
        print("Converting AnnData to R SingleCellExperiment...")
        r_sce_object = ro.conversion.py2rpy(adata_sub)

        # Step 3b: Explicitly convert the SCE to a Seurat object
        # This is the crucial step to get a real Seurat object
        print("Converting SingleCellExperiment to Seurat object...")
        # We specify that the 'X' matrix in AnnData should become the 'counts' slot in Seurat.
        # We set data=None to prevent Seurat from automatically creating a normalized 'data' slot.
        r_seurat_object = seurat_object_pkg.as_Seurat(r_sce_object, counts="X", data=ro.r("NULL"))

    print(f"Saving Seurat object to: {output_rds}")
    # Save the final Seurat R object to an .rds file
    base.saveRDS(r_seurat_object, file=output_rds)

    # Deactivate the converter
    anndata2ri.deactivate()

    print(f"✅ Conversion complete: {output_rds}")


if __name__ == "__main__":
    # Get configuration and define paths
    cfg = readwrite.config()
    input_path = os.path.join(cfg["scrnaseq_references_dir"], "Marteau2024/core_atlas-adata.h5ad")
    output_path = cfg["scrnaseq_reference_marteau_seurat"]

    # Run the conversion
    convert_h5ad_to_seurat_rds(input_path, output_path)
