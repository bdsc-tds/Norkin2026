import dask

dask.config.set({"dataframe.query-planning": False})

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import squidpy as sq
import cellcharter as cc
import matplotlib.pyplot as plt

import sys

sys.path.append("workflow/scripts/")
from readwrite import discover_xenium_paths, read_count_correction_samples, read_xenium_samples


def main(args):
    """
    Main function to run the CellCharter pipeline.
    """
    print("Arguments:", args)

    # --- 1. Define Paths and Parameters ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output file paths based on the output directory
    output_labels_path = output_dir / "labels.parquet"
    output_X_scvi_path = output_dir / "X_scvi.parquet"
    output_X_cellcharter_path = output_dir / "X_cellcharter.parquet"
    output_scvi_model_path = output_dir / "scvi_model"
    output_cellcharter_models_path = output_dir / "cellcharter_models"
    output_plot_path = output_dir / "autok_stability.png"

    # Fixed parameters from the original script
    xenium_levels = ["segmentation", "condition", "panel", "donor", "sample"]
    batch_key = "dataset_id"
    spatial_key = "spatial"
    n_clusters_range = (5, 19)
    max_runs = 10
    convergence_tol = 0.001

    # --- 2. Discover and Read Data ---
    print("Reading data...")

    # Apply filters based on CLI arguments
    segmentations_filter = [args.segmentation]
    conditions_filter = [args.condition] if args.condition != "all" else None
    panels_filter = [args.panel] if args.panel != "all" else None

    # Discover Xenium sample paths
    xenium_paths, _ = discover_xenium_paths(
        analysis_dir=Path(args.seurat_analysis_dir),
        data_dir=Path(args.xenium_dir),
        annotation_dir=Path(args.cell_type_annotation_dir),
        correction_dir=Path(args.count_correction_dir),
        normalisation=args.normalisation,
        reference=args.reference,
        method=args.method,
        level=args.level,
        correction_methods_filter=[args.correction_method],
        segmentations_filter=segmentations_filter,
        conditions_filter=conditions_filter,
        panels_filter=panels_filter,
    )

    if args.correction_method != "raw":
        ads = read_count_correction_samples(xenium_paths, [args.correction_method])
    else:
        ads = {
            "raw": read_xenium_samples(
                xenium_paths["raw"], anndata=True, pool_mode="thread", max_workers=args.max_workers
            )
        }

    # Concatenate samples into a single AnnData object
    adata = sc.concat({k: v for k, v in ads[args.correction_method].items()}, label="dataset_id", join="inner")
    adata.obs[xenium_levels] = pd.DataFrame(
        adata.obs["dataset_id"].tolist(), index=adata.obs.index, columns=xenium_levels
    )
    adata.obs["correction_method"] = args.correction_method

    assert "spatial" in adata.obsm

    # --- 3. Preprocess Data ---
    print("Preprocessing data...")
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    # --- 4. Run CellCharter Pipeline ---
    print("Setting up AnnData for SCVI...")
    if conditions_filter is None:
        scvi.model.SCVI.setup_anndata(
            adata, layer="counts", batch_key="condition", categorical_covariate_keys=[batch_key]
        )

    else:
        scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=batch_key)

    print("Training SCVI model...")
    model = scvi.model.SCVI(adata)
    model.train(early_stopping=True, enable_progress_bar=True)
    adata.obsm["X_scVI"] = model.get_latent_representation(adata).astype(np.float32)

    print("Building spatial graph and aggregating neighbors...")
    sq.gr.spatial_neighbors(
        adata, library_key=batch_key, coord_type="generic", delaunay=True, spatial_key=spatial_key, percentile=99
    )
    cc.gr.aggregate_neighbors(adata, n_layers=3, use_rep="X_scVI", out_key="X_cellcharter", sample_key=batch_key)

    print("Clustering with AutoK...")
    autok = cc.tl.ClusterAutoK(n_clusters=n_clusters_range, max_runs=max_runs, convergence_tol=convergence_tol)
    autok.fit(adata, use_rep="X_cellcharter")

    # --- 5. Store Results and Save Outputs ---
    print("Storing results and saving outputs...")
    df_labels = adata.obs[xenium_levels + ["correction_method"]].copy()
    for k in np.arange(n_clusters_range[0] - 1, n_clusters_range[1] + 2):
        df_labels[k] = autok.predict(adata, use_rep="X_cellcharter", k=k).tolist()

    # Save outputs to the specified paths
    pd.DataFrame(adata.obsm["X_scVI"], index=adata.obs_names).to_parquet(output_X_scvi_path)
    pd.DataFrame(adata.obsm["X_cellcharter"], index=adata.obs_names).to_parquet(output_X_cellcharter_path)
    df_labels.to_parquet(output_labels_path)
    model.save(str(output_scvi_model_path))
    autok.save(str(output_cellcharter_models_path))

    ax = cc.pl.autok_stability(autok, return_ax=True)
    plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Successfully completed. Outputs are saved in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CellCharter niche identification pipeline on Xenium data.")

    # Input data directories
    parser.add_argument("--xenium-dir", required=True, help="Path to the Xenium processed data directory.")
    parser.add_argument("--count-correction-dir", required=True, help="Path to the Xenium count correction directory.")
    parser.add_argument("--seurat-analysis-dir", required=True, help="Path to the standard Seurat analysis directory.")
    parser.add_argument("--cell-type-annotation-dir", required=True, help="Path to the cell type annotation directory.")

    # Output directory
    parser.add_argument("--output-dir", required=True, help="Directory to save all outputs (models, labels, plots).")

    # Key analysis parameters
    parser.add_argument("--correction-method", default="raw", help="Count correction method to use.")
    parser.add_argument("--segmentation", default="proseg_expected", help="Segmentation method to filter by.")
    parser.add_argument("--condition", default="CRC", help="Condition to filter by (e.g., CRC).")
    parser.add_argument("--panel", default="hImmune_v1_mm", help="Gene panel to filter by.")

    # Other parameters from the original script
    parser.add_argument("--normalisation", default="lognorm", help="Normalisation method used for annotations.")
    parser.add_argument("--reference", default="GEO_GSE178341", help="Reference dataset for annotations.")
    parser.add_argument("--method", default="rctd_class_aware", help="Annotation method.")
    parser.add_argument("--level", default="Level1", help="Annotation level.")

    # Technical parameters
    parser.add_argument("--max-workers", type=int, default=6, help="Maximum number of worker threads for data reading.")

    args = parser.parse_args()
    main(args)
