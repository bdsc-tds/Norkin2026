# Load required libraries
library(Seurat)
library(Matrix)
library(dplyr)

setwd(this.path::here()) # Uncomment if you need to set the working directory
source('../../scripts/readwrite.R') # Uncomment if you use these helper functions
cfg <- config() # Uncomment if you use a config file
references_dir <- cfg$scrnaseq_references_dir # Example: set path manually if not using config

# Convert GEO_GSE236581 to Seurat, including metadata
convert_gse236581 <- function() {
  geo_dir <- file.path(references_dir, "GEO_GSE236581")
  output_rds <- file.path(geo_dir, "GSE236581_seurat.rds")

  cat("Converting GEO_GSE236581 to Seurat...\n")

  counts <- ReadMtx(
    mtx = file.path(geo_dir, "GSE236581_counts.mtx.gz"),
    features = file.path(geo_dir, "GSE236581_features.tsv.gz"),
    cells = file.path(geo_dir, "GSE236581_barcodes.tsv.gz")
  )

  seurat_obj <- CreateSeuratObject(counts = counts)

  # --- Add Metadata (Corrected Method) ---
  cat("Reading and adding metadata ...\n")
  metadata_path <- file.path(geo_dir, "GSE236581_CRC-ICB_metadata.txt.gz")

  # Step 1: Read the tab-delimited file without any special arguments.
  # This is safer than relying on row.names=1.
  metadata <- read.table(gzfile(metadata_path), header = TRUE, sep = "", fill = TRUE, stringsAsFactors = FALSE)

  # --- Verification Step (Highly Recommended) ---
  # Check if the cell names in the Seurat object match the metadata row names.
  seurat_cells <- colnames(seurat_obj)
  metadata_cells <- rownames(metadata)

  cat("Verifying cell name alignment...\n")
  cat("Number of cells in Seurat object:", length(seurat_cells), "\n")
  cat("Number of cells in metadata file:", length(metadata_cells), "\n")
  cat("Number of matching cells:", sum(seurat_cells %in% metadata_cells), "\n")

  if (length(seurat_cells) != sum(seurat_cells %in% metadata_cells)) {
    warning(
      "Warning: Not all cells in the Seurat object have corresponding metadata!"
    )
  }

  # Step 4: Add metadata to the Seurat object.
  # Seurat will now correctly align the data based on the matching names.
  seurat_obj <- AddMetaData(seurat_obj, metadata)
  cat("Added metadata to Seurat object.\n")

  # Check the result
  # head(seurat_obj@meta.data)

  saveRDS(seurat_obj, output_rds)
  cat("Saved GEO Seurat object to:", output_rds, "\n")
}


convert_gse178341 <- function() {
  # Define paths for the GSE178341 dataset
  geo_dir <- file.path(references_dir, "GEO_GSE178341")
  h5_path <- file.path(geo_dir, "GSE178341_crc10x_full_c295v4_submit.h5")
  cluster_path <- file.path(
    geo_dir,
    "GSE178341_crc10x_full_c295v4_submit_cluster.csv.gz"
  )
  meta_path <- file.path(
    geo_dir,
    "GSE178341_crc10x_full_c295v4_submit_metatables.csv.gz"
  )
  output_rds <- file.path(geo_dir, "GSE178341_seurat.rds")

  cat("Converting GEO GSE178341 to Seurat...\n")

  # --- Step 1: Load the expression data directly using Read10X_h5 ---
  cat("Reading counts matrix from H5 file using Seurat::Read10X_h5...\n")
  counts <- Read10X_h5(h5_path)

  cat("Successfully loaded counts matrix.\n")
  seurat_obj <- CreateSeuratObject(counts = counts)

  # --- Step 2: Read and merge the metadata files using base R ---
  cat("Reading metadata from .csv.gz files using base R's read.csv()...\n")

  # Use base R's read.csv(), which handles .gz files automatically.
  cluster_df <- read.csv(cluster_path)
  meta_df <- read.csv(meta_path)

  colnames(cluster_df)[1] <- "NAME"
  colnames(meta_df)[1] <- "NAME"

  # Merge the two metadata tables using a full join from dplyr
  full_metadata <- full_join(cluster_df, meta_df, by = "NAME")

  # --- Step 3: Add the merged metadata to the Seurat object (No changes here) ---
  rownames(full_metadata) <- full_metadata$NAME
  full_metadata$NAME <- NULL
  seurat_obj <- AddMetaData(seurat_obj, metadata = full_metadata)
  cat("Successfully added merged metadata to the Seurat object.\n")

  # --- Step 4: Save the final object (No changes here) ---
  saveRDS(seurat_obj, output_rds)
  cat("Saved final Seurat object to:", output_rds, "\n")
}
# Run both conversions
convert_gse236581()
# convert_gse178341()
