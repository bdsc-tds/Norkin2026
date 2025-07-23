#!/usr/bin/env Rscript

# Description:
# This script finds and merges Seurat objects (preprocessed_seurat.rds) for a
# SPECIFIC segmentation and condition pair within a base directory.
#
# Directory Structure Expectation:
# base_dir/{segmentation}/{condition}/{panel}/{donor}/{sample}/.../preprocessed_seurat.rds
#

# --- 1. Load Libraries ---
suppressPackageStartupMessages(library(argparse))
suppressPackageStartupMessages(library(Seurat))
suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tibble))
suppressPackageStartupMessages(library(purrr))
suppressPackageStartupMessages(library(fs))
suppressPackageStartupMessages(library(arrow))


# --- 2. Define and Parse Command-Line Arguments ---
parser <- ArgumentParser(
    description = "Concatenate Seurat objects for a specific segmentation and condition."
)

parser$add_argument(
    "-i",
    "--input-dir",
    type = "character",
    required = TRUE,
    help = "Path to the base directory (e.g., XENIUM_PROCESSED_DIR)."
)

parser$add_argument(
    "-r",
    "--results-dir",
    type = "character",
    required = TRUE,
    help = "Path to the results directory (e.g., RESULTS_DIR)."
)

parser$add_argument(
    "-o",
    "--output-file",
    type = "character",
    required = TRUE,
    help = "Path for the final merged RDS file."
)

parser$add_argument(
    "-s",
    "--segmentation",
    type = "character",
    required = TRUE,
    help = "The specific segmentation method to process (e.g., '10x_0um')."
)

parser$add_argument(
    "-c",
    "--condition",
    type = "character",
    required = TRUE,
    help = "The specific condition to process (e.g., 'CRC')."
)

parser$add_argument(
    "-p",
    "--panel",
    type = "character",
    required = TRUE,
    help = "The specific panel to process (e.g., 'hImmune_v1_mm')."
)

parser$add_argument(
    "-n",
    "--normalisation",
    type = "character",
    required = TRUE,
    help = "The specific normalisation to process (e.g., 'lognorm').",
)

parser$add_argument(
    "-t",
    "--pattern",
    type = "character",
    default = "preprocessed/preprocessed_seurat.rds",
    help = "The file name pattern to search for recursively."
)

args <- parser$parse_args()

# args <- list(
#     input_dir = "/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/norkin_organoid/data/xenium/processed/std_seurat_analysis/",
#     results_dir = "/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/norkin_organoid/results/xenium/embed_and_cluster_panel/raw",
#     output_file = "test.rds",
#     segmentation = "10x_0um",
#     condition = "CRC_PDO",
#     panel = "hImmune_v1_dapi",
#     pattern = "preprocessed/preprocessed_seurat.rds",
#     normalisation = "lognorm"
# )

# --- 3. Main Script Logic ---
cat("=================================================================\n")
cat(sprintf(
    "Starting Seurat Concatenation for:\n -> Segmentation: %s\n -> Condition:    %s\n",
    args$segmentation,
    args$condition,
    args$panel,
    args$normalisation
))
cat("=================================================================\n")

# --- Construct the specific search path and find files ---
# This makes the search much more efficient and targeted
search_path <- file.path(
    args$input_dir,
    args$segmentation,
    args$condition,
    args$panel
)

cat(sprintf("Searching for files in: %s\n", search_path))

if (!dir_exists(search_path)) {
    stop(sprintf(
        "Error: The specified search path does not exist: %s",
        search_path
    ))
}

all_files <- list.files(
    path = search_path,
    recursive = TRUE,
    full.names = TRUE
)

# Filter to ensure the full path suffix matches exactly
seurat_files <- seurat_files[str_ends(
    seurat_files,
    file.path(args$normalisation, args$pattern)
)]

if (length(seurat_files) == 0) {
    stop(sprintf(
        "Error: No files found for segmentation '%s' and condition '%s' matching the pattern '%s'.",
        args$segmentation,
        args$condition,
        args$panel,
        file.path(args$normalisation, args$pattern)
    ))
}

cat(sprintf("Found %d Seurat object(s) to merge.\n\n", length(seurat_files)))

# --- Process and load each Seurat object ---
seurat_list <- list()
dataset_ids <- c()

# Use the base input directory for creating relative paths for metadata
base_path <- path_norm(args$input_dir)

for (file_path in seurat_files) {
    cat(sprintf("Processing: %s\n", file_path))

    # a. Extract metadata from the file path
    relative_path <- path_rel(file_path, start = base_path)
    path_components <- str_split(relative_path, "/")[[1]]

    # b. Assign metadata and create a unique ID
    segmentation_val <- path_components[1]
    condition_val <- path_components[2]
    panel_val <- path_components[3]
    donor_val <- path_components[4]
    sample_val <- path_components[5]

    dataset_id <- paste(
        segmentation_val,
        condition_val,
        panel_val,
        donor_val,
        sample_val,
        sep = "_"
    )
    dataset_ids <- c(dataset_ids, dataset_id)
    cat(sprintf("  -> Extracted Dataset ID: %s\n", dataset_id))

    # c. Load Seurat object and add metadata
    seurat_obj <- readRDS(file_path)
    seurat_obj$cell_id <- colnames(seurat_obj)
    seurat_obj$segmentation <- segmentation_val
    seurat_obj$condition <- condition_val
    seurat_obj$panel <- panel_val
    seurat_obj$donor <- donor_val
    seurat_obj$sample <- sample_val
    seurat_obj$dataset_id <- dataset_id

    seurat_list <- c(seurat_list, seurat_obj)
}

# --- Merge all Seurat objects ---
if (length(seurat_list) > 1) {
    cat("\nMerging all processed Seurat objects...\n")
    cat("This may take a while depending on data size.\n")

    merged_seurat <- merge(
        x = seurat_list[[1]],
        y = seurat_list[2:length(seurat_list)],
        add.cell.ids = dataset_ids,
        project = paste(args$segmentation, args$condition, sep = "_"),
    )
} else if (length(seurat_list) == 1) {
    cat(
        "\nOnly one object found. No merging needed. Saving the single annotated object.\n"
    )
    merged_seurat <- seurat_list[[1]]
    # Set project name for consistency
    Project(merged_seurat) <- paste(
        args$segmentation,
        args$condition,
        sep = "_"
    )
} else {
    stop("Error: No valid Seurat objects could be loaded. Exiting.")
}

cat(sprintf(
    "\nMerged object contains %d cells and %d features.\n",
    ncol(merged_seurat),
    nrow(merged_seurat)
))


# --- load and combine parquet files ---
cat("--- Loading and combining metadata ---\n")

# Define the file paths and prefixes
base_results_path <- file.path(
    args$results_dir,
    args$segmentation,
    args$condition,
    args$panel,
    args$normalisation
)
raw_file_path <- file.path(
    base_results_path,
    "umap_data_n_comps=50_n_neighbors=50_min_dist=0.5_metric=euclidean.parquet"
)
scaled_file_path <- file.path(
    base_results_path,
    "umap_scale_data_n_comps=50_n_neighbors=50_min_dist=0.5_metric=euclidean.parquet"
)

raw_prefix <- paste0(args$normalisation, "_")
scaled_prefix <- paste0(args$normalisation, "_scaled_")

# Create a list to hold the dataframes we successfully load
df_list <- list()

# Load RAW data if it exists
if (file.exists(raw_file_path)) {
    cat(sprintf("Found and loading data: %s\n", basename(raw_file_path)))
    df_list$raw <- read_parquet(raw_file_path) %>%
        # tibble::rownames_to_column(var = "cell") %>% # Convert pandas index to 'cell' column
        rename_with(
            ~ paste0(raw_prefix, .x),
            .cols = -c(cell, segmentation, condition, panel, donor, sample)
        ) # Prefix all columns except 'cell'
} else {
    cat("INFO: Raw data file not found. Skipping.\n")
}

# Load SCALED data if it exists
if (file.exists(scaled_file_path)) {
    cat(sprintf(
        "Found and loading scaled data: %s\n",
        basename(scaled_file_path)
    ))
    df_list$scaled <- read_parquet(scaled_file_path) %>%
        # tibble::rownames_to_column(var = "cell") %>% # Convert pandas index to 'cell' column
        rename_with(
            ~ paste0(scaled_prefix, .x),
            .cols = -c(cell, segmentation, condition, panel, donor, sample)
        ) # Prefix all columns except 'cell'
} else {
    cat("INFO: Scaled data file not found. Skipping.\n")
}

# Check if any data was loaded
if (length(df_list) == 0) {
    stop(
        "FATAL: Neither raw nor scaled metadata files were found. Cannot proceed."
    )
}

# Combine the loaded dataframes into one. This handles 1 or 2 dataframes gracefully.
combined_df <- full_join(
    df_list[[1]],
    df_list[[2]],
    by = c("cell", "segmentation", "condition", "panel", "donor", "sample"),
    suffix = c('', '')
)
cat(sprintf("Combined metadata for %d unique cells.\n", nrow(combined_df)))

combined_df <- combined_df %>%
    mutate(
        dataset_id = paste(
            segmentation,
            condition,
            panel,
            donor,
            sample,
            sep = "_"
        )
    ) %>%
    mutate(full_cell_id = paste(dataset_id, cell, sep = "_"))
combined_df <- combined_df %>%
    as.data.frame() %>%
    column_to_rownames(var = "full_cell_id")

# b) Find the intersection: which cells from your df are ACTUALLY in the Seurat object?
# This is our final list of cells to keep.
cells_to_keep <- intersect(df_prepared$full_cell_id, colnames(merged_seurat))

cat(sprintf(
    "Found %d cells with metadata that are present in the Seurat object.\n",
    length(cells_to_keep)
))
if (length(cells_to_keep) == 0) {
    stop(
        "No matching cells found after join. Please check cell barcode formats."
    )
}


# --- Step 3: Subset Seurat object and add metadata ---
cat(sprintf(
    "Subsetting the Seurat object from %d cells down to %d cells...\n",
    ncol(merged_seurat),
    length(cells_to_keep)
))

# Create the final, smaller Seurat object
seurat_subset <- subset(merged_seurat, cells = cells_to_keep)

# Add the metadata
seurat_subset <- AddMetaData(
    object = seurat_subset,
    metadata = combined_df
)


# --- Step 4: Verification and Save ---
cat(sprintf(" -> Final object has %d cells.\n", ncol(seurat_subset)))
# Save the final object
cat(sprintf("Saving final merged Seurat object to: %s\n", args$output_file))
output_dir <- dirname(args$output_file)
if (!dir_exists(output_dir)) {
    dir_create(output_dir, recurse = TRUE)
}
saveRDS(seurat_subset, file = args$output_file)

cat("-----------------------------------------\n")
cat("Script finished successfully!\n")
