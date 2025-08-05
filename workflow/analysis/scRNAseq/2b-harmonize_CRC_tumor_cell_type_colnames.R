# --- Load Required Libraries ---
# install.packages(c("Seurat", "readr")) # Run this if you don't have them installed
library(Seurat)
library(readr)

# Set working directory to the script's location
# setwd(this.path::here()) # Uncomment if you need to set the working directory
source('../../scripts/readwrite.R') # Uncomment if you use these helper functions
cfg <- config() # Uncomment if you use a config file


# Your configuration list for the datasets
datasets_config <- list(
    "10x_0um_CRC" = list(
        path = file.path(
            cfg$xenium_processed_R_dir,
            "CRC/10x_0um_CRC_TUMOR_only.rds"
        )
    ),
    "10x_5um_CRC" = list(
        path = file.path(
            cfg$xenium_processed_R_dir,
            "CRC/10x_5um_CRC_TUMOR_only.rds"
        )
    ),
    "10x_mm_5um_CRC" = list(
        path = file.path(
            cfg$xenium_processed_R_dir,
            "CRC/10x_mm_5um_CRC_TUMOR_only.rds"
        )
    ),
    "proseg_expected_CRC" = list(
        path = file.path(
            cfg$xenium_processed_R_dir,
            "CRC/proseg_expected_CRC_TUMOR_only.rds"
        )
    )
)

# --- Load all Seurat objects into a named list ---
cat("Loading Seurat objects...\n")
seurat_objects <- lapply(datasets_config, function(config) readRDS(config$path))
cat("All objects loaded.\n\n")


# ============================================================================
# APPLY STANDARDIZED LEVEL COLUMNS
# ============================================================================
cat("========================================================\n")
cat("Standardizing Level Columns Across All Datasets\n")
cat("========================================================\n")

# Define the full set of standardized column names we want in the end
target_levels <- c("Level1")

# Loop through each dataset to process it
for (dataset_name in names(datasets_config)) {
    cat("\n--------------------------------------------------------\n")
    cat("Processing dataset:", dataset_name, "\n")

    config <- datasets_config[[dataset_name]]
    seurat_obj <- seurat_objects[[dataset_name]]

    # # Loop through our target level names (Level1, Level2, etc.)
    # for (level_name in target_levels) {
    #     # Check if a mapping for this level exists in the config for this dataset
    #     if (level_name %in% names(config$level_mapping)) {
    #         # It exists, so get the original source column name
    #         source_col <- config$level_mapping[[level_name]]

    #         # A crucial safety check: does this source column actually exist in the data?
    #         if (source_col %in% colnames(seurat_obj@meta.data)) {
    #             # Yes. Create the new standard column and copy the data into it.
    #             cat(sprintf(
    #                 "  - Creating '%s' from source column '%s'\n",
    #                 level_name,
    #                 source_col
    #             ))
    #             seurat_obj[[level_name]] <- gsub(
    #                 "/",
    #                 "_",
    #                 as.character(seurat_obj@meta.data[[source_col]])
    #             )
    #         } else {
    #             # The column specified in the config was not found in the object.
    #             cat(sprintf(
    #                 "  - WARNING: Source column '%s' not found. Creating '%s' with NA.\n",
    #                 source_col,
    #                 level_name
    #             ))
    #             seurat_obj[[level_name]] <- NA
    #         }
    #     } else {
    #         # No mapping exists for this level. Create the standard column and fill with NA.
    #         cat(sprintf(
    #             "  - No mapping for '%s'. Creating column with NA.\n",
    #             level_name
    #         ))
    #         seurat_obj[[level_name]] <- NA
    #     }
    # }

    # # --- Save the modified object ---
    # saveRDS(seurat_obj, file = config$path)
    # cat("Saved modified inplace to:", config$path, "\n")

    # 2. Write the metadata data frame to the new CSV file
    csv_output_path <- sub("\\.rds$", "_metadata.csv", config$path)
    write.csv(seurat_obj@meta.data, file = csv_output_path, row.names = TRUE)
    cat("Saved metadata index to:", csv_output_path, "\n")

    # Replace the object in our list with the modified one
    seurat_objects[[dataset_name]] <- seurat_obj
}
