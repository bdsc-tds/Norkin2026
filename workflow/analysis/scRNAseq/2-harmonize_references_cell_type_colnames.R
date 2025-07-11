# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SCRIPT: Generate a Cell Type Dictionary from Multiple Seurat References
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This script performs the following steps:
# 1. Uses a detailed configuration list to define Seurat object paths and
#    the mapping of specific metadata columns to conceptual levels (Level1-4).
# 2. Loads all specified Seurat objects.
# 3. Generates a comprehensive dictionary file ('cell_type_dictionary.csv').
#    This file lists every unique cell type label from the specified columns,
#    organized by Level and including the source study name.
#
# This dictionary serves as the starting point for creating a final,
# harmonized mapping file.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# --- SETUP: Load libraries ---
library(Seurat)
library(dplyr)
library(readr)

# Set working directory to the script's location
# setwd(this.path::here()) # Uncomment if you need to set the working directory
source('../../scripts/readwrite.R') # Uncomment if you use these helper functions
cfg <- config() # Uncomment if you use a config file

# --- ACTION REQUIRED: CONFIGURE YOUR DATASETS HERE ---
# This list is the main control panel for the script.
# For each dataset, define:
#   - path: Path to the input Seurat .rds file.
#   - level_mapping: A NAMED LIST that maps a conceptual level (e.g., 'Level1')
#     to the actual column name in the Seurat object's metadata.
#
datasets_config <- list(
    gse178341 = list(
        path = cfg$scrnaseq_reference_GSE178341_seurat,
        level_mapping = list(
            Level1 = "clTopLevel",
            Level2 = "clMidwayPr",
            Level3 = "cl295v11SubFull"
        )
    ),
    gse236581 = list(
        path = cfg$scrnaseq_reference_GSE236581_seurat,
        level_mapping = list(
            Level1 = "MajorCellType",
            Level2 = "SubCellType"
        )
    ),
    marteau = list(
        path = cfg$scrnaseq_reference_marteau_seurat,
        level_mapping = list(
            Level1 = "cell_type_coarse",
            Level2 = "cell_type_middle",
            Level3 = "cell_type",
            Level4 = "cell_type_fine" # Assuming 'cell_type' is the most granular
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
target_levels <- c("Level1", "Level2", "Level3", "Level4")

# Loop through each dataset to process it
for (dataset_name in names(datasets_config)) {
    cat("\n--------------------------------------------------------\n")
    cat("Processing dataset:", dataset_name, "\n")

    config <- datasets_config[[dataset_name]]
    seurat_obj <- seurat_objects[[dataset_name]]

    # Loop through our target level names (Level1, Level2, etc.)
    for (level_name in target_levels) {
        # Check if a mapping for this level exists in the config for this dataset
        if (level_name %in% names(config$level_mapping)) {
            # It exists, so get the original source column name
            source_col <- config$level_mapping[[level_name]]

            # A crucial safety check: does this source column actually exist in the data?
            if (source_col %in% colnames(seurat_obj@meta.data)) {
                # Yes. Create the new standard column and copy the data into it.
                cat(sprintf(
                    "  - Creating '%s' from source column '%s'\n",
                    level_name,
                    source_col
                ))
                seurat_obj[[level_name]] <- seurat_obj@meta.data[[source_col]]
            } else {
                # The column specified in the config was not found in the object.
                cat(sprintf(
                    "  - WARNING: Source column '%s' not found. Creating '%s' with NA.\n",
                    source_col,
                    level_name
                ))
                seurat_obj[[level_name]] <- NA
            }
        } else {
            # No mapping exists for this level. Create the standard column and fill with NA.
            cat(sprintf(
                "  - No mapping for '%s'. Creating column with NA.\n",
                level_name
            ))
            seurat_obj[[level_name]] <- NA
        }
    }

    # --- Save the modified object ---
    saveRDS(seurat_obj, file = config$path)
    cat("Saved modified inplace to:", config$path, "\n")

    # Replace the object in our list with the modified one
    seurat_objects[[dataset_name]] <- seurat_obj
}

cat("\n--------------------------------------------------------\n")
cat("✅ All objects harmonized and saved successfully!\n")
cat("--------------------------------------------------------\n")
