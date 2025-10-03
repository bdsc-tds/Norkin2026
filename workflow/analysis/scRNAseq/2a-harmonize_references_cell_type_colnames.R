library(Seurat)
library(dplyr)
library(readr)

# Set working directory to the script's location
# setwd(this.path::here()) # Uncomment if you need to set the working directory
source('../../scripts/readwrite.R') # Uncomment if you use these helper functions
cfg <- config() # Uncomment if you use a config file

# --- CONFIGURE DATASETS  ---
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

    if (
        dataset_name == "marteau" && "originalexp" %in% names(seurat_obj@assays)
    ) {
        # The Marteau dataset has an 'originalexp' assay that we need to rename
        seurat_obj <- RenameAssays(seurat_obj, 'originalexp', 'RNA')
    }

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
                seurat_obj[[level_name]] <- gsub(
                    "/",
                    "_",
                    as.character(seurat_obj@meta.data[[source_col]])
                )
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
    # saveRDS(seurat_obj, file = config$path)
    # cat("Saved modified inplace to:", config$path, "\n")

    # Replace the object in our list with the modified one
    seurat_objects[[dataset_name]] <- seurat_obj
}


# ============================================================================
# HARMONIZE CELL TYPE COLUMNS
# ============================================================================
cat("========================================================\n")
cat("Adding extra simplified Cell Type Column Across Datasets\n")
cat("========================================================\n")


# 1. Load the cell type mapping CSV file
mapping_file_path <- file.path(
    cfg$metadata_dir,
    "simplified_cell_type_mapping.csv"
)
cat(paste("Loading cell type mapping from:", mapping_file_path, "\n"))

if (!file.exists(mapping_file_path)) {
    stop("Mapping file not found! Please check the path in `cfg$metadata_dir`.")
}
cell_type_map_df <- readr::read_csv(mapping_file_path, show_col_types = FALSE)
lookup_vector <- setNames(
    cell_type_map_df$cell_type_simple,
    cell_type_map_df$cell_type
)
cat("Cell type lookup map created.\n\n")

# 3. Iterate through each dataset, update metadata, and save
for (dataset_name in names(seurat_objects)) {
    cat(paste0("--- Processing dataset: ", dataset_name, " ---\n"))

    seurat_obj <- seurat_objects[[dataset_name]]
    config <- datasets_config[[dataset_name]]

    # 4. Determine the source and new column names based on the dataset
    source_col_name <- NULL
    if (dataset_name == "gse178341") {
        source_col_name <- "Level3"
    } else if (dataset_name == "gse236581") {
        source_col_name <- "Level2"
    }

    # If the dataset is not one of the ones we want to map, skip it.
    if (is.null(source_col_name)) {
        cat("Skipping this dataset as no mapping rule is defined for it.\n\n")
        next
    }

    # Get the actual metadata column name from the config
    new_col_name <- paste0(source_col_name, "_simple")

    cat(paste("Mapping column:", source_col_name, "->", new_col_name, "\n"))

    # Check if the source column exists
    if (!source_col_name %in% colnames(seurat_obj@meta.data)) {
        cat(paste0(
            "ERROR: Source column '",
            source_col_name,
            "' not found in the metadata. Skipping this dataset.\n\n"
        ))
        next
    }

    # 5. Get the original cell types and map them to the new simplified types
    original_cell_types <- seurat_obj@meta.data[[source_col_name]]
    simplified_cell_types <- lookup_vector[as.character(original_cell_types)]
    names(simplified_cell_types) <- colnames(seurat_obj)

    # Report if any cell types were not found in the mapping file
    unmapped_count <- sum(is.na(simplified_cell_types))
    if (unmapped_count > 0) {
        cat(paste0(
            "Warning: ",
            unmapped_count,
            " cells had types not found in the mapping file. They will be set to NA.\n"
        ))
    }

    # 6. Add the new column to the Seurat object's metadata
    seurat_obj[[new_col_name]] <- simplified_cell_types
    cat(paste("Added new metadata column '", new_col_name, "'.\n"))

    # 7. Save the updated Seurat object, overwriting the original file
    # Replace the object in our list with the modified one
    seurat_objects[[dataset_name]] <- seurat_obj

    cat(paste("Saving updated object to:", config$path, "\n"))
    saveRDS(seurat_obj, file = config$path)

    cat(paste0("Successfully updated and saved ", dataset_name, ".\n\n"))
}

cat("\n--------------------------------------------------------\n")
cat("✅ All objects harmonized and saved successfully!\n")
cat("--------------------------------------------------------\n")
