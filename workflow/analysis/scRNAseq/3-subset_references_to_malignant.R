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
        output_path = cfg$scrnaseq_reference_GSE178341_malignant_seurat,
        malignant_cell_name = "Cancer cell"
    ),
    gse236581 = list(
        path = cfg$scrnaseq_reference_GSE236581_seurat,
        output_path = cfg$scrnaseq_reference_GSE236581_malignant_seurat
    ),
    marteau = list(
        path = cfg$scrnaseq_reference_marteau_seurat,
        output_path = cfg$scrnaseq_reference_marteau_malignant_seurat,
        malignant_cell_name = "Cancer cell"
    )
)

# --- Load all Seurat objects into a named list ---
cat("Loading Seurat objects...\n")
seurat_objects <- lapply(datasets_config, function(config) readRDS(config$path))
cat("All objects loaded.\n\n")


# ============================================================================
# SUBSET MALIGNANT CELLS
# ============================================================================
cat("========================================================\n")
cat("Subsetting Malignant Cells From All Datasets\n")
cat("========================================================\n")
level_name = 'Level4'

for (dataset_name in names(datasets_config)) {
    cat("\n--------------------------------------------------------\n")
    cat("Processing dataset:", dataset_name, "\n")

    config <- datasets_config[[dataset_name]]
    seurat_obj <- seurat_objects[[dataset_name]]
    print(unique(seurat_obj[[level_name]]))
    # seurat_obj_malignant <- subset(
    #     seurat_obj,
    #     seurat_obj[[level_name]] == 'Malignant'
    # )

    # # --- Save the modified object ---
    # saveRDS(seurat_obj, file = config$output_path)
    # cat("Saved to:", config$output_path, "\n")
}
