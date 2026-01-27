library(Seurat)
library(dplyr)

# ======================================================
# REQUIRED OBJECT
# ======================================================
# organoids : Xenium Seurat object
# scoring is performed on ALL cells
# ======================================================

## ======================================================
## Differentiation signatures
## ======================================================

sig_well <- c(
  "TFF3","REG4","CEACAM6","CEACAM8","MUC5AC",
  "CDX1","CDX2","ANPEP","ACE","AQP1",
  "PPARGC1A","TMPRSS2","IQGAP2","GLIPR2","CCL28"
)

sig_poor <- c(
  "MKI67","CDK1","UBE2C","CENPF","ORC6",
  "STMN1","MYC","SOX9","RNF43","CD44"
)

## ======================================================
## Initialize metadata columns
## ======================================================

organoids$well_diff_score <- NA_real_
organoids$poor_diff_score <- NA_real_
organoids$diff_score      <- NA_real_

## ======================================================
## Precompute cell counts per donor
## ======================================================

cell_counts <- table(organoids$donor)

## ======================================================
## Score donors independently (Xenium-safe)
## ======================================================

for (d in unique(organoids$donor)) {
  
  message("Scoring donor: ", d)
  
  ## ---- skip donors with no cells (safety) ----
  if (!d %in% names(cell_counts) || cell_counts[d] == 0) {
    message("  skipping donor (no cells): ", d)
    next
  }
  
  organoids_d <- subset(
    organoids,
    subset = donor == d
  )
  
  ## ---- identify donor-specific data layer ----
  data_layer <- grep(
    "^data\\.",
    Layers(organoids_d[["Xenium"]]),
    value = TRUE
  )
  
  if (length(data_layer) != 1) {
    message("  skipping donor (ambiguous layers): ", d)
    next
  }
  
  expr <- LayerData(
    object = organoids_d,
    assay  = "Xenium",
    layer  = data_layer
  )
  
  sig_well_use <- intersect(sig_well, rownames(expr))
  sig_poor_use <- intersect(sig_poor, rownames(expr))
  
  ## ---- require minimal gene coverage ----
  if (length(sig_well_use) < 3 || length(sig_poor_use) < 3) {
    message("  skipping donor (too few signature genes): ", d)
    next
  }
  
  well_score <- colMeans(expr[sig_well_use, , drop = FALSE])
  poor_score <- colMeans(expr[sig_poor_use, , drop = FALSE])
  
  organoids$well_diff_score[names(well_score)] <- well_score
  organoids$poor_diff_score[names(poor_score)] <- poor_score
  organoids$diff_score[names(well_score)]      <- well_score - poor_score
}

## ======================================================
## QC summary
## ======================================================

qc_table_organoids <- organoids@meta.data %>%
  transmute(
    donor  = donor,
    scored = !is.na(diff_score)
  ) %>%
  group_by(donor) %>%
  summarise(
    n_cells     = n(),
    n_scored    = sum(scored),
    frac_scored = n_scored / n_cells,
    .groups = "drop"
  ) %>%
  arrange(desc(frac_scored))

print(qc_table_organoids)
