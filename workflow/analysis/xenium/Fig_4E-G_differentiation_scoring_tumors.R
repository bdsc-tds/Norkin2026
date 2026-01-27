library(Seurat)
library(dplyr)

# ======================================================
# REQUIRED OBJECT
# ======================================================
# tumor : Xenium Seurat object
# epithelial cells must be labeled as:
#   tumor$cell_type == "epi cell"

#tumor = readRDS("proseg_expected_CRC_hImmune_v1_mm_lognorm.rds")

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

tumor$well_diff_score <- NA_real_
tumor$poor_diff_score <- NA_real_
tumor$diff_score      <- NA_real_

## ======================================================
## Precompute epithelial cell counts per donor
## ======================================================

epi_counts <- table(
  tumor$donor[tumor$cell_type == "epi cell"]
)

## ======================================================
## Score donors independently (Xenium-safe)
## ======================================================

for (d in unique(tumor$donor)) {
  
  message("Scoring donor: ", d)
  
  ## ---- skip donors with no epithelial cells ----
  if (!d %in% names(epi_counts) || epi_counts[d] == 0) {
    message("  skipping donor (no epithelial cells): ", d)
    next
  }
  
  tumor_d <- subset(
    tumor,
    subset = donor == d & cell_type == "epi cell"
  )
  
  ## ---- identify donor-specific data layer ----
  data_layer <- grep(
    "^data\\.",
    Layers(tumor_d[["Xenium"]]),
    value = TRUE
  )
  
  if (length(data_layer) != 1) {
    message("  skipping donor (ambiguous layers): ", d)
    next
  }
  
  expr <- LayerData(
    object = tumor_d,
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
  
  tumor$well_diff_score[names(well_score)] <- well_score
  tumor$poor_diff_score[names(poor_score)] <- poor_score
  tumor$diff_score[names(well_score)]      <- well_score - poor_score
}

## ======================================================
## QC summary
## ======================================================

qc_table <- tumor@meta.data %>%
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

print(qc_table)

## PLOTs printing 

library(ggplot2)
library(scales)
library(dplyr)

## ============================================================
## USER PARAMETERS
## ============================================================

out_dir <- "~/Diff_score_spatial"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

point_size <- 0.5

# color scale
low_col  <- "#2166ac"
mid_col  <- "white"
high_col <- "#b2182b"
na_col   <- "#7c7c7c"

# cap extreme values for visualization
cap_quantiles <- c(0.02, 0.98)

## ============================================================
## LOOP OVER FOVs (most robust unit)
## ============================================================

for (fov_name in Images(tumor)) {
  
  message("Processing ", fov_name)
  
  ## ---- cells + coords ----
  cells  <- Cells(tumor[[fov_name]])
  coords <- tumor@images[[fov_name]]@boundaries$centroids@coords
  
  ## ---- metadata ----
  df <- data.frame(
    cell_id    = cells,
    x          = coords[,1],
    y          = coords[,2],
    diff_score = tumor$diff_score[cells],
    donor      = tumor$donor[cells]
  )
  
  ## ---- skip FOVs without scores ----
  if (all(is.na(df$diff_score))) {
    message("  ↪ skipping (no diff_score)")
    next
  }
  
  ## ---- cap extremes (per FOV) ----
  q <- quantile(df$diff_score, cap_quantiles, na.rm = TRUE)
  df$diff_score_cap <- pmin(pmax(df$diff_score, q[1]), q[2])
  
  ## ---- plot ----
  p <- ggplot() +
    
    # background (NA)
    geom_point(
      data = df[is.na(df$diff_score_cap), ],
      aes(x, y),
      color = na_col,
      size = point_size,
      alpha = 0.4
    ) +
    
    # foreground (scored)
    geom_point(
      data = df[!is.na(df$diff_score_cap), ],
      aes(x, y, color = diff_score_cap),
      size = point_size
    ) +
    
    coord_equal() +
    scale_y_reverse() +
    
    scale_color_gradient2(
      low = low_col,
      mid = mid_col,
      high = high_col,
      midpoint = 0,
      name = "diff_score"
    ) +
    
    theme_void() +
    theme(
      plot.background = element_rect(fill = "black", color = NA),
      legend.position = "right"
    )
  
  ## ---- filename ----
  donor_name <- unique(na.omit(df$donor))
  donor_name <- donor_name[1]
  
  out_file <- file.path(
    out_dir,
    paste0(donor_name, "_", fov_name, "_diff_score.png")
  )
  
  ## ---- save ----
  ggsave(
    filename = out_file,
    plot = p,
    width = 6,
    height = 6,
    dpi = 600,
    bg = "black"
  )
}






