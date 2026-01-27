library(Seurat)
library(dplyr)
library(tibble)
library(ComplexHeatmap)
library(circlize)
library(ggplot2)
library(arrow)


## ============================================================
## LOAD DATA
## ============================================================

data_caf <- readRDS(
  "proseg_expected_CRC_PDO_CAF_hImmune_v1_dapi_lognorm.rds"
)

adata_malignant <- read_parquet(
  "adata_malignant.parquet"
)

## ============================================================
## REQUIRED COLUMNS
## ============================================================

stopifnot("full_id" %in% colnames(adata_malignant))
stopifnot("donor_corrected" %in% colnames(adata_malignant))

## ============================================================
## CHECK OVERLAP
## ============================================================

length(intersect(colnames(data_caf), adata_malignant$full_id))
length(colnames(data_caf))

## ============================================================
## BUILD NAMED VECTOR (full_id → donor_corrected)
## ============================================================

donor_vec <- adata_malignant$donor_corrected
names(donor_vec) <- adata_malignant$full_id

## ============================================================
## CREATE FULL-LENGTH VECTOR (NA DEFAULT)
## ============================================================

donor_full <- rep(NA_character_, ncol(data_caf))
names(donor_full) <- colnames(data_caf)

## ============================================================
## FILL ONLY MATCHED CELLS
## ============================================================

matched_cells <- intersect(colnames(data_caf), names(donor_vec))

donor_full[matched_cells] <- donor_vec[matched_cells]

## ============================================================
## ADD METADATA (SAFE)
## ============================================================

data_caf <- AddMetaData(
  object   = data_caf,
  metadata = donor_full,
  col.name = "donor_corrected"
)

## ============================================================
## FINAL CHECK
## ============================================================

table(data_caf$donor_corrected, useNA = "ifany")




## ============================
## PARAMETERS
## ============================
logfc_cutoff   <- 0.6
min_pct_cutoff <- 0.2
n_cells_plot   <- 300

out_file <- "CAF_shared_genes_heatmap_logFC0.6.png"

genes_inflammatory_strict <- c(
  "CXCL1","CXCL3","CXCL5","CXCL6","CXCL14",
  "CSF1","IL1R1","IL15RA","IDO1",
  "IFIT3","IFITM3","LY6E","SOCS3","JAK3","DERL3",
  "ICAM1","CD274","TNFRSF13C","RUNX3","ANXA1"
)

## ============================
## DE FUNCTION
## ============================
run_de <- function(obj, donor, donor_caf) {
  
  sub <- subset(
    obj,
    cells = Cells(obj)[obj$donor_corrected %in% c(donor, donor_caf)]
  )
  
  sub$donor_corrected <- factor(
    sub$donor_corrected,
    levels = c(donor, donor_caf)
  )
  
  Idents(sub) <- "donor_corrected"
  DefaultAssay(sub) <- "Xenium"
  
  FindMarkers(
    sub,
    ident.1 = donor_caf,
    ident.2 = donor,
    logfc.threshold = 0,
    min.pct = 0
  ) %>%
    rownames_to_column("gene") %>%
    mutate(direction = ifelse(avg_log2FC > 0, "Up", "Down"))
}

## ============================
## RUN DE
## ============================
de_pt22 <- run_de(data_caf, "Pt_22", "Pt_22_CAFs")
de_pt3  <- run_de(data_caf, "Pt_3",  "Pt_3_CAFs")

## ============================
## SHARED GENES
## ============================
shared_genes <- inner_join(
  de_pt22 %>%
    filter(abs(avg_log2FC) >= logfc_cutoff,
           pct.1 >= min_pct_cutoff | pct.2 >= min_pct_cutoff) %>%
    select(gene, direction_Pt22 = direction),
  de_pt3 %>%
    filter(abs(avg_log2FC) >= logfc_cutoff,
           pct.1 >= min_pct_cutoff | pct.2 >= min_pct_cutoff) %>%
    select(gene, direction_Pt3 = direction),
  by = "gene"
) %>%
  filter(direction_Pt22 == direction_Pt3)

genes_use <- intersect(shared_genes$gene, rownames(data_caf))
message("Number of shared DE genes: ", length(genes_use))

## ============================
## GENE ANNOTATION + ORDER
## ============================
gene_annot <- shared_genes %>%
  filter(gene %in% genes_use) %>%
  mutate(
    Up_down = ifelse(direction_Pt22 == "Up",
                     "Up in PDO_CAF",
                     "Down in PDO_CAF"),
    Signature = ifelse(gene %in% genes_inflammatory_strict,
                       "Inflammatory",
                       "Other")
  ) %>%
  column_to_rownames("gene")

gene_order <- c(
  rownames(gene_annot)[gene_annot$Up_down == "Up in PDO_CAF"],
  rownames(gene_annot)[gene_annot$Up_down == "Down in PDO_CAF"]
)

## ============================
## SUBSET CELLS
## ============================
cells_keep <- Cells(data_caf)[
  data_caf$donor_corrected %in%
    c("Pt_22","Pt_22_CAFs","Pt_3","Pt_3_CAFs")
]

data_sub <- subset(data_caf, cells = cells_keep)
DefaultAssay(data_sub) <- "Xenium"

## ============================
## EXPRESSION MATRIX
## ============================
expr_mat <- GetAssayData(
  data_sub,
  slot = if ("scale.data" %in% slotNames(data_sub@assays$Xenium))
    "scale.data" else "data"
)[gene_order, ]

## ============================
## SUBSAMPLE CELLS
## ============================
set.seed(1)
cells_plot <- unlist(lapply(
  split(colnames(expr_mat), data_sub$donor_corrected),
  function(x) sample(x, min(n_cells_plot, length(x)))
))

expr_mat <- expr_mat[, cells_plot]

## ============================
## Z-SCORE
## ============================
z_mat <- t(scale(t(expr_mat)))
z_mat[is.na(z_mat)] <- 0
z_mat_t <- t(z_mat)

## ============================
## ORDER ROWS
## ============================
row_order <- order(
  factor(
    data_sub$donor_corrected[rownames(z_mat_t)],
    levels = c("Pt_22_CAFs", "Pt_3_CAFs", "Pt_22", "Pt_3")
  )
)
z_mat_t <- z_mat_t[row_order, ]
## ============================
## CELL-LEVEL ANNOTATION (ROWS)
## ============================
ha_row <- ComplexHeatmap::rowAnnotation(
  donor = factor(
    data_sub$donor_corrected[rownames(z_mat_t)],
    levels = c("Pt_22_CAFs", "Pt_3_CAFs", "Pt_22", "Pt_3")
  ),
  col = list(
    donor = c(
      "Pt_22_CAFs" = "#E4211C",
      "Pt_3_CAFs"  = "#FF7F00",
      "Pt_22"      = "#4DAF4A",
      "Pt_3"       = "#377EB8"
    )
  ),
  show_annotation_name = FALSE
)

## ============================
## GENE-LEVEL ANNOTATION (COLUMNS)
## ============================
gene_annot_df <- gene_annot %>%
  filter(rownames(gene_annot) %in% colnames(z_mat_t))

ha_col <- HeatmapAnnotation(
  Up_down   = gene_annot_df$Up_down,
  Signature = gene_annot_df$Signature,
  col = list(
    Up_down = c(
      "Up in PDO_CAF"   = "#D73027",
      "Down in PDO_CAF" = "#4575B4"
    ),
    Signature = c(
      "Inflammatory" = "#984EA3",
      "Other"        = "#BDBDBD"
    )
  ),
  annotation_name_side = "left"
)

## ============================
## HEATMAP
## ============================
ht <- Heatmap(
  z_mat_t,
  name = "Z-score",
  col = colorRamp2(
    c(-2, 0, 2),
    c("#00f9f9", "black", "yellow")
  ),
  cluster_rows    = FALSE,
  cluster_columns = FALSE,
  left_annotation   = ha_row,
  bottom_annotation = ha_col,
  show_row_names    = FALSE,
  show_column_names = TRUE,
  column_names_rot  = 45,
  column_names_gp   = gpar(fontsize = 8),
  row_title = "Cells",
  column_title = paste0(
    "Shared CAF-induced genes (|log2FC| ≥ ",
    logfc_cutoff,
    ", min.pct ≥ ",
    min_pct_cutoff,
    ")\nPt_22 and Pt_3"
  ),
  use_raster = FALSE
)

## ============================
## DRAW
## ============================
draw(
  ht,
  heatmap_legend_side     = "right",
  annotation_legend_side  = "right"
)

## ============================
## OPTIONAL SAVE
## ============================
png(out_file, width = 1800, height = 1200, res = 150)
draw(ht, heatmap_legend_side = "right", annotation_legend_side = "right")
dev.off()
