library(Seurat)
library(dplyr)
library(ComplexHeatmap)
library(circlize)
library(grid)


data_caf=readRDS("proseg_expected_CRC_PDO_CAF_hImmune_v1_dapi_lognorm.rds")
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


# ============================================================
# 1. Subset Seurat object
# ============================================================

caf_Pt_28 <- subset(
  data_caf,
  subset = donor_corrected %in% c("Pt_28", "Pt_28_drug")
)

table(caf_Pt_28$donor_corrected)

# ============================================================
# 1. DE genes
# ============================================================

caf_Pt_28$condition <- caf_Pt_28$donor_corrected
Idents(caf_Pt_28) <- "condition"
DefaultAssay(caf_Pt_28) <- "Xenium"

de_Pt28 <- FindMarkers(
  caf_Pt_28,
  ident.1 = "Pt_28_drug",
  ident.2 = "Pt_28",
  logfc.threshold = 0.6,
  min.pct = 0.2,
  test.use = "wilcox"
)

# add gene names as column
de_Pt28 <- de_Pt28 %>%
  tibble::rownames_to_column("gene") %>%
  arrange(desc(avg_log2FC))



# ============================================================
# 2. Define AKT inhibition gene sets (EDIT HERE)
# ============================================================

# ============================================================
# 2. Define AKT inhibition gene sets FROM Pt_28 DE
# ============================================================

genes_up <- de_Pt28 %>%
  filter(
    p_val_adj < 0.05,
    avg_log2FC >= 0.6
  ) %>%
  pull(gene)

genes_down <- de_Pt28 %>%
  filter(
    p_val_adj < 0.05,
    avg_log2FC <= -0.6
  ) %>%
  pull(gene)

genes_all <- intersect(
  c(genes_up, genes_down),
  rownames(caf_Pt_28)
)

length(genes_up)
length(genes_down)


# ============================================================
# 3. Sample cells per condition (BASE R, SAFE)
# ============================================================

set.seed(1)

cells_ctrl <- colnames(caf_Pt_28)[caf_Pt_28$donor_corrected == "Pt_28"]
cells_drug <- colnames(caf_Pt_28)[caf_Pt_28$donor_corrected == "Pt_28_drug"]

cells_ctrl <- sample(cells_ctrl, min(3000, length(cells_ctrl)))
cells_drug <- sample(cells_drug, min(3000, length(cells_drug)))

cells_use <- c(cells_ctrl, cells_drug)

table(caf_Pt_28$donor_corrected[cells_use])

# ============================================================
# 4. Extract per-cell expression (Xenium layer)
# ============================================================

expr <- LayerData(
  caf_Pt_28,
  assay = "Xenium",
  layer = "data"
)[genes_all, cells_use]

expr <- as.matrix(expr)

# ============================================================
# 5. Optional gene filtering (robust, reviewer-safe)
# ============================================================

expr_detected <- expr > 0

gene_prev_ctrl <- rowMeans(expr_detected[, cells_ctrl, drop = FALSE])
gene_prev_drug <- rowMeans(expr_detected[, cells_drug, drop = FALSE])

keep_prev <- (gene_prev_ctrl >= 0.10) | (gene_prev_drug >= 0.10)

expr_z_tmp <- t(scale(t(expr)))
expr_z_tmp[is.na(expr_z_tmp)] <- 0

gene_sd <- apply(expr_z_tmp, 1, sd)
keep_sd <- gene_sd >= 0.5

genes_keep <- names(which(keep_prev & keep_sd))
expr <- expr[genes_keep, , drop = FALSE]

# ============================================================
# 6. Z-score per gene
# ============================================================

mat_z <- t(scale(t(expr)))
mat_z[is.na(mat_z)] <- 0

# ============================================================
# 7. Column annotation (cells)
# ============================================================

cell_condition <- caf_Pt_28$donor_corrected[cells_use]

col_ha <- HeatmapAnnotation(
  Condition = cell_condition,
  col = list(
    Condition = c(
      "Pt_28" = "#4daf4a",
      "Pt_28_drug" = "#e41a1c"
    )
  )
)

# ============================================================
# 8. Row annotations
#    - Direction (Up / Down)
#    - Signaling module (NO "other" label)
# ============================================================

gene_direction <- factor(
  ifelse(rownames(mat_z) %in% genes_up, "Up", "Down"),
  levels = c("Up", "Down")
)

gene_module <- case_when(
  rownames(mat_z) %in% c("ERBB2","ERBB3","PIK3CA","EGFR","MET") ~ "RTK feedback",
  
  rownames(mat_z) %in% c("IL18","IL23A","VSIR","RORC","STAT6") ~ 
    "Inflammatory / immune",
  
  rownames(mat_z) %in% c("ICAM1","L1CAM","SEMA3F","CEACAM6","CEACAM8") ~ 
    "Adhesion / guidance",
  
  rownames(mat_z) %in% c("ID4","ID2","MYC","JAG1") ~ 
    "Stem / survival",
  
  rownames(mat_z) %in% c("PDK1","FGFR1","FLT1","ANGPT1","APLN","SYK","SMAD3") ~ 
    "AKT / PI3K core",
  
  rownames(mat_z) %in% c("POSTN","DCN","LOXL2","COL18A1","ANPEP") ~ 
    "ECM / stromal",
  
  rownames(mat_z) %in% c("JUN","EGR3","G0S2") ~ 
    "Immediate early / stress",
  
  TRUE ~ NA_character_
)


row_ha <- rowAnnotation(
  Direction = gene_direction,
  Module = gene_module,
  col = list(
    Direction = c(Up = "#d73027", Down = "#4575b4"),
    Module = c(
      "RTK feedback" = "#984ea3",
      "Inflammatory / immune" = "#e41a1c",
      "Adhesion / guidance" = "#ff7f00",
      "Stem / survival" = "#4daf4a",
      "AKT / PI3K core" = "#377eb8",
      "ECM / stromal" = "#a65628",
      "Immediate early / stress" = "#999999"
    )
  ),
  na_col = "transparent",
  annotation_width = unit(c(2, 4), "mm")
)

# ============================================================
# 9. Order cells (ctrl → drug)
# ============================================================

ord_cells <- order(cell_condition)
mat_z <- mat_z[, ord_cells]

# ============================================================
# 10. Draw heatmap (final style)
# ============================================================

heat_cols <- colorRamp2(
  c(-2, 0, 2),
  c("#00C8C8", "#1A1A1A", "#F2E94E")
)

ht <- Heatmap(
  mat_z,
  name = "Z-score",
  col = heat_cols,
  cluster_rows = F,
  cluster_columns = FALSE,
  show_column_names = FALSE,
  row_names_gp = gpar(fontsize = 6),
  top_annotation = col_ha,
  left_annotation = row_ha,
  column_title = "Pt_28 AKT inhibition",
  use_raster = FALSE
)

draw(
  ht,
  heatmap_legend_side = "right",
  annotation_legend_side = "right"
)
