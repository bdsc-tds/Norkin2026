library(Seurat)
library(Matrix)
library(dplyr)
library(tidyr)
library(tibble)
library(pheatmap)
library(arrow)

## ============================================================
## 1️⃣ Extract and merge log-normalized expression (data layers)
## ============================================================


adata_malignant <- read_parquet("adata_malignant.parquet")
data_dev=readRDS("proseg_expected_CRC_PDO_DEV_hImmune_v1_mm_lognorm.rds")
small_organoids <- read.csv("small_organoids.csv")

## ============================================================
## 2️⃣ Sanity checks
## ============================================================

# Required columns
stopifnot("full_id" %in% colnames(adata_malignant))
stopifnot("donor_corrected" %in% colnames(adata_malignant))

# Check overlap
length(intersect(colnames(data_dev), adata_malignant$full_id))
length(colnames(data_dev))

## ============================================================
## Build named vector (full_id → donor_corrected)
## ============================================================

donor_vec <- adata_malignant$donor_corrected
names(donor_vec) <- adata_malignant$full_id

## ============================================================
## Create full-length vector with NA defaults
## ============================================================

donor_full <- rep(NA_character_, ncol(data_dev))
names(donor_full) <- colnames(data_dev)

## ============================================================
## Fill only matched cells
## ============================================================

matched_cells <- intersect(colnames(data_dev), names(donor_vec))

donor_full[matched_cells] <- donor_vec[matched_cells]

## ============================================================
## Add metadata (NOW SAFE)
## ============================================================

data_dev <- AddMetaData(
  object   = data_dev,
  metadata = donor_full,
  col.name = "donor_corrected"
)

table(data_dev$donor_corrected)

## ============================================================
## Assign organoid size class from explicit cell list
## ============================================================

data_dev$org_size_class <- "big"

data_dev$org_size_class[
  colnames(data_dev) %in% small_organoids$x
] <- "small"


## ============================================================
## Assign time classes based on explicit rules
## ============================================================

# initialize
data_dev$time <- NA_character_

## 1️⃣ small_early: ONLY 4d, 8d
data_dev$time[
  data_dev$donor_corrected %in% c(
    "Pt_8_4_days",
    "Pt_8_8_days"
  )
] <- "small_early"

## 2️⃣ small_later: weeks 3–5 AND explicitly small
data_dev$time[
  data_dev$donor_corrected %in% c(
    "Pt_8_3_weeks",
    "Pt_8_4_weeks",
    "Pt_8_5_weeks"
  ) &
    colnames(data_dev) %in% small_organoids$x
] <- "small_later"

## 3️⃣ big: 2–5 weeks AND NOT in small_organoids
data_dev$time[
  data_dev$donor_corrected %in% c(
    "Pt_8_2_weeks",
    "Pt_8_3_weeks",
    "Pt_8_4_weeks",
    "Pt_8_5_weeks"
  ) &
    !colnames(data_dev) %in% small_organoids$x
] <- "big"



#create non-spatial seurat object for gene expression extraction


xen <- data_dev

data_layers <- grep("^data\\.", Layers(xen[["Xenium"]]), value = TRUE)

expr_list <- lapply(data_layers, function(layer) {
  GetAssayData(
    object = xen,
    assay  = "Xenium",
    layer  = layer
  )
})

# merge columns (cells)
expr_merged <- do.call(cbind, expr_list)

# ensure unique cell names
expr_merged <- expr_merged[, !duplicated(colnames(expr_merged))]

## ============================================================
## 2️⃣ Create NON-SPATIAL Seurat object
## ============================================================

seurat_ns <- CreateSeuratObject(
  counts = expr_merged,
  assay  = "RNA",
  meta.data = xen@meta.data[colnames(expr_merged), , drop = FALSE]
)

## ============================================================
## 3️⃣ (Optional but recommended) normalize & scale
## ============================================================

seurat_ns <- NormalizeData(seurat_ns, verbose = FALSE)
seurat_ns <- ScaleData(seurat_ns, verbose = FALSE)

## ============================================================
## 4️⃣ Sanity checks
## ============================================================

# metadata preserved
stopifnot(all(colnames(seurat_ns) %in% rownames(seurat_ns@meta.data)))

# key annotations still there
table(seurat_ns$time)
table(seurat_ns$org_size_class)

seurat_ns


library(dplyr)
library(tibble)

genes <- rownames(seurat_ns)

mean_df <- FetchData(
  seurat_ns,
  vars = c(genes, "time")
) %>%
  pivot_longer(
    cols = all_of(genes),
    names_to = "gene",
    values_to = "expr"
  ) %>%
  group_by(gene, time) %>%
  summarise(mean_expr = mean(expr), .groups = "drop") %>%
  pivot_wider(
    names_from = time,
    values_from = mean_expr
  )

later_dominant_genes <- mean_df %>%
  filter(
    small_later > small_early,
    small_later > big
  ) %>%
  mutate(
    later_vs_early = small_later - small_early,
    later_vs_big   = small_later - big
  ) %>%
  arrange(desc(later_vs_early + later_vs_big))

head(later_dominant_genes, 20)



## ============================================================
## 1️⃣ Top 20 dormant candidate genes
## ============================================================


prolif_genes <- c(
  "MKI67","TOP2A","CDK1","CCNB1","CCNB2","TYMS",
  "STMN1","CENPF","UBE2C","ORC1","ORC6","MCM2",
  "MCM3","MCM4","MCM5","PCNA","HMGB2"
)

dormant_candidates <- later_dominant_genes %>%
  filter(!gene %in% prolif_genes)


genes_use <- dormant_candidates %>%
  slice_head(n = 20) %>%
  pull(gene)

## ============================================================
## 2️⃣ Fetch expression + metadata
## ============================================================

df <- FetchData(
  seurat_ns,
  vars = c(genes_use, "donor_corrected", "time", "org_size_class")
)

## ============================================================
## 3️⃣ Map donor_corrected → heatmap groups
## ============================================================

df <- df %>%
  mutate(
    heat_group = case_when(
      donor_corrected == "Pt_8_4_days"  ~ "4d",
      donor_corrected == "Pt_8_8_days"  ~ "8d",
      donor_corrected == "Pt_8_2_weeks" ~ "2w",
      
      donor_corrected == "Pt_8_3_weeks" & org_size_class == "big"   ~ "3w_big",
      donor_corrected == "Pt_8_4_weeks" & org_size_class == "big"   ~ "4w_big",
      donor_corrected == "Pt_8_5_weeks" & org_size_class == "big"   ~ "5w_big",
      
      donor_corrected == "Pt_8_3_weeks" & org_size_class == "small" ~ "3w_small",
      donor_corrected == "Pt_8_4_weeks" & org_size_class == "small" ~ "4w_small",
      donor_corrected == "Pt_8_5_weeks" & org_size_class == "small" ~ "5w_small",
      
      TRUE ~ NA_character_
    )
  ) %>%
  filter(!is.na(heat_group))

## ============================================================
## 4️⃣ Mean expression per gene × group
## ============================================================

mean_df <- df %>%
  pivot_longer(
    cols = all_of(genes_use),
    names_to = "gene",
    values_to = "expr"
  ) %>%
  group_by(gene, heat_group) %>%
  summarise(mean_expr = mean(expr), .groups = "drop") %>%
  pivot_wider(
    names_from = heat_group,
    values_from = mean_expr
  )

## ============================================================
## 5️⃣ Build matrix in correct biological order
## ============================================================

col_order <- c(
  "4d", "8d", "2w",
  "3w_big", "4w_big", "5w_big",
  "3w_small", "4w_small", "5w_small"
)

mat <- mean_df %>%
  column_to_rownames("gene") %>%
  as.matrix()

mat <- mat[, col_order]

## ============================================================
## 6️⃣ Row-wise scaling
## ============================================================

mat_scaled <- t(scale(t(mat)))

pheatmap(
  mat_scaled,
  cluster_rows = TRUE,
  cluster_cols = FALSE,
  color = colorRampPalette(c("cyan", "black", "yellow"))(100),
  fontsize_row = 12,
  fontsize_col = 15,
  border_color = NA,
  main = "Dormant candidate genes\nTemporal and size-resolved"
)


