## ============================================================
## Stromal-centered niche archetypes (k = 10)
## CRC Xenium
## ============================================================

## ----------------------------
## Libraries
## ----------------------------
library(arrow)
library(dplyr)
library(ggplot2)
library(scatterpie)
library(ComplexHeatmap)
library(circlize)
library(RColorBrewer)
library(grid)

## ----------------------------
## 1) Load data
## ----------------------------
df <- read_parquet("df_composition_CRC_Level1_with_programs.parquet")

## ----------------------------
## 2) Define tumor cell states
## ----------------------------
tumor_states <- c(
  "Chemokine_EMT",
  "Chemokine_Inflammatory",
  "Cycling_high",
  "Goblet_Inflammatory",
  "Goblet_MUC",
  "Senescent_Inflammatory",
  "Stem-like",
  "Transit-amplifying"
)

## ----------------------------
## 3) Apply niche filters
## ----------------------------
df_filt <- df %>%
  filter(
    ## stromal-centered
    `stromal cell` >= 1,
    
    ## ≥ 2 tumor-program cells
    rowSums(across(all_of(tumor_states))) >= 1
  )

message("Filtered niches: ", nrow(df_filt))

## ----------------------------
## 4) Build fractional matrix
## ----------------------------
mat <- df_filt %>%
  select(-full_id) %>%
  as.matrix()

mat_frac <- mat / rowSums(mat)

## make names safe for plotting
colnames(mat_frac) <- make.names(colnames(mat_frac))

## ----------------------------
## 5) K-means clustering
## ----------------------------
set.seed(1)

k <- 10
km <- kmeans(
  mat_frac,
  centers  = k,
  iter.max = 100,
  nstart   = 10
)

cluster_id <- km$cluster

## ----------------------------
## 6) Cluster summaries
## ----------------------------
cluster_counts <- as.data.frame(table(cluster_id))
colnames(cluster_counts) <- c("cluster", "n_niches")
cluster_counts$cluster <- as.integer(as.character(cluster_counts$cluster))
cluster_counts <- cluster_counts[order(-cluster_counts$n_niches), ]

cluster_comp <- as.data.frame(mat_frac) %>%
  mutate(cluster = cluster_id) %>%
  group_by(cluster) %>%
  summarise(across(everything(), mean), .groups = "drop")

df_pies <- cluster_comp %>%
  left_join(cluster_counts, by = "cluster") %>%
  arrange(desc(n_niches))

## ----------------------------
## 7) Pie layout (compact grid)
## ----------------------------
ncol_grid <- 4
nrow_grid <- 3

df_pies <- df_pies %>%
  mutate(
    x = ((row_number() - 1) %% ncol_grid) + 1,
    y = nrow_grid - ((row_number() - 1) %/% ncol_grid),
    r = sqrt(n_niches) / max(sqrt(n_niches)) * 0.48
  )

## ----------------------------
## 8) Color palette (original)
## ----------------------------
cell_colors <- c(
  ## tumor states
  "Cycling_high"             = "#17BECF",
  "Transit.amplifying"       = "#0433FF",
  "Goblet_MUC"               = "#C77CFF",
  "Goblet_Inflammatory"      = "#FE3FFA",
  "Senescent_Inflammatory"   = "#FFEEAA",
  "Chemokine_Inflammatory"   = "#2CA02C",
  "Chemokine_EMT"            = "#FEA501",
  "Stem.like"                = "#F6252F",
  
  ## cell types
  "B.cell"                   = "#F55D5A",
  "mast.cell"                = "#45AA04",
  "myeloid.cell"             = "#1BB583",
  "plasma.cell"              = "#17A6E8",
  "stromal.cell"             = "#9371FC",
  "lymphoid.cell"            = "#F740CE"
)

## ----------------------------
## 9) Pie chart plot
## ----------------------------
p_pies <- ggplot() +
  geom_scatterpie(
    data = df_pies,
    aes(x = x, y = y, r = r),
    cols = setdiff(
      colnames(df_pies),
      c("cluster", "n_niches", "x", "y", "r")
    ),
    color = "black"
  ) +
  scale_fill_manual(values = cell_colors, drop = FALSE) +
  coord_equal(expand = FALSE) +
  theme_void() +
  labs(
    title = "Stromal-centered niche archetypes (k = 10)",
    subtitle = "Pie = mean composition, size = number of niches"
  )

print(p_pies)

## ----------------------------
## 10) Heatmap preparation
## ----------------------------
row_order <- order(cluster_id)
mat_plot  <- mat_frac[row_order, ]
cluster_ord <- factor(cluster_id[row_order], levels = 1:k)

cluster_colors <- setNames(
  brewer.pal(k, "Set3"),
  levels(cluster_ord)
)

ha_row <- rowAnnotation(
  NicheCluster = cluster_ord,
  col = list(NicheCluster = cluster_colors),
  show_annotation_name = FALSE,
  annotation_width = unit(4, "mm")
)

## clean x-axis labels (no dots)
pretty_col_labels <- gsub("\\.", " ", colnames(mat_plot))

## ----------------------------
## 11) Heatmap object
## ----------------------------
ht <- Heatmap(
  mat_plot,
  name = "Fraction",
  col = colorRamp2(
    c(0, 0.25, 0.5, 1),
    c("white", "#E6DFFF", "#B9A7FF", "#9371FC")
  ),
  cluster_rows = FALSE,
  cluster_columns = TRUE,
  show_row_names = FALSE,
  column_labels = pretty_col_labels,
  column_names_rot = 45,
  column_names_gp = gpar(fontsize = 8),
  left_annotation = ha_row,
  row_title = "Stromal-centered niches",
  column_title = "Cell states / cell types",
  use_raster = FALSE
)

draw(ht)

## ----------------------------
## 12) Save heatmap as high-res PNG
## ----------------------------
png(
  filename = "stromal_centered_niches_k10_heatmap.png",
  width  = 2600,
  height = 3800,
  res    = 300
)

draw(ht)
dev.off()
