## ============================================================
## Libraries
## ============================================================

library(dplyr)
library(ggplot2)
library(scales)
library(viridis)

## ============================================================
## Input
## ============================================================


umap_df <- read.csv("umap_df.csv")
df_morphology_umap <- umap_df

## ============================================================
## Sanity checks
## ============================================================

stopifnot(
  all(c(
    "UMAP1", "UMAP2", "organoid_id",
    "solidity", "area", "eccentricity", "interior_holes_percentage",
    "frac_Cycling_high", "frac_Goblet_Inflammatory", "frac_Stem.like"
  ) %in% colnames(df_morphology_umap))
)

stopifnot(
  nrow(df_morphology_umap) ==
    length(unique(df_morphology_umap$organoid_id))
)

#Figure 3C

shape_colors <- c(
  "Branched"   = "#542788",
  "Elongated"  = "#6BAED6",
  "Globular"   = "#D73027",
  "Spherical"  = "#2B7522",
  "Compact"    = "#F46D43",
  "Irregular"  = "#FEE08B"
)

ggplot(
  umap_df,
  aes(x = UMAP1, y = UMAP2, color = Shape_name)
) +
  geom_point(size = 1, alpha = 0.8) +
  scale_color_manual(values = shape_colors) +
  theme_classic(base_size = 12) +
  labs(
    x = "UMAP 1",
    y = "UMAP 2",
    color = "Morphology shape"
  )

## ============================================================
## Helper: 90th percentile cap
## ============================================================

cap_90 <- function(x) {
  quantile(x, probs = 0.9, na.rm = TRUE)
}

## ============================================================
## Helper: UMAP plot with capped gradient
## ============================================================

plot_umap_capped <- function(df, value_col, label, option = "viridis", log10_transform = FALSE) {
  
  vals <- df[[value_col]]
  if (log10_transform) {
    vals <- log10(vals + 1)
  }
  
  cap_val <- cap_90(vals)
  
  df_plot <- df %>%
    mutate(value_plot = pmin(vals, cap_val))
  
  ggplot(
    df_plot,
    aes(UMAP1, UMAP2, color = value_plot)
  ) +
    geom_point(size = 0.9) +
    scale_color_viridis_c(
      option = option,
      limits = c(min(df_plot$value_plot, na.rm = TRUE), cap_val),
      oob = scales::squish,
      labels = percent_format(accuracy = 1)
    ) +
    theme_void() +
    labs(
      title = label,
      color = NULL
    ) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5)
    )
}

## ============================================================
## 1) MORPHOLOGY FEATURES (Figure 3D – top row)
## ============================================================

p_solidity <- plot_umap_capped(
  df_morphology_umap,
  value_col = "solidity",
  label = "Solidity",
  option = "plasma"
)

p_area <- plot_umap_capped(
  df_morphology_umap,
  value_col = "area",
  label = "Area (log10, capped)",
  option = "plasma",
  log10_transform = TRUE
)

p_eccentricity <- plot_umap_capped(
  df_morphology_umap,
  value_col = "eccentricity",
  label = "Eccentricity",
  option = "plasma"
)

p_holes <- plot_umap_capped(
  df_morphology_umap,
  value_col = "interior_holes_percentage",
  label = "Interior holes (%)",
  option = "plasma"
)

## ============================================================
## 2) PROGRAM FRACTIONS (Figure 3D – bottom row)
## ============================================================

p_cycling_high <- plot_umap_capped(
  df_morphology_umap,
  value_col = "frac_Cycling_high",
  label = "Cycling_high (top 10% capped)",
  option = "plasma"
)

p_goblet_infl <- plot_umap_capped(
  df_morphology_umap,
  value_col = "frac_Goblet_Inflammatory",
  label = "Goblet_Inflammatory (top 10% capped)",
  option = "plasma"
)

p_stem_like <- plot_umap_capped(
  df_morphology_umap,
  value_col = "frac_Stem.like",
  label = "Stem-like (top 10% capped)",
  option = "plasma"
)

## ============================================================
## Output objects
## ============================================================
# Morphology features:
#   p_solidity
#   p_area
#   p_eccentricity
#   p_holes
#
# Program fractions:
#   p_cycling_high
#   p_goblet_infl
#   p_stem_like
#
# Combine with patchwork / cowplot if desired
## ============================================================
