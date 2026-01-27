## ============================================================
## Libraries
## ============================================================

library(dplyr)
library(ggplot2)
library(scales)
library(tidyr)

## ============================================================
## Load data (organoid-level morphology UMAP table)
## ============================================================

df_morphology_umap <- read.csv("umap_df.csv")

## ============================================================
## Assumptions
## ============================================================
# - 1 row = 1 organoid
# - Shape_name already uses:
#   c("Branched","Irregular","Spherical",
#     "Globular","Elongated","Compact")
# - Program fractions stored as frac_* columns
# - Fractions sum ~1 per organoid

## ============================================================
## Settings
## ============================================================

program_cols <- c(
  "frac_Chemokine_EMT",
  "frac_Cycling_high",
  "frac_Goblet_Inflammatory",
  "frac_Goblet_MUC",
  "frac_Stem.like",
  "frac_Transit.amplifying"
)

program_col_fraction <- "frac_Cycling_high"

shape_colors <- c(
  "Branched"   = "#542788",
  "Elongated"  = "#6BAED6",
  "Globular"   = "#D73027",
  "Spherical"  = "#2B7522",
  "Compact"    = "#F46D43",
  "Irregular"  = "#FEE08B"
)

## ============================================================
## Shannon entropy helper
## ============================================================

shannon_entropy <- function(p) {
  p <- p[p > 0]
  -sum(p * log(p))
}

## ============================================================
## Prepare plotting data + compute entropy
## ============================================================

df_plot <- df_morphology_umap %>%
  rowwise() %>%
  mutate(
    shannon_entropy = shannon_entropy(
      c_across(all_of(program_cols))
    )
  ) %>%
  ungroup()

## ============================================================
## Order morphology shapes by mean entropy (HIGH → LOW)
## ============================================================

shape_order <- df_plot %>%
  group_by(Shape_name) %>%
  summarise(
    mean_entropy = mean(shannon_entropy, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_entropy)) %>%   # 🔥 HIGH → LOW
  pull(Shape_name)

df_plot <- df_plot %>%
  mutate(
    Shape_name = factor(Shape_name, levels = shape_order)
  )

## ============================================================
## Helper: box-style summary (IQR + mean)
## ============================================================

build_box_stats <- function(df, value_col) {
  df %>%
    group_by(Shape_name) %>%
    summarise(
      q25  = quantile(.data[[value_col]], 0.25, na.rm = TRUE),
      q75  = quantile(.data[[value_col]], 0.75, na.rm = TRUE),
      mean = mean(.data[[value_col]], na.rm = TRUE),
      ymin = min(.data[[value_col]], na.rm = TRUE),
      ymax = max(.data[[value_col]], na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      x    = as.numeric(Shape_name),
      xmin = x - 0.325,
      xmax = x + 0.325
    )
}

## ============================================================
## 1) Cycling_high fraction by morphology shape
## ============================================================

df_box_cycling <- build_box_stats(df_plot, program_col_fraction)

p_cycling_box <- ggplot() +
  
  # IQR box
  geom_rect(
    data = df_box_cycling,
    aes(xmin = xmin, xmax = xmax, ymin = q25, ymax = q75, fill = Shape_name),
    color = "black",
    linewidth = 0.6,
    alpha = 0.85
  ) +
  
  # min–max line
  geom_segment(
    data = df_box_cycling,
    aes(x = x, xend = x, y = ymin, yend = ymax),
    linewidth = 0.6
  ) +
  
  # mean line
  geom_segment(
    data = df_box_cycling,
    aes(x = xmin, xend = xmax, y = mean, yend = mean),
    linewidth = 0.9
  ) +
  
  # individual organoids
  geom_jitter(
    data = df_plot,
    aes(x = as.numeric(Shape_name), y = .data[[program_col_fraction]]),
    width = 0.15,
    size = 1.2,
    color = "black",
    alpha = 0.6
  ) +
  
  scale_x_continuous(
    breaks = seq_along(shape_order),
    labels = shape_order
  ) +
  scale_fill_manual(values = shape_colors) +
  scale_y_continuous(
    labels = percent_format(accuracy = 1),
    expand = expansion(mult = c(0, 0.05))
  ) +
  theme_classic(base_size = 14) +
  labs(
    x = "Morphology shape",
    y = "Cycling_high fraction"
  ) +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 30, hjust = 1)
  )

## ============================================================
## 2) Shannon entropy by morphology shape
## ============================================================

df_box_entropy <- build_box_stats(df_plot, "shannon_entropy")

p_entropy_box <- ggplot() +
  
  geom_rect(
    data = df_box_entropy,
    aes(xmin = xmin, xmax = xmax, ymin = q25, ymax = q75, fill = Shape_name),
    color = "black",
    linewidth = 0.6,
    alpha = 0.85
  ) +
  
  geom_segment(
    data = df_box_entropy,
    aes(x = x, xend = x, y = ymin, yend = ymax),
    linewidth = 0.6
  ) +
  
  geom_segment(
    data = df_box_entropy,
    aes(x = xmin, xend = xmax, y = mean, yend = mean),
    linewidth = 0.9
  ) +
  
  geom_jitter(
    data = df_plot,
    aes(x = as.numeric(Shape_name), y = shannon_entropy),
    width = 0.15,
    size = 1.2,
    color = "black",
    alpha = 0.6
  ) +
  
  scale_x_continuous(
    breaks = seq_along(shape_order),
    labels = shape_order
  ) +
  scale_fill_manual(values = shape_colors) +
  theme_classic(base_size = 14) +
  labs(
    x = "Morphology shape",
    y = "Shannon entropy"
  ) +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 30, hjust = 1)
  )

## ============================================================
## Output objects
## ============================================================
# p_cycling_box : Cycling_high fraction by morphology shape
# p_entropy_box : Shannon entropy by morphology shape
#
# Save with ggsave() as needed
## ============================================================

p_cycling_box
p_entropy_box
