## ============================================================
## Libraries
## ============================================================

library(arrow)
library(dplyr)
library(ggplot2)
library(scales)
library(tidyr)

## ============================================================
## Load data
## ============================================================

adata_malignant <- read_parquet(
  "adata_malignant.parquet"
)


## ============================================================
## Program colors
## ============================================================

program_colors <- c(
  "Cycling_high"           = "#17BECF",
  "Transit-amplifying"     = "#0433FF",
  "Goblet_MUC"             = "#C77CFF",
  "Goblet_Inflammatory"    = "#FE3FFA",
  "Senescent_Inflammatory" = "#FFEEAA",
  "Chemokine_Inflammatory" = "#2CA02C",
  "Chemokine_EMT"          = "#FEA501",
  "Stem-like"              = "#F6252F"
)


## ============================================================
## Subset CRC_PDO organoids with valid annotations
## ============================================================

adata_org <- adata_malignant %>%
  filter(
    condition == "CRC_PDO",
    !is.na(organoid_id),
    !is.na(ProgramCluster)
  )

## ============================================================
## Compute organoid-level Shannon entropy
## ============================================================

entropy_organoid <- adata_org %>%
  count(organoid_id, ProgramCluster) %>%
  group_by(organoid_id) %>%
  mutate(p = n / sum(n)) %>%
  summarise(
    entropy = -sum(p * log(p)),
    .groups = "drop"
  )

## ============================================================
## Order organoids by increasing entropy
## ============================================================

organoid_order_entropy <- entropy_organoid %>%
  arrange(entropy) %>%
  pull(organoid_id)

## ============================================================
## Program frequency per organoid
## ============================================================

freq_organoid <- adata_org %>%
  count(organoid_id, ProgramCluster) %>%
  group_by(organoid_id) %>%
  mutate(freq = n / sum(n)) %>%
  ungroup() %>%
  mutate(
    organoid_id    = factor(organoid_id, levels = organoid_order_entropy),
    ProgramCluster = factor(ProgramCluster, levels = names(program_colors))
  )

## ============================================================
## Plot: Program composition ordered by entropy
## ============================================================

p_org_entropy_clean <- ggplot(
  freq_organoid,
  aes(x = organoid_id, y = freq, fill = ProgramCluster)
) +
  geom_col(width = 1) +
  scale_fill_manual(values = program_colors) +
  scale_y_continuous(
    labels = percent_format(accuracy = 1),
    expand = c(0, 0)
  ) +
  scale_x_discrete(expand = c(0, 0)) +
  theme_classic(base_size = 12) +
  theme(
    axis.text.x  = element_blank(),
    axis.ticks.x = element_blank(),
    axis.title   = element_blank(),
    legend.position = "right"
  )

p_org_entropy_clean
