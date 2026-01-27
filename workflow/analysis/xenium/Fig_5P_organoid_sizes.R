library(dplyr)
library(ggplot2)

## ----------------------------
## 1) Subset adata
## ----------------------------


adata_malignant <- read_parquet(
  "adata_malignant.parquet"
)


df_cells <- adata_malignant %>%
  filter(
    condition == "CRC_PDO_CAF",
    donor_corrected %in% c("Pt_3", "Pt_3_CAFs"),
    !is.na(organoid_id)
  )

## ----------------------------
## 2) Count cells per organoid
## ----------------------------
df_counts <- df_cells %>%
  count(donor_corrected, organoid_id, name = "n_cells")

## ----------------------------
## 3) Factor order
## ----------------------------
df_counts$donor_corrected <- factor(
  df_counts$donor_corrected,
  levels = c("Pt_3", "Pt_3_CAFs")
)

## ----------------------------
## 4) Boxplot with MEAN replacing median (visually)
## ----------------------------
p_box <- ggplot(
  df_counts,
  aes(x = donor_corrected, y = n_cells, fill = donor_corrected)
) +
  ## boxplot WITHOUT median line
  geom_boxplot(
    width = 0.6,
    outlier.shape = NA,
    color = "black",
    median.linewidth = 0   # ⬅ hide median completely
  ) +
  ## MEAN line drawn where median would normally be
  stat_summary(
    fun = mean,
    geom = "crossbar",
    width = 0.6,
    color = "black",
    linewidth = 0.6,
    middle.linewidth = 0.6
  ) +
  ## individual organoids
  geom_jitter(
    width = 0.15,
    size = 2,
    alpha = 0.7,
    color = "black"
  ) +
  scale_fill_manual(
    values = c(
      "Pt_3"      = "#4daf4a",
      "Pt_3_CAFs" = "#E4211C"
    )
  ) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.1))) +
  labs(
    x = NULL,
    y = "Cells per organoid"
  ) +
  theme_classic(base_size = 12) +
  theme(
    legend.position = "none",
    plot.title    = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5)
  )

## ----------------------------
## 5) Draw
## ----------------------------
p_box