## ============================================================
## 0. Libraries
## ============================================================
library(arrow)
library(dplyr)
library(ggplot2)
library(ggpubr)
library(scales)

## ============================================================
## 1. Load data
## ============================================================

setwd("~/Downloads")

adata_malignant <- read_parquet(
  "adata_malignant_obs_bbknn_15.parquet"
)

clinical_data_xenium <- read.csv(
  "clinical_data_xenium.csv",
  stringsAsFactors = FALSE
)

## ============================================================
## 2. Prepare clinical table (1 row per patient)
## ============================================================

clinical_clean <- clinical_data_xenium %>%
  distinct(Patient_ID, .keep_all = TRUE)

## ============================================================
## 3. Join clinical annotations into adata
## ============================================================

adata_malignant <- adata_malignant %>%
  left_join(
    clinical_clean,
    by = c("donor_corrected" = "Patient_ID")
  )

## ============================================================
## 4. Define program mapping (Leiden 0.8 bbknn)
## ============================================================

cluster_map_08 <- c(
  "0" = "Cycling_high",
  "1" = "Cycling_high",
  "2" = "Cycling_high",
  "3" = "Transit-amplifying",
  "4" = "Chemokine_Inflammatory",
  "5" = "Goblet_MUC",
  "6" = "Chemokine_EMT",
  "7" = "Stem-like",
  "8" = "Goblet_Inflammatory",
  "9" = "Senescent_Inflammatory"
)

program_colors <- c(
  "Cycling_high"            = "#17BECF",
  "Goblet_Inflammatory"     = "#FE3FFA",
  "Transit-amplifying"      = "#0433FF",
  "Stem-like"               = "#F6252F",
  "Chemokine_EMT"           = "#FEA501",
  "Goblet_MUC"              = "#C77CFF",
  "Chemokine_Inflammatory"  = "#2CA02C",
  "Senescent_Inflammatory"  = "#FFEEAA"
)

## ============================================================
## 5. Assign ProgramCluster
## ============================================================

adata_malignant <- adata_malignant %>%
  mutate(
    ProgramCluster = cluster_map_08[as.character(leiden_0.8_bbknn)]
  )

stopifnot(!any(is.na(adata_malignant$ProgramCluster)))

## ============================================================
## 6. Compute donor-level program fractions (CRC_PDO only)
##    Uses clinical Site (Primary / Metastasis)
## ============================================================

prog_frac_donor <- adata_malignant %>%
  filter(
    condition == "CRC_PDO",
    Site %in% c("Primary", "Metastasis")
  ) %>%
  count(donor_corrected, ProgramCluster, Site) %>%
  group_by(donor_corrected) %>%
  mutate(frac = n / sum(n)) %>%
  ungroup()

## ============================================================
## 7. Colors
## ============================================================

site_colors <- c(
  "Primary"    = "#1f77b4",
  "Metastasis" = "#d62728"
)

## ============================================================
## 8. Plot function (stable, publication-safe)
## ============================================================

plot_program_site <- function(program_name, df) {
  
  df_use <- df %>% filter(ProgramCluster == program_name)
  
  ggplot(
    df_use,
    aes(x = Site, y = frac, fill = Site, color = Site)
  ) +
    geom_boxplot(
      width = 0.6,
      outlier.shape = NA,
      alpha = 0.35,
      size = 1
    ) +
    geom_jitter(
      width = 0.15,
      size = 2.5,
      alpha = 0.85
    ) +
    stat_compare_means(
      comparisons = list(c("Primary", "Metastasis")),
      method = "wilcox.test",
      aes(
        label = ifelse(
          ..p.. < 0.05,
          paste0(..p.signif.., "  p = ", signif(..p.., 2)),
          paste0("p = ", signif(..p.., 2))
        )
      ),
      size = 4.5,
      tip.length = 0.02
    ) +
    scale_fill_manual(values = site_colors) +
    scale_color_manual(values = site_colors) +
    scale_y_continuous(
      labels = percent_format(accuracy = 1),
      expand = expansion(mult = c(0, 0.15))
    ) +
    labs(
      x = NULL,
      y = paste0(program_name, " fraction"),
      title = paste0(program_name, " program fraction per CRC_PDO donor")
    ) +
    theme_classic(base_size = 14) +
    theme(
      legend.position = "none",
      plot.title = element_text(face = "bold")
    )
}

## ============================================================
## 9. Generate plots for all programs
## ============================================================

programs <- names(program_colors)

plots_programs <- lapply(
  programs,
  plot_program_site,
  df = prog_frac_donor
)

names(plots_programs) <- programs

## ============================================================
## 10. Display (QC)
## ============================================================

for (p in names(plots_programs)) {
  print(plots_programs[[p]])
}

## ============================================================
## 11. Save plots (high quality)
## ============================================================

outdir <- "~/Desktop/program_site_boxplots_leiden08"
dir.create(outdir, showWarnings = FALSE)

for (p in names(plots_programs)) {
  ggsave(
    filename = file.path(outdir, paste0(p, "_Primary_vs_Metastasis.png")),
    plot = plots_programs[[p]],
    width = 4.5,
    height = 5,
    dpi = 300
  )
}
