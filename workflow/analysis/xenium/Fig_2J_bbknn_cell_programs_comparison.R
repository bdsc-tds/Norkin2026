library(arrow)
library(dplyr)
library(ggplot2)
library(scales)

# ============================================================
# 0️⃣ Load data
# ============================================================

setwd("~/Downloads")

adata_malignant <- read_parquet(
  "adata_malignant_obs_bbknn_15.parquet"
)

# ============================================================
# 1️⃣ Program mapping + colors (Leiden 0.8)
# ============================================================

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
  "Transit-amplifying"      = "#0433FF",
  "Goblet_MUC"              = "#C77CFF",
  "Goblet_Inflammatory"     = "#FE3FFA",
  "Senescent_Inflammatory"  = "#FFEEAA",
  "Chemokine_Inflammatory"  = "#2CA02C",
  "Chemokine_EMT"           = "#FEA501",
  "Stem-like"               = "#F6252F"
)

# ============================================================
# 2️⃣ Identify overlapping donors (CRC + CRC_PDO)
# ============================================================

donors_overlap <- adata_malignant %>%
  filter(condition %in% c("CRC", "CRC_PDO")) %>%
  distinct(donor_corrected, condition) %>%
  count(donor_corrected) %>%
  filter(n == 2) %>%
  pull(donor_corrected)

# ============================================================
# 3️⃣ Prepare data (map Leiden 0.8 → programs)
# ============================================================

adata_08 <- adata_malignant %>%
  mutate(
    ProgramCluster = cluster_map_08[as.character(leiden_0.8_bbknn)]
  ) %>%
  filter(
    donor_corrected %in% donors_overlap,
    condition %in% c("CRC", "CRC_PDO"),
    !is.na(ProgramCluster)
  )

# ============================================================
# 4️⃣ Compute PDO Shannon entropy
# ============================================================

entropy_df <- adata_08 %>%
  filter(condition == "CRC_PDO") %>%
  group_by(donor_corrected, ProgramCluster) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(donor_corrected) %>%
  mutate(p = n / sum(n)) %>%
  summarise(entropy = -sum(p * log(p)), .groups = "drop")

donor_order_entropy <- entropy_df %>%
  arrange(desc(entropy)) %>%
  pull(donor_corrected)

# ============================================================
# 5️⃣ Frequency table
# ============================================================

freq_08 <- adata_08 %>%
  group_by(donor_corrected, condition, ProgramCluster) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(donor_corrected, condition) %>%
  mutate(freq = n / sum(n)) %>%
  ungroup() %>%
  mutate(
    donor_corrected = factor(donor_corrected, levels = donor_order_entropy),
    condition       = factor(condition, levels = c("CRC", "CRC_PDO")),
    ProgramCluster  = factor(ProgramCluster, levels = names(program_colors))
  )

# ============================================================
# 6️⃣ Final barplot
# ============================================================

p08_entropy <- ggplot(freq_08, aes(
  x    = condition,
  y    = freq,
  fill = ProgramCluster
)) +
  geom_col(width = 0.9) +
  scale_fill_manual(values = program_colors) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  facet_wrap(~ donor_corrected, nrow = 1) +
  labs(
    title    = "CRC vs PDO – Program composition (Leiden 0.8)",
    subtitle = "Donors with matched CRC & PDO, ordered by PDO Shannon entropy",
    x        = NULL,
    y        = "Percent of cells",
    fill     = "Program cluster"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    strip.text          = element_text(face = "bold", size = 11),
    axis.text.x         = element_text(size = 7),
    panel.grid.major.x  = element_blank(),
    legend.position     = "right"
  )

p08_entropy
