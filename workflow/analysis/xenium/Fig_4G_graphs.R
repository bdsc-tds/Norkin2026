library(dplyr)
library(ggplot2)

## ============================================================
## 0️⃣ PARAMETERS
## ============================================================

priority_donors <- c("Pt_22", "Pt_14")
exclude_donors  <- c("Pt_39")

source_levels <- c("Healthy reference", "Tumor region", "PDO")

## ============================================================
## 1️⃣ VALID DONORS (TUMOR ∩ ORGANOIDS)
## ============================================================

donors_mm=readRDS("proseg_expected_CRC_PDO_hImmune_v1_mm_lognorm.rds")
donors_dapi=readRDS("proseg_expected_CRC_PDO_hImmune_v1_dapi_lognorm.rds")
tumor=readRDS()

donors_mm   <- unique(as.character(organoids_mm$donor_corrected))
donors_dapi <- unique(as.character(organoids_dapi$donor_corrected))
organoid_donors <- union(donors_mm, donors_dapi)

tumor_donors <- unique(as.character(tumor$donor))

valid_donors <- intersect(tumor_donors, organoid_donors)
valid_donors <- setdiff(valid_donors, exclude_donors)

stopifnot(all(priority_donors %in% valid_donors))
message("✔ Using ", length(valid_donors), " matched donors")

## ============================================================
## 2️⃣ TUMOR DOTS = TAILS
## ============================================================

## ============================================================
## 2️⃣ TUMOR DOTS = 100-SPOT SPATIAL REGIONS (REPLACES TAILS)
## ============================================================

library(Seurat)
library(dplyr)

nx <- 10
ny <- 10
min_cells_per_region <- 30

region_scores <- list()

for (fov_name in Images(tumor)) {
  
  message("Processing tumor FOV ", fov_name)
  
  cells  <- Cells(tumor[[fov_name]])
  coords <- tumor@images[[fov_name]]@boundaries$centroids@coords
  
  df <- data.frame(
    cell_id    = cells,
    x          = coords[,1],
    y          = coords[,2],
    diff_score = tumor$diff_score[cells],
    donor      = tumor$donor[cells]
  )
  
  df <- df %>%
    filter(
      !is.na(diff_score),
      donor %in% valid_donors
    )
  
  if (nrow(df) == 0) next
  
  df <- df %>%
    mutate(
      x_bin = cut(x, breaks = nx, labels = FALSE),
      y_bin = cut(y, breaks = ny, labels = FALSE),
      region_id = paste0("R", x_bin, "_", y_bin)
    )
  
  df_region <- df %>%
    group_by(donor, fov = fov_name, region_id) %>%
    summarise(
      mean_diff = mean(diff_score),
      n_cells   = n(),
      .groups   = "drop"
    ) %>%
    filter(n_cells >= min_cells_per_region)
  
  if (nrow(df_region) == 0) next
  
  region_scores[[fov_name]] <- df_region
}

region_scores_df <- bind_rows(region_scores)

tumor_df <- region_scores_df %>%
  transmute(
    donor_id  = donor,
    source    = "Tumor region",
    mean_diff = mean_diff
  )

## ============================================================
## 3️⃣ PDO DOTS = ORGANOID_ID
## ============================================================

pdo_mm_df <- organoids_mm@meta.data %>%
  filter(donor_corrected %in% valid_donors) %>%
  group_by(
    donor_id = donor_corrected,
    organoid_id
  ) %>%
  summarise(
    mean_diff = mean(diff_score, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(source = "PDO")

pdo_dapi_df <- organoids_dapi@meta.data %>%
  filter(donor_corrected %in% valid_donors) %>%
  group_by(
    donor_id = donor_corrected,
    organoid_id
  ) %>%
  summarise(
    mean_diff = mean(diff_score, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(source = "PDO")

## ============================================================
## 4️⃣ COMBINE ALL DOTS
## ============================================================

plot_df <- bind_rows(
  tumor_df,
  pdo_mm_df,
  pdo_dapi_df
)

stopifnot(all(plot_df$donor_id %in% valid_donors))

## ============================================================
## 5️⃣ DONOR DIFFERENTIATION STRIPE (CLINICAL DATA)
## ============================================================

donor_diff <- clinical_data_xenium %>%
  transmute(
    donor_id = Patient_ID,
    differentiation = case_when(
      differentiation == "well differentiated" ~ "well differentiated",
      differentiation %in% c("intermediate", "poorly/intermediate") ~ "intermediate / poorly",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(donor_id %in% valid_donors) %>%
  distinct(donor_id, .keep_all = TRUE)

## ============================================================
## 6️⃣ DONOR ORDER (TUMOR MEDIAN, PRIORITY FIRST)
## ============================================================

donor_scores <- plot_df %>%
  filter(source == "Tumor region") %>%
  group_by(donor_id) %>%
  summarise(
    tumor_median_diff = median(mean_diff, na.rm = TRUE),
    .groups = "drop"
  )

ordered_rest <- donor_scores %>%
  filter(!donor_id %in% priority_donors) %>%
  arrange(desc(tumor_median_diff)) %>%
  pull(donor_id)

donor_order_final <- c(
  priority_donors,
  ordered_rest,
  setdiff(valid_donors, c(priority_donors, ordered_rest))
)

## ============================================================
## 7️⃣ X MAP (Healthy → Tumor → PDO)
## ============================================================

x_map <- plot_df %>%
  distinct(donor_id, source) %>%
  mutate(
    donor_id = factor(donor_id, levels = donor_order_final),
    source   = factor(source, levels = source_levels)
  ) %>%
  arrange(donor_id, source) %>%
  mutate(x = row_number())

## ============================================================
## 8️⃣ MERGE X + STRIPES
## ============================================================

plot_df2 <- plot_df %>%
  left_join(x_map, by = c("donor_id", "source")) %>%
  filter(!is.na(x)) %>%
  mutate(group_id = interaction(donor_id, source, drop = TRUE))

stripe_df <- x_map %>%
  group_by(donor_id) %>%
  summarise(
    xmin = min(x) - 0.5,
    xmax = max(x) + 0.5,
    .groups = "drop"
  ) %>%
  left_join(donor_diff, by = "donor_id")

y_min <- min(plot_df2$mean_diff, na.rm = TRUE)
y_max <- max(plot_df2$mean_diff, na.rm = TRUE)

stripe_df <- stripe_df %>%
  mutate(
    ymin = y_min - 0.60,
    ymax = y_min - 0.35
  )

## ============================================================
## 9️⃣ FINAL PLOT
## ============================================================

p_final <- ggplot(
  plot_df2,
  aes(x = x, y = mean_diff, fill = source, color = source, group = group_id)
) +
  geom_rect(
    data = stripe_df,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = differentiation),
    inherit.aes = FALSE,
    color = NA
  ) +
  geom_boxplot(width = 0.7, outlier.shape = NA, alpha = 0.35) +
  geom_jitter(width = 0.15, size = 1.1, alpha = 0.7) +
  scale_x_continuous(
    breaks = (stripe_df$xmin + stripe_df$xmax) / 2,
    labels = stripe_df$donor_id,
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  scale_fill_manual(
    values = c(
      "Healthy reference"     = "#2166ac",
      "Tumor region"          = "#b2182b",
      "PDO"                   = "#000000",
      "well differentiated"   = "#4daf4a",
      "intermediate / poorly" = "#ffcc00"
    ),
    na.value = "#DDDDDD"
  ) +
  scale_color_manual(
    values = c(
      "Healthy reference" = "#2166ac",
      "Tumor region"      = "#b2182b",
      "PDO"               = "#000000"
    )
  ) +
  coord_cartesian(
    ylim = c(min(stripe_df$ymin) - 0.1, y_max),
    clip = "off"
  ) +
  labs(
    x = "Donor",
    y = "Mean diff_score",
    title = "Tumor differentiation with healthy reference and PDOs"
  ) +
  theme_classic(base_size = 14) +
  theme(
    axis.text.x  = element_text(angle = 45, hjust = 1),
    legend.title = element_blank(),
    plot.margin  = margin(t = 10, r = 10, b = 35, l = 10)
  )

p_final
