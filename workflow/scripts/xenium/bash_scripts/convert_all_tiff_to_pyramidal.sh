#!/bin/bash

# Define the CORRESPONDENCES object
declare -A CORRESPONDENCES=(
    # ["run_1_1"]="0Y6H 19II 14VS OAFN 1CFW 12WP OUC1"
    # ["run_1_2"]="03FO OYRI 1H3R OWMY 1DCI 1EGQ O056"
    # ["run_2_1"]="1HVQ 1CNN 077I 1GAA 1J25 131N OWJ3 14PT"
    # ["run_2_2"]="169V 1BI7 1CI5 1FMS 12NM 0LR9 1GVB 1GNS"
    # ["run_3"]="0Z84 1JET 4_1HVQ_big 5_1HVQ_big 7_OY6Hsmall 8_OY6Hsmallmiddle 9_OY6H_middle_and_big 10_OY6Hmiddlebig 11_OY6H_middle_and_big 12_OY6Hbighuge"
    # ["run_4_1"]="1HVQ_big 1HVQ_big_preview 1HVQ_big_CAFs 1HVQ_big_CAFs_preview 0WFQ_big 0WFQ_big_preview 1DDI 1DDI_preview 07WM 07WM_preview 1CFV 1CFV_preview 1HVC 1HVC_preview OUC4 OUC4_preview 12I1 12I1_preview"
    # ["run_4_1"]="1HVQ_big 1HVQ_big_CAFs 0WFQ_big 1DDI 07WM 1CFV 1HVC OUC4 12I1"
    ["run_4_1"]="1DDI 0WFQ 1HVQ_big 07WM 1CFV 1HVC 12I1 OUC4 1HVQ_big_CAFs"
    # ["run_4_2"]="1H3R_drug 1DDI_CAFs 1H3R_2_drug 07WM_CAFs 1H3R_ctrl 1DDI 1H3R"
)

# Base directories
INPUT_BASE="/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/ome_tiff"
OUTPUT_BASE="/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/ome_tiff_pyr"
BFTOOLS_PATH="/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/bftools/bfconvert"

# Create output directories if they don't exist
mkdir -p "$OUTPUT_BASE/run_1_1"
mkdir -p "$OUTPUT_BASE/run_1_2"
mkdir -p "$OUTPUT_BASE/run_2_1"
mkdir -p "$OUTPUT_BASE/run_2_2"
mkdir -p "$OUTPUT_BASE/run_3"
mkdir -p "$OUTPUT_BASE/run_4_1"
mkdir -p "$OUTPUT_BASE/run_4_2"

# Loop through all runs and their corresponding files
for run in "${!CORRESPONDENCES[@]}"; do
    IFS=' ' read -ra files <<< "${CORRESPONDENCES[$run]}"
    
    for file in "${files[@]}"; do
        input_file="$INPUT_BASE/$run/${file}.ome.tiff"
        output_file="$OUTPUT_BASE/$run/${file}.ome.tiff"
        
        # Check if input file exists
        if [[ ! -f "$input_file" ]]; then
            echo "Warning: Input file $input_file does not exist, skipping..."
            continue
        fi
        
        # Create the sbatch command
        # sbatch_command="sbatch --wrap=\"$BFTOOLS_PATH -bigtiff -series 0 -pyramid-resolutions 4 -pyramid-scale 2 -tilex 256 -tiley 256 '$input_file' '$output_file'\" --time=12:00:00 --mem=64G"
        sbatch_command="sbatch --wrap=\"$BFTOOLS_PATH -bigtiff -series 0 -pyramid-resolutions 7 -pyramid-scale 2 -tilex 1024 -tiley 1024 '$input_file' '$output_file'\" --time=12:00:00 --mem=64G"
        
        echo "Submitting job for: $input_file -> $output_file"
        eval "$sbatch_command"
        
        # Small delay to avoid overwhelming the scheduler
        sleep 1
    done
done

echo "All pyramid conversion jobs submitted!"