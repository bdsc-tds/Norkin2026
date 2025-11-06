#!/bin/bash

# Define the CORRESPONDENCES object
declare -A CORRESPONDENCES=(
    # ["run_1"]="1HVQ 1CNN 077I 1GAA 1J25 131N OWJ3 14PT"
    # ["run_2"]="169V 1BI7 1CI5 1FMS 12NM 0LR9 1GVB 1GNS"
    # ["run_1"]="1HVQ"
    # ["run_2"]="169V"
    # ["run_3"]="0Z84 1JET 4_1HVQ_big 5_1HVQ_big 7_OY6Hsmall 8_OY6Hsmallmiddle 9_OY6H_middle_and_big 10_OY6Hmiddlebig 11_OY6H_middle_and_big 12_OY6Hbighuge"
    ["run_4_1"]="1HVQ_big 1HVQ_big_CAFs 0WFQ_big 1DDI 07WM 1CFV 1HVC OUC4 12I1"
)

# Base directories
INPUT_BASE="/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/ome_tiff"
OUTPUT_BASE="/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/ome_tiff_pyr"
BFTOOLS_PATH="/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/bftools/bfconvert"

# Create output directories if they don't exist
mkdir -p "$OUTPUT_BASE/run_1"
mkdir -p "$OUTPUT_BASE/run_2"
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
        sbatch_command="sbatch --wrap=\"$BFTOOLS_PATH -bigtiff -series 0 -pyramid-resolutions 4 -pyramid-scale 2 -tilex 256 -tiley 256 '$input_file' '$output_file'\" --time=12:00:00 --mem=64G"
        
        echo "Submitting job for: $input_file -> $output_file"
        eval "$sbatch_command"
        
        # Small delay to avoid overwhelming the scheduler
        sleep 1
    done
done

echo "All pyramid conversion jobs submitted!"