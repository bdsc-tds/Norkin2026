#!/bin/bash

# Path to your Python script and environment
PYTHON_SCRIPT="/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/workflow/scripts/xenium/morphology_code/czi_to_ome_region_based.py"
PYTHON_PATH="/work/PRTNR/CHUV/DIR/rgottar1/spatial/conda_envs/norkin_morphology/bin/python"

# Path to regions CSV file
REGION_CSV="/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/norkin_organoid/data/xenium/metadata/Regions_coordinates_18samples.csv"

# Output directory
OUTPUT_DIR="/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/ome_tiff/run_4_1"

# Get the number of rows in the CSV file
echo "Getting number of rows from CSV file..."

NUM_ROWS=$($PYTHON_PATH -c "
import pandas as pd
df = pd.read_csv('$REGION_CSV')
print(len(df))
")

echo "Found $NUM_ROWS regions to process"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/logs

# Loop through all row indices and submit jobs
for ((row_index=0; row_index<NUM_ROWS; row_index++)); do
    echo "Submitting job for row index: $row_index"
    sbatch \
        --job-name="czi_${row_index}" \
        --output="${OUTPUT_DIR}/logs/czi_${row_index}.%j.out" \
        --error="${OUTPUT_DIR}/logs/czi_${row_index}.%j.err" \
        --time=12:00:00 \
        --mem=128G \
        --wrap="$PYTHON_PATH $PYTHON_SCRIPT --row_index $row_index --region_csv '$REGION_CSV' --output_dir '$OUTPUT_DIR'"
    
    # Small delay to avoid overwhelming the scheduler
    sleep 1
done

echo "All $NUM_ROWS jobs submitted! Monitoring queue with: squeue -u \$USER"