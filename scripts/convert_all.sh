#!/bin/bash

# Define ranges for scale_factor, blur, row, and col
runs=("run_1" "run_2")  # Example: 1 (no scaling), 2 (half size), 4 (quarter size)
indices=(0 1 2 3 4 5 6 7)

# Loop through all combinations of parameters
for run in "${runs[@]}"; do
  for index in "${indices[@]}"; do
    echo "Running script with run=$run and index=$index"
    
    # Use the full path to Python in your conda environment
    sbatch --wrap="/work/PRTNR/CHUV/DIR/rgottar1/spatial/conda_envs/norkin_morphology/bin/python /work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/code/czi_to_ome.py --run $run --index $index" --time=12:00:00 --mem=128G
    
  done
done

echo "Batch processing complete!"