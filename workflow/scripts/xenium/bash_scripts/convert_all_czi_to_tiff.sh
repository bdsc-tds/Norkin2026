#!/bin/bash

# Define ranges for scale_factor, blur, row, and col
# runs=("run_1" "run_2")  # Example: 1 (no scaling), 2 (half size), 4 (quarter size)
# indices=(0 1 2 3 4 5 6 7)
runs=("run_1" "run_2")  # Example: 1 (no scaling), 2 (half size), 4 (quarter size)
indices=(1 2 3 4 5 6 6 7)
# methods=(1 2 3)

# Loop through all combinations of parameters
for run in "${runs[@]}"; do
  for index in "${indices[@]}"; do
    # for method in "${methods[@]}"; do
      echo "Running script with run=$run and index=$index and method=$method"
      # sbatch --wrap="dcsrsoft use 20241118; module load miniforge3; conda_init; conda activate prometex; python3.10 run_registration.py --scale_factor $scale_factor --blur $blur --row $row --col $col" --time=12:00:00 --mem=16G
      # sbatch --wrap="conda activate /work/PRTNR/CHUV/DIR/rgottar1/spatial/conda_envs/norkin_morphology; python /work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/code/czi_to_ome.py --run $run --index $index --method $method" --time=12:00:00 --mem=128G
      sbatch --wrap="python /work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/code/czi_to_ome.py --run $run --index $index" --time=12:00:00 --mem=128G
    # done
  done
done

echo "Batch processing complete!"