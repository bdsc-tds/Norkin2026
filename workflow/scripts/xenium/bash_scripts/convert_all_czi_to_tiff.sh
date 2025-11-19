#!/bin/bash

# Path to your Python script and environment
PYTHON_SCRIPT="/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/workflow/scripts/xenium/morphology_code/czi_to_ome.py"
PYTHON_PATH="/work/PRTNR/CHUV/DIR/rgottar1/spatial/conda_envs/norkin_morphology/bin/python"

# Define runs array - can be pre-populated or empty
runs=("run_4_1")

# For pre-defined runs, get their indices from CORRESPONDENCES
echo "Using pre-defined runs, computing indices from CORRESPONDENCES..."

# First, declare the arrays
for run in "${runs[@]}"; do
    declare -a "indices_$run"
done

# Then populate them
for run in "${runs[@]}"; do
    # Get indices as a string and convert to array
    indices_string=$($PYTHON_PATH -c "
import sys
sys.path.append('/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/workflow/scripts/xenium/morphology_code/')
from czi_to_ome import CORRESPONDENCES

if '$run' in CORRESPONDENCES:
    id_list = CORRESPONDENCES['$run']
    length = len(id_list)
    indices = ' '.join(str(i) for i in range(length))
    print(indices)
")
    
    # Convert the string to an array
    eval "indices_$run=($indices_string)"
    
    echo "Run $run has indices: ${indices_string}"
done

# Loop through all combinations of parameters
for run in "${runs[@]}"; do
    # Get the indices array for this run
    var_name="indices_$run[@]"
    indices=("${!var_name}")
    
    for index in "${indices[@]}"; do
        echo "Running script with run=$run and index=$index"
        sbatch --wrap="$PYTHON_PATH $PYTHON_SCRIPT --run $run --index $index" --time=12:00:00 --mem=128G
    done
done

echo "Batch processing complete!"