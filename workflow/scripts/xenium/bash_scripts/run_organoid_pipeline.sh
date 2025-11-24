#!/bin/bash
# Individual job submission version

ALIGNMENTS_ROOT="/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/alignments"

# Define sample filter - add sample IDs here to filter, leave empty to process all samples
SAMPLES_FILTER=(
    "1HVQ_big_CAFs"
    "1HVQ_big"
    "1DDI_CAFs"
    "1H3R_2_ctrl"
)
# Example: SAMPLES_FILTER=("sample1" "sample2" "sample3")

# Build patient and run arrays
SAMPLES=()
RUNS=()

echo "Scanning alignment directories..."
for run_dir in "$ALIGNMENTS_ROOT"/run_*; do
    run_name=$(basename "$run_dir")
    for sample_dir in "$run_dir"/*_qupath_alignment_files; do
        if [ -d "$sample_dir" ]; then
            sample_id=$(basename "$sample_dir" _qupath_alignment_files)
            
            # Apply filter if SAMPLES_FILTER is not empty
            if [ ${#SAMPLES_FILTER[@]} -eq 0 ]; then
                # No filter, include all samples
                SAMPLES+=("$sample_id")
                RUNS+=("$run_name")
            else
                # Check if sample_id is in SAMPLES_FILTER
                for filtered_sample in "${SAMPLES_FILTER[@]}"; do
                    if [ "$sample_id" == "$filtered_sample" ]; then
                        SAMPLES+=("$sample_id")
                        RUNS+=("$run_name")
                        break
                    fi
                done
            fi
        fi
    done
done

echo "Found samples: ${SAMPLES[*]}"
echo "Found runs: ${RUNS[*]}"

# Create directories
mkdir -p /work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/organoids_h\&e/images/
mkdir -p logs

echo "=== Generating organoid manifest..."
/work/PRTNR/CHUV/DIR/rgottar1/spatial/conda_envs/lazyslide_env/bin/python /work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/workflow/scripts/xenium/morphology_code/generate_manifest.py "${SAMPLES[@]}" "${RUNS[@]}"

MANIFEST_CSV="/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/organoid_manifest.csv"
TOTAL_JOBS=$(tail -n +2 "$MANIFEST_CSV" | wc -l)

echo "Submitting $TOTAL_JOBS individual jobs..."

# Read manifest and submit jobs
INDEX=0
while IFS=, read -r sample_id organoid_id status run_name; do
    # Skip header
    if [[ "$sample_id" == "sample_id" ]]; then
        continue
    fi
    
    # Remove quotes if present
    sample_id=$(echo "$sample_id" | tr -d '"')
    organoid_id=$(echo "$organoid_id" | tr -d '"')
    run_name=$(echo "$run_name" | tr -d '"')
    
    # Submit individual job
    sbatch << EOF
#!/bin/bash
#SBATCH --job-name=${sample_id}_${INDEX}_${organoid_id}
#SBATCH --output=logs/${sample_id}_${INDEX}_${organoid_id}.out
#SBATCH --error=logs/${sample_id}_${INDEX}_${organoid_id}.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1

/work/PRTNR/CHUV/DIR/rgottar1/spatial/conda_envs/lazyslide_env/bin/python /work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/workflow/scripts/xenium/morphology_code/extract_organoid.py "$sample_id" "$organoid_id" "$run_name"
EOF

    ((INDEX++))
done < "$MANIFEST_CSV"

echo "Submitted $INDEX individual jobs"