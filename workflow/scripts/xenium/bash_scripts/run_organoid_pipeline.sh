#!/bin/bash
# Individual job submission version

ALIGNMENTS_ROOT="/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/alignments"

# Build patient and run arrays
PATIENTS=()
RUNS=()

echo "Scanning alignment directories..."
for run_dir in "$ALIGNMENTS_ROOT"/run_*; do
    run_name=$(basename "$run_dir")
    for patient_dir in "$run_dir"/*_qupath_alignment_files; do
        if [ -d "$patient_dir" ]; then
            patient_id=$(basename "$patient_dir" _qupath_alignment_files)
            PATIENTS+=("$patient_id")
            RUNS+=("$run_name")
        fi
    done
done

echo "Found patients: ${PATIENTS[*]}"

# Create directories
mkdir -p /work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/organoids_h\&e/images/
mkdir -p logs

echo "=== Generating organoid manifest..."
/work/PRTNR/CHUV/DIR/rgottar1/spatial/conda_envs/lazyslide_env/bin/python /work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/workflow/scripts/xenium/morphology_code/generate_manifest.py "${PATIENTS[@]}" "${RUNS[@]}"

# MANIFEST_CSV="/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/organoid_manifest.csv"
# TOTAL_JOBS=$(tail -n +2 "$MANIFEST_CSV" | wc -l)

# echo "Submitting $TOTAL_JOBS individual jobs..."

# # Read manifest and submit jobs
# INDEX=0
# while IFS=, read -r patient_id organoid_id status run_name; do
#     # Skip header
#     if [[ "$patient_id" == "patient_id" ]]; then
#         continue
#     fi
    
#     # Remove quotes if present
#     patient_id=$(echo "$patient_id" | tr -d '"')
#     organoid_id=$(echo "$organoid_id" | tr -d '"')
#     run_name=$(echo "$run_name" | tr -d '"')
    
#     # Submit individual job
#     sbatch << EOF
# #!/bin/bash
# #SBATCH --job-name=org_${patient_id}_${INDEX}_${organoid_id}
# #SBATCH --output=logs/org_${patient_id}_${INDEX}_${organoid_id}.out
# #SBATCH --error=logs/org_${patient_id}_${INDEX}_${organoid_id}.err
# #SBATCH --time=04:00:00
# #SBATCH --mem=16G
# #SBATCH --cpus-per-task=1

# /work/PRTNR/CHUV/DIR/rgottar1/spatial/conda_envs/lazyslide_env/bin/python /work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/workflow/scripts/xenium/morphology_code/extract_organoid.py "$patient_id" "$organoid_id" "$run_name"
# EOF

#     ((INDEX++))
# done < "$MANIFEST_CSV"

# echo "Submitted $INDEX individual jobs"