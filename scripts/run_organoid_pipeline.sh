#!/bin/bash
# Individual job submission version
# Usage: ./run_organoid_pipeline_individual.sh [patient1] [patient2] ...

# Set default patients if none provided
if [ $# -eq 0 ]; then
    PATIENTS=("1CNN" "1GAA" "1GVB" "1J25" "14PT" "131N")
    echo "No patients specified, using defaults: ${PATIENTS[*]}"
else
    PATIENTS=("$@")
    echo "Using specified patients: ${PATIENTS[*]}"
fi

# Create directories
mkdir -p /work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/organoids_h&e/images/
mkdir -p logs

echo "=== Generating organoid manifest..."
/work/PRTNR/CHUV/DIR/rgottar1/spatial/conda_envs/lazyslide_env/bin/python /work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/code/generate_manifest.py "${PATIENTS[@]}"

MANIFEST_CSV="/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/organoid_manifest.csv"
TOTAL_JOBS=$(tail -n +2 "$MANIFEST_CSV" | wc -l)

echo "Submitting $TOTAL_JOBS individual jobs..."

# Read manifest and submit jobs
INDEX=0
while IFS=, read -r patient_id organoid_id status; do
    # Skip header
    if [[ "$patient_id" == "patient_id" ]]; then
        continue
    fi
    
    # Remove quotes if present
    patient_id=$(echo "$patient_id" | tr -d '"')
    organoid_id=$(echo "$organoid_id" | tr -d '"')
    
    # Submit individual job
    sbatch << EOF
#!/bin/bash
#SBATCH --job-name=org_${patient_id}_${INDEX}
#SBATCH --output=logs/org_${patient_id}_${INDEX}.out
#SBATCH --error=logs/org_${patient_id}_${INDEX}.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1

/work/PRTNR/CHUV/DIR/rgottar1/spatial/conda_envs/lazyslide_env/bin/python /work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/code/extract_organoid.py "$patient_id" "$organoid_id"
EOF

    ((INDEX++))
done < "$MANIFEST_CSV"

echo "Submitted $TOTAL_JOBS individual jobs"