#!/usr/bin/env python3
"""
Generate a CSV manifest of all organoids to process.
Usage: python generate_organoid_manifest.py [patient1] [patient2] ...
"""

import sys
import pandas as pd

sys.path.append("/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1")
from norkin_organoid.workflow.scripts.xenium.morphology_code.get_embeddings import NorkinOrganoidDataset

# Default patients if none provided
DEFAULT_PATIENTS = ["1CNN", "1GAA", "1GVB", "1J25", "14PT", "131N"]
OUTPUT_CSV = "/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/organoid_manifest.csv"
ORGANOID_ID_COLUMN_KEY = "component_and_cluster_and_lasso"

def generate_manifest(patient_ids):
    """Generate CSV with all patient-organoid combinations."""
    dataset = NorkinOrganoidDataset(standardize_scale=False, scale=True, fill=True, use_cached_adata=False, use_cached_masks=False, organoid_id_column_key=ORGANOID_ID_COLUMN_KEY)
    
    manifest_data = []
    
    for patient_id in patient_ids:
        print(f"Processing patient: {patient_id}")
        
        try:
            joined_df = dataset.get_organoid_df_by_id(patient_id=patient_id)
            organoid_ids = joined_df[dataset.organoid_id_column_key].unique().tolist()
            
            for organoid_id in organoid_ids:
                manifest_data.append({
                    'patient_id': patient_id,
                    'organoid_id': organoid_id,
                    'status': 'pending'
                })
                
            print(f"  Found {len(organoid_ids)} organoids for patient {patient_id}")
            
        except Exception as e:
            print(f"  Error processing patient {patient_id}: {e}")
    
    # Create DataFrame and save
    manifest_df = pd.DataFrame(manifest_data)
    manifest_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Manifest saved to: {OUTPUT_CSV}")
    print(f"Total organoids: {len(manifest_df)}")
    
    return manifest_df

if __name__ == "__main__":
    # Use provided patient IDs or defaults
    if len(sys.argv) > 1:
        patient_ids = sys.argv[1:]
    else:
        patient_ids = DEFAULT_PATIENTS
        
    print(f"Generating manifest for patients: {patient_ids}")
    manifest = generate_manifest(patient_ids)