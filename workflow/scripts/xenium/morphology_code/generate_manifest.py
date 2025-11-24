#!/usr/bin/env python3
"""
Generate a CSV manifest of all organoids to process.
Usage: python generate_organoid_manifest.py sample1 sample2 ... run1 run2 ...
"""

import sys
import pandas as pd

sys.path.append("/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1")
from norkin_organoid.workflow.scripts.xenium.morphology_code.get_embeddings import NorkinOrganoidDataset

OUTPUT_CSV = "/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/organoid_manifest.csv"
ORGANOID_ID_COLUMN_KEY = "component_and_cluster_and_lasso"

def generate_manifest(patient_ids, run_names):
    """Generate CSV with all patient-organoid combinations."""
    dataset = NorkinOrganoidDataset(standardize_scale=False, scale=True, fill=True, use_cached_adata=True, use_cached_masks=True, organoid_id_column_key=ORGANOID_ID_COLUMN_KEY)
    
    manifest_data = []
    
    for sample_id, run_name in zip(sample_ids, run_names):
        print(f"Processing sample: {sample_id} (run: {run_name})")
        
        try:
            joined_df = dataset.get_organoid_df_by_id(run_name=run_name, sample_id=sample_id)
            organoid_ids = joined_df[dataset.organoid_id_column_key].unique().tolist()
            
            for organoid_id in organoid_ids:
                manifest_data.append({
                    'sample_id': sample_id,
                    'organoid_id': organoid_id,
                    'status': 'pending',
                    'run_name': run_name
                })
                
            print(f"  Found {len(organoid_ids)} organoids for sample {sample_id}")
            
        except Exception as e:
            import pdb; pdb.set_trace()
            print(f"  Error processing sample {sample_id}: {e}")
    
    # Create DataFrame and save
    manifest_df = pd.DataFrame(manifest_data)
    manifest_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Manifest saved to: {OUTPUT_CSV}")
    print(f"Total organoids: {len(manifest_df)}")
    
    return manifest_df

if __name__ == "__main__":
    # Expecting all samples first, then all runs
    if len(sys.argv) > 1:
        all_args = sys.argv[1:]
        mid_point = len(all_args) // 2
        sample_ids = all_args[:mid_point]  # First half are samples
        run_names = all_args[mid_point:]    # Second half are runs
    else:
        raise ValueError("No arguments provided - should be called with patient1 patient2 ... run1 run2 ...")
    
    if len(sample_ids) != len(run_names):
        raise ValueError(f"Number of samples ({len(sample_ids)}) does not match number of runs ({len(run_names)})")
        
    print(f"Generating manifest for {len(sample_ids)} samples")
    print(f"Samples: {sample_ids}")
    print(f"Runs: {run_names}")
    manifest = generate_manifest(sample_ids, run_names)