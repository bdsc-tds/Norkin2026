#!/usr/bin/env python3
"""
CZII to OME-TIFF Converter for Spatial Transcriptomics Regions

This script converts specific regions from a CZI whole-slide image to OME-TIFF format
based on coordinate mappings from Xenium spatial transcriptomics data.

The script performs coordinate transformations between Xenium coordinate space 
and CZI image space to extract specific regions of interest.

Dependencies between CZI file and DataFrame:
------------------------------------------------------------
1. CZI FILE REQUIREMENTS:
   - Whole-slide H&E image in CZI format
   - Should contain the entire tissue region captured in the DataFrame
   - Expected to be a single scene/mosaic image

2. DATAFRAME REQUIREMENTS:
   The CSV file must contain the following columns with Xenium coordinates:
   - 'Name': Unique identifier for each region (e.g., "1CFV", "2CFV")
   - 'x_min', 'x_max', 'y_min', 'y_max': Bounding box coordinates in Xenium space
   
   Example DataFrame structure:
   Name    x_min    x_max    y_min    y_max
   1CFV    1000     2000     3000     4000
   2CFV    2500     3500     1500     2500

3. COORDINATE TRANSFORMATION:
   - Xenium Y coordinates map to CZI X coordinates (with flip)
   - Xenium X coordinates map to CZI Y coordinates
   - Coordinate system includes expansion and clipping to handle boundaries
   - The transformation accounts for the flipped coordinate systems between platforms

Usage examples:
------------------------------------------------------------
# Process a single region by row index
python czi_to_ome_single.py --row_index 0

# Process with custom file paths
python czi_to_ome_single.py --row_index 5 \
    --czi_path /path/to/image.czi \
    --region_csv /path/to/regions.csv \
    --output_dir /path/to/output

# Batch processing via SLURM (see accompanying bash script)
"""

import pandas as pd
import numpy as np
import argparse
from aicspylibczi import CziFile
import tifffile
import sys

def main():
    parser = argparse.ArgumentParser(
        description='Convert CZI to OME-TIFF for a specific region by row index',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--row_index', type=int, required=True, 
                       help='Row index (0-based) in the dataframe to process')
    parser.add_argument('--czi_path', type=str, 
                       default="/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/organoids_h&e/czi/amadurga-10-10-2025-003.czi",
                       help='Path to CZI whole-slide image file')
    parser.add_argument('--region_csv', type=str,
                       default="/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/norkin_organoid/data/xenium/metadata/Regions_coordinates_18samples.csv",
                       help='Path to CSV file containing region coordinates with columns: Name, x_min, x_max, y_min, y_max')
    parser.add_argument('--output_dir', type=str,
                       default="/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/ome_tiff/run_4_1",
                       help='Output directory for OME-TIFF files')
    
    args = parser.parse_args()
    
    # Load CZI and region data
    print(f"Loading CZI file: {args.czi_path}")
    czi = CziFile(args.czi_path)
    
    print(f"Loading region coordinates: {args.region_csv}")
    region_df = pd.read_csv(args.region_csv)
    
    # Validate required columns exist in DataFrame
    required_columns = ['Name', 'x_min', 'x_max', 'y_min', 'y_max']
    missing_columns = [col for col in required_columns if col not in region_df.columns]
    if missing_columns:
        print(f"Error: DataFrame missing required columns: {missing_columns}")
        print(f"Available columns: {list(region_df.columns)}")
        return 1
    
    # Get CZI bounding box for coordinate transformation
    czi_bbox = czi.get_scene_bounding_box()
    print(f"CZI image dimensions: {czi_bbox.w} x {czi_bbox.h} pixels")
    print(f"CZI bounding box: x={czi_bbox.x}, y={czi_bbox.y}, w={czi_bbox.w}, h={czi_bbox.h}")
    
    def add_hne_bboxes_to_region_df(czi_bbox, region_df):
        """
        Transform Xenium coordinates to CZI image coordinates.
        
        This function:
        1. Expands Xenium bounding boxes by fixed margins
        2. Clips coordinates to original Xenium bounds
        3. Calculates scaling factors between coordinate systems
        4. Applies coordinate transformation with axis flipping
        
        Parameters:
        -----------
        czi_bbox : object
            CziFile bounding box with x, y, w, h attributes
        region_df : pandas.DataFrame
            DataFrame with Xenium coordinates
            
        Returns:
        --------
        pandas.DataFrame
            Updated DataFrame with CZI coordinates added
        """
        
        # Expansion parameters for Xenium bounding boxes
        xenium_expansion_x = 1000
        xenium_expansion_y = 1000

        # Store original bounds for clipping
        old_xenium_ymin = region_df['y_min'].min()
        old_xenium_ymax = region_df['y_max'].max()
        old_xenium_xmin = region_df['x_min'].min()
        old_xenium_xmax = region_df['x_max'].max()

        print(f"Original Xenium bounds: X({old_xenium_xmin}, {old_xenium_xmax}), Y({old_xenium_ymin}, {old_xenium_ymax})")

        # Expand bounding boxes
        region_df['x_max'] += xenium_expansion_x
        region_df['x_min'] -= xenium_expansion_x
        region_df['y_max'] += xenium_expansion_y
        region_df['y_min'] -= xenium_expansion_y

        # Clip to original bounds to prevent invalid coordinates
        region_df['x_min'] = np.clip(region_df['x_min'], old_xenium_xmin, old_xenium_xmax)
        region_df['x_max'] = np.clip(region_df['x_max'], old_xenium_xmin, old_xenium_xmax)
        region_df['y_min'] = np.clip(region_df['y_min'], old_xenium_ymin, old_xenium_ymax)
        region_df['y_max'] = np.clip(region_df['y_max'], old_xenium_ymin, old_xenium_ymax)

        # Calculate overall Xenium bounds for scaling
        xenium_xmin = region_df['x_min'].min()
        xenium_xmax = region_df['x_max'].max()
        xenium_ymin = region_df['y_min'].min()
        xenium_ymax = region_df['y_max'].max()

        xenium_width = xenium_xmax - xenium_xmin
        xenium_height = xenium_ymax - xenium_ymin
        
        print(f"Expanded Xenium bounds: X({xenium_xmin}, {xenium_xmax}), Y({xenium_ymin}, {xenium_ymax})")
        print(f"Xenium dimensions: {xenium_width} x {xenium_height}")

        # Calculate scaling factors between coordinate systems
        # Note: Xenium X maps to CZI Y, Xenium Y maps to CZI X (with flip)
        xenium_x_to_czi_y_scale_factor = czi_bbox.h / xenium_width
        xenium_y_to_czi_x_scale_factor = czi_bbox.w / xenium_height
        xenium_x_to_czi_y_shift_factor = czi_bbox.y
        xenium_y_to_czi_x_shift_factor = czi_bbox.x

        print(f"Scaling factors - X->Y: {xenium_x_to_czi_y_scale_factor:.6f}, Y->X: {xenium_y_to_czi_x_scale_factor:.6f}")

        # Coordinate transformation functions
        xenium_x_to_czi_y = lambda xenium_x: xenium_x * xenium_x_to_czi_y_scale_factor + xenium_x_to_czi_y_shift_factor
        xenium_y_to_czi_x = lambda xenium_y: (old_xenium_ymax - xenium_y) * xenium_y_to_czi_x_scale_factor + xenium_y_to_czi_x_shift_factor

        # Apply transformations
        # Note: Xenium Y coordinates map to CZI X coordinates (with flip)
        region_df['czi_xmin'] = region_df['y_max'].apply(xenium_y_to_czi_x)
        region_df['czi_xmax'] = region_df['y_min'].apply(xenium_y_to_czi_x)
        region_df['czi_width'] = region_df['czi_xmax'] - region_df['czi_xmin']

        # Xenium X coordinates map to CZI Y coordinates
        region_df['czi_ymin'] = region_df['x_min'].apply(xenium_x_to_czi_y)
        region_df['czi_ymax'] = region_df['x_max'].apply(xenium_x_to_czi_y)
        region_df['czi_height'] = region_df['czi_ymax'] - region_df['czi_ymin']

        return region_df.copy()

    # Apply coordinate transformations
    print("Transforming coordinates from Xenium to CZI space...")
    updated_region_df = add_hne_bboxes_to_region_df(czi_bbox, region_df)
    
    # Validate row index
    if args.row_index < 0 or args.row_index >= len(updated_region_df):
        print(f"Error: Row index {args.row_index} is out of range. Dataframe has {len(updated_region_df)} rows (0-{len(updated_region_df)-1}).")
        print(f"Available regions: {list(updated_region_df['Name'])}")
        return 1
    
    # Get the specific row to process
    row = updated_region_df.iloc[args.row_index]
    region_name = row['Name']
    
    print(f"\nProcessing region: {region_name} (index: {args.row_index})")
    print(f"Xenium coordinates: X({row['x_min']}, {row['x_max']}), Y({row['y_min']}, {row['y_max']})")
    print(f"CZI coordinates: X({row['czi_xmin']:.1f}, {row['czi_xmax']:.1f}), Y({row['czi_ymin']:.1f}, {row['czi_ymax']:.1f})")
    print(f"CZI region size: {row['czi_width']:.1f} x {row['czi_height']:.1f} pixels")
    
    # Extract and save the image
    try:
        print("Reading mosaic from CZI file...")
        im = czi.read_mosaic(region=(
            int(row['czi_xmin']) + 1, 
            int(row['czi_ymin']) + 1,
            int(row['czi_width']) - 2, 
            int(row['czi_height']) - 2, 
        ), C=0, scale_factor=1.0)[0]

        output_path = f"{args.output_dir}/{region_name}.tiff"
        print(f"Saving OME-TIFF: {output_path}")
        print(f"Image shape: {im.shape}, dtype: {im.dtype}")
                
        tifffile.imwrite(output_path, im)
        print(f"Successfully saved original: {output_path}")

        print("Reading mosaic preview from CZI file...")
        im = czi.read_mosaic(region=(
            int(row['czi_xmin']) + 1, 
            int(row['czi_ymin']) + 1,
            int(row['czi_width']) - 2, 
            int(row['czi_height']) - 2, 
        ), C=0, scale_factor=0.2)[0]

        output_path = f"{args.output_dir}/{region_name}_preview.tiff"
        print(f"Saving OME-TIFF preview: {output_path}")
        print(f"Preview image shape: {im.shape}, dtype: {im.dtype}")
        
        tifffile.imwrite(output_path, im)
        print(f"Successfully saved preview: {output_path}")

        return 0
        
    except Exception as e:
        print(f"Error processing region {region_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())