#!/usr/bin/env python3
import sys
import os
import glob
import argparse
from matplotlib import pyplot as plt
import tifffile
import cv2
import numpy as np
import pandas as pd
import spatialdata as sd
import geopandas as gpd

sys.path.append("/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1")
from norkin_organoid.workflow.scripts.xenium.morphology_code.get_embeddings import NorkinOrganoidDataset

SCALE_FACTOR = 1 / 0.2125
ALIGNMENTS_ROOT_PATH = "/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/alignments/"
OUTPUT_DIR = "/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/organoids_h&e/images/"
PREVIEW_DIR = "/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/organoids_h&e/image_previews/"
FEATURES_DIR = "/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/organoids_h&e/features_sdata/"

def get_microscopy(run_name, sample_id):
    base_dir = "/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/ome_tiff_pyr"
    glob_pth = f"{base_dir}/{run_name}/{sample_id}.ome.tiff"
    matching_files = glob.glob(glob_pth)
    if len(matching_files) != 1:
        raise ValueError(f"Expected 1 OME-TIFF for sample '{sample_id}' from pth {glob_pth}, found {len(matching_files)}: {matching_files}")
    return tifffile.imread(matching_files[0])

def get_alignment_path(run_name, sample_id):
    alignment_path = os.path.join(ALIGNMENTS_ROOT_PATH, run_name, f"{sample_id}_qupath_alignment_files", "matrix.csv")
    if not os.path.exists(alignment_path):
        raise FileNotFoundError(f"Alignment matrix not found at {alignment_path}")
    return alignment_path

def get_transform_matrix(run_name, sample_id):
    alignment_path = get_alignment_path(run_name, sample_id)
    return pd.read_csv(alignment_path, header=None).to_numpy()

def get_geodf_affine_transform_matrix(matrix):
    return [
        matrix[0][0],  # a
        matrix[0][1],  # b
        matrix[1][0],  # d
        matrix[1][1],  # e
        matrix[0][2],  # xoff
        matrix[1][2],  # yoff
    ]

def extract_reoriented_optimized(microscopy, joined_df, transform_matrix, organoid_id, output_size=None, organoid_id_column_key="component_and_cluster_labels"):
    joined_df = joined_df.copy()
    joined_df_organoid = joined_df[joined_df[organoid_id_column_key] == organoid_id]
    joined_df_organoid['geometry'] = joined_df_organoid['geometry'].scale(xfact=SCALE_FACTOR, yfact=SCALE_FACTOR, origin=(0,0))

    organoid_to_histopath_matrix_geodf = [
        transform_matrix[0][0],
        transform_matrix[0][1],
        transform_matrix[1][0],
        transform_matrix[1][1],
        transform_matrix[0][2],
        transform_matrix[1][2],
    ]

    joined_df_organoid['geometry'] = joined_df_organoid['geometry'].affine_transform(organoid_to_histopath_matrix_geodf)
    joined_df_organoid['geometry'] = joined_df_organoid['geometry'].buffer(distance=100)
    
    histo_bounds = joined_df_organoid.total_bounds
    minx, miny, maxx, maxy = [int(coord) for coord in histo_bounds]
    microscopy_cropped = microscopy[:, miny:maxy, minx:maxx]

    return (minx, miny, maxx, maxy), joined_df_organoid, microscopy_cropped

def write_pyramidal_ome_tiff(array, filename):
    with tifffile.TiffWriter(filename, bigtiff=True) as tif:
        tif.write(array, photometric='rgb' if array.shape[2] in [3, 4] else 'minisblack',
                 metadata={'axes': 'YXC'}, subfiletype=1, tile=(256, 256))

def save_png_preview(result, sample_id, organoid_id, root_dir=PREVIEW_DIR):
    os.makedirs(root_dir, exist_ok=True)
    
    if result.dtype != np.uint8:
        preview = np.clip(result, 0, 255).astype(np.uint8) if result.max() > 1.0 else (np.clip(result, 0, 1) * 255).astype(np.uint8)
    else:
        preview = result
    
    h, w = preview.shape[:2]
    max_dim = 1000
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        preview = cv2.resize(preview, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    
    preview_path = os.path.join(root_dir, f"{sample_id}_{organoid_id}.png")
    cv2.imwrite(preview_path, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    return preview_path

def wsi_is_blank(wsi, tiff_path):
    def get_dims(output_path): 
        import tifffile
        return tifffile.imread(output_path).shape

    width, height, _ = get_dims(tiff_path)
    total_area = width * height
    tissue_fraction = wsi.shapes['tissues'].area[0] / total_area
    return bool(tissue_fraction > 0.99)


def extract_lazyslide_features(
    sample_id,
    organoid_id, 
    joined_df=None, 
    organoid_bbox=None, 
    transform_matrix=None, 
    model_type="plip", 
    save_pth=FEATURES_DIR, 
    tiff_pth=OUTPUT_DIR, 
    he_min_x=None,
    he_min_y=None,
    tissue_selection_strategy="default"):
    """
    tissue selection strategies
    1) "default": use lazyslide tissue detection on H&E image and use all of the secondary tissues
    2) "segmentation_iou": use the segmentation mask with boundary in conjunction with an IOU requirement for the tissue
    3) "largest": use the largest tissue in the segmentation mask.
    """

    import lazyslide as zs
    from wsidata import open_wsi 
    
    os.makedirs(save_pth, exist_ok=True)
    organoid_path = os.path.join(tiff_pth, f"{sample_id}_{organoid_id}.ome.tiff")
    
    if not os.path.exists(organoid_path):
        raise FileNotFoundError(f"Warning: Organoid file not found: {organoid_path}")
    
    wsi = open_wsi(organoid_path)
    wsi.set_mpp(8.625e-2)
    
    zs.pp.find_tissues(wsi)
    if wsi_is_blank(wsi, organoid_path):
        raise Exception("Organoid is blank.")
        
    try: 
        tile_coords_hne_relative, _ = zs.pp.tile_tissues(wsi,
                        256, 
                        overlap=2.0/3, 
                        mpp=0.50,
                        edge=True, 
                        background_filter=True,
                        background_filter_mode="exact",
                        background_fraction=1.0,
                        return_tiles=True
        )

        os.makedirs("/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/organoids_h&e/tiles", exist_ok=True)
        zs.pl.tiles(wsi, 
                tissue_id="all", 
                linewidth=0.5)
        plt.savefig(f"/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/organoids_h&e/tiles/{sample_id}_{organoid_id}.png", dpi=300)

        zs.tl.feature_extraction(wsi, model_type, amp=True)

        tile_coords_hne_absolute = tile_coords_hne_relative.copy()
        tile_coords_hne_absolute['geometry'] = tile_coords_hne_absolute['geometry'].translate(xoff=he_min_x, yoff=he_min_y)
        tile_coords_xenium_absolute = tile_coords_hne_absolute.copy()
        histopath_to_orgnanoid_matrix_geodf = get_geodf_affine_transform_matrix(transform_matrix)
        tile_coords_xenium_absolute['geometry'] = tile_coords_xenium_absolute['geometry'].affine_transform(histopath_to_orgnanoid_matrix_geodf)
        tile_coords_xenium_absolute['geometry'] = tile_coords_xenium_absolute['geometry'].scale(1/SCALE_FACTOR, 1/SCALE_FACTOR, origin=(0,0))
        
        cells_in_tile = gpd.sjoin(joined_df, tile_coords_xenium_absolute, how='inner', predicate='intersects')
        cells_in_tile = sd.models.ShapesModel.parse(cells_in_tile)
        # Convert geometry to WKT string format
        adata = wsi.fetch.features_anndata(model_type)

        sdata = sd.SpatialData(
            shapes={
                "tile_coords_hne_absolute": tile_coords_hne_absolute, 
                "tile_coords_hne_relative": tile_coords_hne_relative, 
                "tile_coords_xenium_absolute": tile_coords_xenium_absolute,
                "cells_in_tile": cells_in_tile,
            },
            tables={"features_adata": adata},
        )

        features_path = os.path.join(FEATURES_DIR, f"{sample_id}_{organoid_id}_features.zarr")
        sdata.write(features_path, overwrite=True)
        
        print(f"Features saved: {features_path} (shape: {adata.X.shape})")

        return sdata
    except Exception as e: 
        raise Exception("No tiles of sufficient background were identified.")

def transform_tile_coords_to_he_space(tile_coords, organoid_bbox, transform_matrix):
    """
    Transform tile coordinates from organoid space back to original H&E whole slide space
    """
    # Tile coords are in the extracted organoid image space
    # We need to transform them back through the inverse process
    
    # Get the center of each tile in organoid space
    tile_centers_x = tile_coords[:, 0] + tile_coords[:, 2] / 2  # x + width/2
    tile_centers_y = tile_coords[:, 1] + tile_coords[:, 3] / 2  # y + height/2
    
    # Scale from organoid image space back to original microscopy coordinates
    # (reverse the SCALE_FACTOR applied during extraction)
    scaled_centers_x = tile_centers_x / SCALE_FACTOR
    scaled_centers_y = tile_centers_y / SCALE_FACTOR
    
    # Apply the bbox offset to get coordinates relative to the organoid bbox
    bbox_coords_x = organoid_bbox[0] + scaled_centers_x
    bbox_coords_y = organoid_bbox[1] + scaled_centers_y
    
    # Transform through the alignment matrix to get H&E whole slide coordinates
    he_coords = []
    for x, y in zip(bbox_coords_x, bbox_coords_y):
        point_orig = np.array([x, y, 1])
        point_he = transform_matrix @ point_orig
        point_he = point_he[:2] / point_he[2]
        he_coords.append(point_he)
    
    return np.array(he_coords)

def pad_image_evenly(image, target_dim, color=(0, 0, 0)):
    h, w, d = image.shape
    
    if h >= target_dim and w >= target_dim:
        return image
    
    pad_h = max(0, target_dim - h)
    pad_w = max(0, target_dim - w)
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded_image = cv2.copyMakeBorder(
        image,
        top=pad_top,
        bottom=pad_bottom,
        left=pad_left,
        right=pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=color
    )
    
    return padded_image

def main_from_args(sample_id, organoid_id, run_name, model_type):
    print(f"Processing {organoid_id} from {sample_id} (run: {run_name})")
    
    # Define error directory for failure tracking
    ERROR_DIR = "/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/err"  # Replace with your actual fixed folder path
    
    try:
        organoid_id_column_key = 'component_and_cluster_and_lasso'
        dataset = NorkinOrganoidDataset(
            standardize_scale=False, 
            scale=True, 
            fill=True,
            organoid_id_column_key=organoid_id_column_key, 
            use_cached_adata=True,
            use_cached_masks=True,
        )

        microscopy = get_microscopy(run_name, sample_id)
        transform_matrix = get_transform_matrix(run_name, sample_id)

        joined_df = dataset.get_organoid_df_by_id(run_name=run_name, sample_id=sample_id)
        histopath_bbox, transformed_df, result = extract_reoriented_optimized(microscopy, joined_df, np.linalg.inv(transform_matrix), organoid_id=organoid_id, organoid_id_column_key=organoid_id_column_key)
        result = pad_image_evenly(np.moveaxis(result, 0, 2), target_dim=1500, color=(210, 207, 209))

        output_path = os.path.join(OUTPUT_DIR, f"{sample_id}_{organoid_id}.ome.tiff")
        write_pyramidal_ome_tiff(result, output_path)
        preview_path = save_png_preview(result, sample_id, organoid_id)
    
    except Exception as e:
        # Create error directory if it doesn't exist
        os.makedirs(ERROR_DIR, exist_ok=True)
        
        # Create file with sample_id and organoid_id in the name, containing just the exception
        error_filename = f"error_{sample_id}_{organoid_id}.txt"
        error_filepath = os.path.join(ERROR_DIR, error_filename)
        
        # Write just the exception to file
        with open(error_filepath, 'w') as f:
            f.write(str(e))
        
        print(f"Error processing {organoid_id} from {sample_id}: {str(e)}")
        print(f"Error details written to: {error_filepath}")
        
        # Re-raise the exception to maintain original error behavior
        raise

    # Pass the additional parameters to extract_lazyslide_features
    sdata = extract_lazyslide_features(sample_id, organoid_id, joined_df=joined_df, transform_matrix=transform_matrix, model_type=model_type, he_min_x=histopath_bbox[0], he_min_y=histopath_bbox[1])
    
    return output_path, preview_path, sdata

def main_from_csv(manifest_csv, index, model_type):
    manifest_df = pd.read_csv(manifest_csv)
    if index >= len(manifest_df): raise ValueError(f"Index {index} out of range")
    row = manifest_df.iloc[index]
    main_from_args(row['sample_id'], row['organoid_id'], row['run_name'], model_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_id', nargs='?')
    parser.add_argument('organoid_id', nargs='?')
    parser.add_argument('run_name', nargs='?')
    parser.add_argument('--from-csv')
    parser.add_argument('--index', type=int)
    parser.add_argument('--model-type', type=str, default="plip")
    
    args = parser.parse_args()
    
    if args.from_csv and args.index is not None:
        main_from_csv(args.from_csv, args.index, args.model_type)
    elif args.sample_id and args.organoid_id and args.run_name:
        main_from_args(args.sample_id, args.organoid_id, args.run_name, args.model_type)
    else:
        parser.print_help()
        sys.exit(1)