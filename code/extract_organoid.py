#!/usr/bin/env python3
import sys
import os
import glob
import argparse
import tifffile
import cv2
import numpy as np
import pandas as pd

sys.path.append("/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1")
from norkin_organoid.code.get_embeddings import NorkinOrganoidDataset

SCALE_FACTOR = 1 / 0.2125
ALIGNMENTS_ROOT_PATH = "/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/alignments/{}_qupath_alignment_files"
OUTPUT_DIR = "/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/organoids_h&e/images/"
PREVIEW_DIR = "/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/organoids_h&e/image_previews/"
FEATURES_DIR = "/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/organoids_h&e/features/"
MODEL_TYPE = "plip"  # Change to "resnet50" or other supported models

def get_microscopy(patient_id):
    base_dir = "/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/ome_tiff_pyr"
    matching_files = glob.glob(f"{base_dir}/r**/{patient_id}.ome.tiff")
    if len(matching_files) != 1:
        raise ValueError(f"Expected 1 OME-TIFF for patient '{patient_id}', found {len(matching_files)}")
    return tifffile.imread(matching_files[0])

def get_transform_matrix(patient_id):
    alignment_path = os.path.join(ALIGNMENTS_ROOT_PATH.format(patient_id), "matrix.csv")
    return pd.read_csv(alignment_path, header=None).to_numpy()

def extract_reoriented_optimized(large_input_image, bbox, transform_matrix, output_size=None):
    min_x, min_y, max_x, max_y = bbox * SCALE_FACTOR
    
    corners_original = np.array([[min_x, min_y, 1], [max_x, min_y, 1], [max_x, max_y, 1], [min_x, max_y, 1]]).T
    corners_transformed = (transform_matrix @ corners_original).T
    corners_transformed = corners_transformed[:, :2] / corners_transformed[:, 2:]
    
    x_min = max(0, int(np.floor(np.min(corners_transformed[:, 0]))))
    y_min = max(0, int(np.floor(np.min(corners_transformed[:, 1]))))
    x_max = min(large_input_image.shape[1], int(np.ceil(np.max(corners_transformed[:, 0]))))
    y_max = min(large_input_image.shape[0], int(np.ceil(np.max(corners_transformed[:, 1]))))

    if output_size is None: output_size = (x_max - x_min, y_max - y_min)

    roi = large_input_image[y_min:y_max, x_min:x_max]
    if roi.size == 0: return np.zeros((output_size[1], output_size[0], large_input_image.shape[2]), dtype=large_input_image.dtype)
    
    map_x = np.zeros((output_size[1], output_size[0]), dtype=np.float32)
    map_y = np.zeros((output_size[1], output_size[0]), dtype=np.float32)
    
    for y in range(output_size[1]):
        for x in range(output_size[0]):
            x_orig = min_x + (x / output_size[0]) * (max_x - min_x)
            y_orig = min_y + (y / output_size[1]) * (max_y - min_y)
            point_orig = np.array([x_orig, y_orig, 1])
            point_img = transform_matrix @ point_orig
            point_img = point_img[:2] / point_img[2]
            map_x[y, x] = point_img[0] - x_min
            map_y[y, x] = point_img[1] - y_min
    
    reoriented = cv2.remap(roi, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return reoriented

def write_pyramidal_ome_tiff(array, filename):
    with tifffile.TiffWriter(filename, bigtiff=True) as tif:
        tif.write(array, photometric='rgb' if array.shape[2] in [3, 4] else 'minisblack',
                 metadata={'axes': 'YXC'}, subfiletype=1, tile=(256, 256))

def save_png_preview(result, organoid_id):
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    
    if result.dtype != np.uint8:
        preview = np.clip(result, 0, 255).astype(np.uint8) if result.max() > 1.0 else (np.clip(result, 0, 1) * 255).astype(np.uint8)
    else:
        preview = result
    
    h, w = preview.shape[:2]
    max_dim = 1000
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        preview = cv2.resize(preview, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    
    preview_path = os.path.join(PREVIEW_DIR, f"{organoid_id}.png")
    cv2.imwrite(preview_path, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    return preview_path

def extract_lazyslide_features(organoid_id):
    try:
        import lazyslide as zs
        from wsidata import open_wsi
        
        os.makedirs(FEATURES_DIR, exist_ok=True)
        organoid_path = os.path.join(OUTPUT_DIR, f"{organoid_id}.ome.tiff")
        
        if not os.path.exists(organoid_path):
            print(f"Warning: Organoid file not found: {organoid_path}")
            return None
        
        wsi = open_wsi(organoid_path)
        wsi.set_mpp(8.625e-2)
        
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, 128)
        zs.tl.feature_extraction(wsi, MODEL_TYPE, amp=True)
        
        adata = wsi.fetch.features_anndata(MODEL_TYPE)
        features_path = os.path.join(FEATURES_DIR, f"{organoid_id}_features.h5ad")
        adata.write(features_path)
        
        print(f"Features saved: {features_path} (shape: {adata.X.shape})")
        return adata
        
    except ImportError as e:
        print(f"Warning: Could not import lazyslide: {e}")
        return None
    except Exception as e:
        print(f"Warning: Feature extraction failed: {e}")
        return None

def main_from_args(patient_id, organoid_id):
    print(f"Processing {organoid_id} from {patient_id}")
    
    dataset = NorkinOrganoidDataset(standardize_scale=False, scale=True, fill=True)
    microscopy = get_microscopy(patient_id)
    transform_matrix = get_transform_matrix(patient_id)
    
    organoid_bbox = np.array(dataset.organoid_square_bboxes[organoid_id])
    result = extract_reoriented_optimized(np.moveaxis(microscopy, 0, 2), organoid_bbox, np.linalg.inv(transform_matrix))
    
    output_path = os.path.join(OUTPUT_DIR, f"{organoid_id}.ome.tiff")
    write_pyramidal_ome_tiff(result, output_path)
    preview_path = save_png_preview(result, organoid_id)
    adata = extract_lazyslide_features(organoid_id)
    
    return output_path, preview_path, adata

def main_from_csv(manifest_csv, index):
    manifest_df = pd.read_csv(manifest_csv)
    if index >= len(manifest_df): raise ValueError(f"Index {index} out of range")
    row = manifest_df.iloc[index]
    main_from_args(row['patient_id'], row['organoid_id'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('patient_id', nargs='?')
    parser.add_argument('organoid_id', nargs='?')
    parser.add_argument('--from-csv')
    parser.add_argument('--index', type=int)
    
    args = parser.parse_args()
    
    if args.from_csv and args.index is not None:
        main_from_csv(args.from_csv, args.index)
    elif args.patient_id and args.organoid_id:
        main_from_args(args.patient_id, args.organoid_id)
    else:
        parser.print_help()
        sys.exit(1)