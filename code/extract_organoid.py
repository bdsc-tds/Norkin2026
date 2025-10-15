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

def save_png_preview(result, organoid_id, root_dir=PREVIEW_DIR):
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
    
    preview_path = os.path.join(root_dir, f"{organoid_id}.png")
    cv2.imwrite(preview_path, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    return preview_path

def extract_lazyslide_features(organoid_id, organoid_bbox=None, transform_matrix=None, model_type="plip", save_pth=FEATURES_DIR, tiff_pth=OUTPUT_DIR):
    import lazyslide as zs
    from wsidata import open_wsi 
    
    os.makedirs(save_pth, exist_ok=True)
    organoid_path = os.path.join(tiff_pth, f"{organoid_id}.ome.tiff")
    
    if not os.path.exists(organoid_path):
        raise FileNotFoundError(f"Warning: Organoid file not found: {organoid_path}")
    
    wsi = open_wsi(organoid_path)
    wsi.set_mpp(8.625e-2)
    
    zs.pp.find_tissues(wsi)
    zs.pp.tile_tissues(wsi,
                       256, 
                       overlap=2.0/3, 
                       mpp=0.50,
                       edge=False, 
                       background_filter=True,
                       background_fraction=0.5,
    )
    zs.tl.feature_extraction(wsi, model_type, amp=True)
    
    adata = wsi.fetch.features_anndata(model_type)
    
    # # Add H&E coordinates to the AnnData object
    # if organoid_bbox is not None and transform_matrix is not None:
    #     # Get tile coordinates from lazyslide
    #     tile_coords = wsi.tiles.coords
        
    #     # Transform coordinates back to original H&E space
    #     he_coords = transform_tile_coords_to_he_space(tile_coords, organoid_bbox, transform_matrix)
        
    #     # Add coordinates to AnnData obs
    #     adata.obs['he_x'] = he_coords[:, 0]
    #     adata.obs['he_y'] = he_coords[:, 1]
        
    #     # Also store the original organoid bbox and transform info
    #     adata.uns['organoid_bbox'] = organoid_bbox
    #     adata.uns['transform_matrix'] = transform_matrix
    #     adata.uns['organoid_id'] = organoid_id
    
    features_path = os.path.join(save_pth, f"{organoid_id}_features.h5ad")
    adata.write(features_path)
    
    print(f"Features saved: {features_path} (shape: {adata.X.shape})")
    # print(f"H&E coordinates added: {he_coords.shape}")
    return adata

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

def main_from_args(patient_id, organoid_id, model_type):
    print(f"Processing {organoid_id} from {patient_id}")
    
    dataset = NorkinOrganoidDataset(standardize_scale=False, scale=True, fill=True)
    microscopy = get_microscopy(patient_id)
    transform_matrix = get_transform_matrix(patient_id)
    
    organoid_bbox = np.array(dataset.organoid_square_bboxes[organoid_id])
    result = extract_reoriented_optimized(np.moveaxis(microscopy, 0, 2), organoid_bbox, np.linalg.inv(transform_matrix))
    result = pad_image_evenly(result, target_dim=1500, color=(210, 207, 209))
    
    output_path = os.path.join(OUTPUT_DIR, f"{organoid_id}.ome.tiff")
    write_pyramidal_ome_tiff(result, output_path)
    preview_path = save_png_preview(result, organoid_id)
    
    # Pass the additional parameters to extract_lazyslide_features
    adata = extract_lazyslide_features(organoid_id, organoid_bbox, transform_matrix, model_type)
    
    return output_path, preview_path, adata

def main_from_csv(manifest_csv, index, model_type):
    manifest_df = pd.read_csv(manifest_csv)
    if index >= len(manifest_df): raise ValueError(f"Index {index} out of range")
    row = manifest_df.iloc[index]
    main_from_args(row['patient_id'], row['organoid_id'], model_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('patient_id', nargs='?')
    parser.add_argument('organoid_id', nargs='?')
    parser.add_argument('--from-csv')
    parser.add_argument('--index', type=int)
    parser.add_argument('--model-type', type=str, default="plip")
    
    args = parser.parse_args()
    
    if args.from_csv and args.index is not None:
        main_from_csv(args.from_csv, args.index, args.model_type)
    elif args.patient_id and args.organoid_id:
        main_from_args(args.patient_id, args.organoid_id, args.model_type)
    else:
        parser.print_help()
        sys.exit(1)