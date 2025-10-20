import os
import argparse
import tifffile
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter

PATHS = {
    "run_1": "/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/Xenium_PDO_run_2_1_HE_16-07-2025.czi", 
    "run_2": "/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/Xenium_PDO_run_2_2_HE-16-07-2025.czi"
}

CORRESPONDENCES = {
    "run_1": [
        "1HVQ",
        "1CNN", 
        "077I", 
        "1GAA",
        "1J25",
        "131N",
        "OWJ3",
        "14PT",
    ],
    "run_2": [
        "169V",
        "1BI7",
        "1CI5",
        "1FMS",
        "12NM",
        "OLR9",
        "1GVB",
        "1GNS"
    ]
}

# import os
# import numpy as np
# from czifile import CziFile
# import tifffile

# def process_image(run, index):
#     """Process a specific image from a run based on index using czifile"""
#     if run not in PATHS:
#         raise ValueError(f"Invalid run: {run}. Available runs: {list(PATHS.keys())}")
    
#     if index >= len(CORRESPONDENCES[run]):
#         raise ValueError(f"Index {index} out of range for run {run}. Max index: {len(CORRESPONDENCES[run]) - 1}")
    
#     # Get the patient ID for the given index
#     patient_id = CORRESPONDENCES[run][index]
    
#     # Load the CZI file
#     with CziFile(PATHS[run]) as czi:
#         import pdb; pdb.set_trace()
#         # Get all sc
#         # enes (subblocks)
#         # scenes = czi.subblock_directory
#         # print("scenes", scenes)
#         # print("scenes", [scene.__dir__() for scene in scenes])
        
#         if index >= len(scenes):
#             raise ValueError(f"Index {index} out of range for CZI file. File contains {len(scenes)} scenes.")
        
#         # Get the specific scene (subblock) data
#         scene_data = czi.subblock_directory[index]
#         print(f"Processing scene {index} with data {scene_data} for Patient ID: {patient_id}")

#         image_data = czi.imread(subblock=index)
        
#         # Remove singleton dimensions and get image data in TCZYX order
#         # CZI data typically has dimensions: [T, C, Z, Y, X] or similar
#         image_data = np.squeeze(image_data)
        
#         print(f"Image for Patient ID: {patient_id} has shape {image_data.shape} and dtype {image_data.dtype}")
        
#         # Create output directory
#         os.makedirs(f"/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/ome_tiff/{run}", exist_ok=True)
        
#         # Save as OME-TIFF
#         ome_tiff_path = f"/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/ome_tiff/{run}/{patient_id}.ome.tiff"
        
#         # Save with OME-TIFF metadata
#         tifffile.imwrite(
#             ome_tiff_path,
#             image_data,
#             ome=True,
#             metadata={
#                 'Pixels': {
#                     'PhysicalSizeX': scene_data.x_size if hasattr(scene_data, 'x_size') else None,
#                     'PhysicalSizeY': scene_data.y_size if hasattr(scene_data, 'y_size') else None,
#                     'PhysicalSizeZ': scene_data.z_size if hasattr(scene_data, 'z_size') else None,
#                 }
#             }
#         )
        
#         print(f"Saved OME-TIFF to {ome_tiff_path}")
    
#     return ome_tiff_path

# def process_image(run, index):
#     """Process a specific image from a run based on index"""
#     if run not in PATHS:
#         raise ValueError(f"Invalid run: {run}. Available runs: {list(PATHS.keys())}")
    
#     if index >= len(CORRESPONDENCES[run]):
#         raise ValueError(f"Index {index} out of range for run {run}. Max index: {len(CORRESPONDENCES[run]) - 1}")
    
#     # Load the image
#     img = AICSImage(PATHS[run])
    
#     # Get the scene ID for the given index
#     scene_id = img.scenes[index]
#     patient_id = CORRESPONDENCES[run][index]
    
#     # Set the scene and get image data
#     img.set_scene(scene_id)
#     img_data = img.get_image_data("TCZYX")
#     print(f"Image {scene_id} (Patient ID: {patient_id}) has shape {img_data.shape}")
    
#     # Create output directory
#     os.makedirs(f"/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/ome_tiff/{run}", exist_ok=True)
    
#     # Save as OME-TIFF
#     ome_tiff_path = f"/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/ome_tiff/{run}/{patient_id}.ome.tiff"
#     img.save(ome_tiff_path)
#     print(f"Saved OME-TIFF to {ome_tiff_path}")
    
#     return ome_tiff_path

def process_image(run, index):
    """Process a specific image from a run based on index"""
    if run not in PATHS:
        raise ValueError(f"Invalid run: {run}. Available runs: {list(PATHS.keys())}")
    
    if index >= len(CORRESPONDENCES[run]):
        raise ValueError(f"Index {index} out of range for run {run}. Max index: {len(CORRESPONDENCES[run]) - 1}")
    
    # Load the image
    img = AICSImage(PATHS[run])
    
    # Get the scene ID for the given index
    scene_id = img.scenes[index]
    patient_id = CORRESPONDENCES[run][index]
    
    # Set the scene and get image data
    img.set_scene(scene_id)
    img_shape = img.shape
    print(f"img.shape: {img_shape}")
    img_data = img.get_image_data()
    img_data_shape = img_data.shape
    print(f"Image {scene_id} (Patient ID: {patient_id}) has img_data.shape {img_data.shape}")
    if 3 not in img_data.shape:
        raise ValueError(f"Image {scene_id} (Patient ID: {patient_id}) does not have 3 channels, has shape {img_data.shape}")

    # Create output directory
    os.makedirs(f"/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/ome_tiff/{run}", exist_ok=True)
    
    # Save as OME-TIFF
    
    # if method == 1:
    #     ome_tiff_path = f"/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/ome_tiff/{run}/method{method}_{patient_id}.ome.tiff"
    #     img.save(ome_tiff_path)
    #     print(f"Saved OME-TIFF method 1 to {ome_tiff_path}")
    # if method == 2:
    ome_tiff_path = f"/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/ome_tiff/{run}/{patient_id}.ome.tiff"
    tifffile.imwrite(ome_tiff_path, img_data, ome=True)
    print(f"Saved OME-TIFF method 2 to {ome_tiff_path}")
    # if method == 3:
    #     ome_tiff_path = f"/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/ome_tiff/{run}/method{method}_{patient_id}.ome.tiff"
    #     img.save(ome_tiff_path, select_scenes=[scene_id])
    #     print(f"Saved OME-TIFF method 3 to {ome_tiff_path}")

    im = tifffile.imread(ome_tiff_path)
    print(f"tiff shape: {im.shape}; should be {img_data_shape}")

    return ome_tiff_path

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Process CZI images and convert to OME-TIFF format")
    parser.add_argument("--run", required=True, choices=["run_1", "run_2"], 
                       help="Which run to process (run_1 or run_2)")
    parser.add_argument("--index", type=int, required=True, 
                       help="Index of the image to process (0-based)")
    # parser.add_argument("--method", type=int, required=True, default=1, choices=[1, 2, 3],
    #                    help="method to save OME-TIFF (1 or 2)")
    
    args = parser.parse_args()
    
    try:
        print(f"Starting process image {args.index} from {args.run}")
        result_path = process_image(args.run, args.index)
        print(f"Successfully processed image {args.index} from {args.run}")
    except Exception as e:
        print(f"Error processing image: {e}")
        exit(1)