import os 
import torch 
from PIL import Image
import geopandas as gpd
from IPython.display import display
from huggingface_hub import snapshot_download

from trident import OpenSlideWSI
from trident.segmentation_models import segmentation_model_factory
from trident.Converter import AnyToTiffConverter

converter = AnyToTiffConverter(
    job_dir="/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/TRIDENT/data/processed_tiff_files",  
    bigtiff=False,
)

converter.process_file(
    input_file="/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/TRIDENT/data/kho_xenium_pdo_run_2_1_he_16-07-2025-czi_2025-07-29_0752/Xenium_PDO_run_2_1_HE_16-07-2025.czi",
    mpp=8.625e-8,
    zoom=1,
)