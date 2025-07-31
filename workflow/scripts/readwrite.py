import dask

dask.config.set({"dataframe.query-planning": False})
import yaml
import pandas as pd
import os
import json
import scipy
import geopandas as gpd
import dask.dataframe as dd
import spatialdata
import spatialdata_io
import warnings
import anndata as ad
import scanpy as sc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
from types import MappingProxyType
from spatialdata.models import (
    # Image2DModel,
    # Labels2DModel,
    # Labels3DModel,
    ShapesModel,
    PointsModel,
)
from collections import defaultdict
from typing import Optional, List

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "../../config/config.yml")


def config(path=config_path):
    """
    Read the configuration file and return a dictionary of config values.

    Parameters
    ----------
    path : str
        The path to the configuration file. Defaults to the value of
        `config_path` if not provided.

    Returns
    -------
    cfg : dict
        A dictionary of configuration values. All values are strings and
        have been converted to absolute paths by prepending the value of
        `cfg["base_dir"]`.
    """
    with open(path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
        for k in cfg.keys():
            if "umap_" in k:
                continue
            cfg[k] = os.path.join(cfg["base_dir"], cfg[k])
    return cfg


def xenium_specs(path):
    path = Path(path)
    with open(path / "experiment.xenium") as f:
        specs = json.load(f)
    return specs


######### Xenium readers
def discover_xenium_paths(
    analysis_dir: Path,
    data_dir: Path,
    annotation_dir: Optional[Path] = None,
    correction_dir: Optional[Path] = None,
    normalisation: Optional[str] = None,
    reference: Optional[str] = None,
    method: Optional[str] = None,
    level: Optional[str] = None,
    correction_methods_filter: List[str] = ["raw", "split_fully_purified"],
    segmentations_filter: Optional[List[str]] = None,
    conditions_filter: Optional[List[str]] = None,
    panels_filter: Optional[List[str]] = None,
):
    """
    Discovers Xenium sample paths with optional, high-performance filtering.

    Args:
        analysis_dir: Path to the base directory for Seurat analysis.
        data_dir: Path to the base directory containing the raw Xenium data.
        segmentations_filter: Optional list of segmentation names to include.
        conditions_filter: Optional list of condition names to include.
        panels_filter: Optional list of panel names to include.

    Returns:
        A tuple of two standard dictionaries: (xenium_paths, xenium_annot_paths).
    """
    # Use sets for fast lookups (O(1)). This is much faster than list lookups.
    seg_set = set(segmentations_filter) if segmentations_filter else None
    cond_set = set(conditions_filter) if conditions_filter else None
    panel_set = set(panels_filter) if panels_filter else None

    xenium_paths = defaultdict(dict)
    xenium_annot_paths = defaultdict(dict)

    glob_pattern = "*/*/*/*/*"
    for sample_path in analysis_dir.glob(glob_pattern):
        if not sample_path.is_dir():
            continue

        k = sample_path.relative_to(analysis_dir).parts

        # --- High-performance filtering logic ---
        if (
            (seg_set and k[0] not in seg_set)
            or (cond_set and k[1] not in cond_set)
            or (panel_set and k[2] not in panel_set)
        ):
            continue  # Skip this path if it doesn't match the filters

        # If we reach here, the path has passed all filters. Proceed as before.
        name = "/".join(k)

        if "raw" in correction_methods_filter:
            if "proseg" in k[0]:
                raw_path = data_dir / f"{'/'.join(('proseg',) + k[1:])}/raw_results"
            else:
                raw_path = data_dir / f"{name}/normalised_results/outs"
            xenium_paths["raw"][k] = raw_path

            if annotation_dir is not None:
                annot_path = annotation_dir / (
                    f"{name}/{normalisation}/reference_based/{reference}/{method}/{level}/single_cell/labels.parquet"
                )
                xenium_annot_paths["raw"][k] = annot_path

        if "split_fully_purified" in correction_methods_filter:
            corrected_base = (
                f"{name}/{normalisation}/reference_based/{reference}/{method}/{level}/single_cell/split_fully_purified"
            )
            corrected_path = correction_dir / f"{corrected_base}/corrected_counts.h5"
            xenium_paths["split_fully_purified"][k] = corrected_path

    # Return standard dicts for better compatibility with downstream tools
    return dict(xenium_paths), dict(xenium_annot_paths)


def read_xenium_specs(xenium_specs_file):
    xenium_specs_file = Path(xenium_specs_file)
    with open(xenium_specs_file) as f:
        specs = json.load(f)
    return specs


def xenium_proseg(
    path,
    cells_boundaries=True,
    cells_boundaries_layers=True,
    nucleus_boundaries=False,
    cells_as_circles=False,
    cells_labels=False,
    nucleus_labels=False,
    transcripts=True,
    morphology_mip=False,
    morphology_focus=False,
    aligned_images=False,
    cells_table=True,
    n_jobs=1,
    imread_kwargs=MappingProxyType({}),
    image_models_kwargs=MappingProxyType({}),
    labels_models_kwargs=MappingProxyType({}),
    cells_metadata=True,
    xeniumranger_dir=None,
    xenium_specs=True,
    pandas_engine="pyarrow",
    verbose=False,
):
    """
    Reads a Xenium segmentation run and returns a `SpatialData` object.

    Parameters
    ----------
    path : str
        The directory path of the segmentation run.
    cells_boundaries : bool or str, optional
        The file path of the cell boundaries GeoJSON file.
    cells_boundaries_layers : bool or str, optional
        The file path of the cell boundaries layers GeoJSON file.
    nucleus_boundaries : bool or str, optional
        Whether to read nucleus boundaries. Not implemented.
    cells_as_circles : bool or str, optional
        Whether to read cells as circles.
    cells_labels : bool or str, optional
        Whether to read cells labels. Not implemented.
    nucleus_labels : bool or str, optional
        Whether to read nucleus labels. Not implemented.
    transcripts : bool or str, optional
        The file path of the transcripts CSV file.
    morphology_mip : bool, optional
        Whether to read morphology MIP images.
    morphology_focus : bool, optional
        Whether to read morphology focus images.
    aligned_images : bool, optional
        Whether to read aligned images.
    cells_table : bool or str, optional
        The file path of the cells table CSV file.
    n_jobs : int, optional
        The number of jobs to use. Not implemented.
    imread_kwargs : dict, optional
        Keyword arguments to pass to `imread`.
    image_models_kwargs : dict, optional
        Keyword arguments to pass to `ImageModel`.
    labels_models_kwargs : dict, optional
        Keyword arguments to pass to `LabelsModel`.
    cells_metadata : bool or str, optional
        The file path of the cells metadata CSV file.
    xeniumranger_dir : str, optional
        The directory path of the XeniumRanger run.
    xenium_specs : bool or str, optional
        The file path of the Xenium specs file.
    pandas_engine : str, optional
        The pandas engine to use when reading CSV files.

    Returns
    -------
    sdata : SpatialData
        The `SpatialData` object.
    """
    path = Path(path)

    # unsupported options compared to spatialdata_io.xenium
    if nucleus_boundaries:
        raise ValueError("reading nucleus_boundaries not implemented for proseg")
    if cells_labels:
        raise ValueError("reading cells_labels not implemented for proseg")
    if nucleus_labels:
        raise ValueError("reading nucleus_labels not implemented for proseg")
    if n_jobs > 1:
        raise ValueError("n_jobs>1 not supported")

    # default expected file paths
    def parse_arg(arg, default):
        if arg:
            return default
        elif isinstance(arg, str):
            return Path(arg)
        else:
            return arg

    cells_metadata = parse_arg(cells_metadata, path / "cell-metadata.csv.gz")
    cells_boundaries = parse_arg(cells_boundaries, path / "cell-polygons.geojson.gz")
    cells_boundaries_layers = parse_arg(cells_boundaries_layers, path / "cell-polygons-layers.geojson.gz")
    transcripts = parse_arg(transcripts, path / "transcript-metadata.csv.gz")
    cells_table = parse_arg(cells_table, path / "expected-counts.csv.gz")
    xeniumranger_dir = parse_arg(xeniumranger_dir, None)
    if xeniumranger_dir is not None or isinstance(xenium_specs, str):
        xenium_specs = parse_arg(xenium_specs, xeniumranger_dir / "experiment.xenium")

    ### images
    if morphology_mip or morphology_focus or aligned_images:
        if verbose:
            print("Reading images...")
        sdata_images = spatialdata_io.xenium(
            xeniumranger_dir,
            cells_table=False,
            cells_as_circles=False,
            cells_boundaries=False,
            nucleus_boundaries=False,
            cells_labels=False,
            nucleus_labels=False,
            transcripts=False,
            morphology_mip=morphology_mip,
            morphology_focus=morphology_focus,
            aligned_images=aligned_images,
            imread_kwargs=imread_kwargs,
            image_models_kwargs=image_models_kwargs,
            labels_models_kwargs=labels_models_kwargs,
        )

        images = sdata_images.images
    else:
        images = {}

    ### tables
    region = "cell_polygons"
    region_key = "region"
    instance_key = "cell_id"

    # flag columns not corresponding to genes
    if isinstance(cells_table, Path):
        if verbose:
            print("Reading cells table...")
        df_table = pd.read_csv(cells_table, engine=pandas_engine)

        control_columns = df_table.columns.str.contains("|".join(["BLANK_", "Codeword", "NegControl"]))

        table = ad.AnnData(
            df_table.iloc[:, ~control_columns],
            uns={
                "spatialdata_attrs": {
                    "region": region,
                    "region_key": region_key,
                    "instance_key": instance_key,
                }
            },
        )

        if isinstance(cells_metadata, Path):
            if verbose:
                print("Reading cells metadata...")

            df_cells_metadata = pd.read_csv(cells_metadata, engine=pandas_engine).rename(columns={"cell": "cell_id"})
            table.obs = pd.concat((df_cells_metadata, df_table.iloc[:, control_columns]), axis=1)
            table.obsm["spatial"] = table.obs[["centroid_x", "centroid_y"]].values
            table.obs = table.obs
        table.obs[region_key] = region

        # sparsify .X
        table.X = scipy.sparse.csr_matrix(table.X)
        tables = {"table": table}
    else:
        tables = {}

    ### labels
    # not implemented
    labels = {}

    ### points
    if isinstance(transcripts, Path):
        if verbose:
            print("Reading transcripts...")

        df_transcripts = dd.read_csv(transcripts, blocksize=None).rename(
            columns={"gene": "feature_name", "assignment": "cell_id"}
        )
        points = {"transcripts": PointsModel.parse(df_transcripts)}
    else:
        points = {}

    ### shapes
    shapes = {}

    # read specs
    if isinstance(xenium_specs, Path):
        if verbose:
            print("Reading specs...")
        specs = read_xenium_specs(xenium_specs)

        # get xenium pixel size
        scale = spatialdata.transformations.Scale(
            [1.0 / specs["pixel_size"], 1.0 / specs["pixel_size"]], axes=("x", "y")
        )
        transformations = {"global": scale}
    else:
        transformations = None
        if isinstance(cells_boundaries, Path) or isinstance(cells_boundaries_layers, Path):
            warnings.warn(
                """
                Couldn't load xenium specs file with pixel size. 
                Not applying scale transformations to shapes.
                Please specify xeniumranger_dir or xenium_specs
                """
            )

    # read cells boundaries
    if isinstance(cells_boundaries, Path):
        if verbose:
            print("Reading cells boundaries...")

        df_cells_boundaries = gpd.read_file("gzip://" + cells_boundaries.as_posix()).rename(columns={"cell": "cell_id"})
        shapes["cells_boundaries"] = ShapesModel.parse(df_cells_boundaries, transformations=transformations)

    # read cells boundaries layers
    if isinstance(cells_boundaries_layers, Path):
        if verbose:
            print("Reading cells boundaries layers...")

        df_cells_boundaries_layers = gpd.read_file("gzip://" + cells_boundaries_layers.as_posix()).rename(
            columns={"cell": "cell_id"}
        )
        shapes["cells_boundaries_layers"] = ShapesModel.parse(
            df_cells_boundaries_layers, transformations=transformations
        )

    # convert cells boundaries to circles
    if cells_as_circles:
        if verbose:
            print("Converting cells boundaries to circle...")

        shapes["cells_boundaries_circles"] = spatialdata.to_circles(shapes["cells_boundaries"])

    ### sdata
    sdata = spatialdata.SpatialData(images=images, labels=labels, points=points, shapes=shapes, tables=tables)

    return sdata


def read_xenium_sample(
    path,
    cells_as_circles=False,
    cells_boundaries=False,
    cells_boundaries_layers=False,
    nucleus_boundaries=False,
    cells_labels=False,
    nucleus_labels=False,
    transcripts=False,
    morphology_mip=False,
    morphology_focus=False,
    aligned_images=False,
    cells_table=True,
    anndata=False,
    xeniumranger_dir=None,
    sample_name=None,
):
    """
    Reads a xenium sample from a directory path.

    Parameters
    ----------
    path (str): The directory path of the segmentation run.
    cells_as_circles (bool): Whether to include cell polygons as circles or not.
    cells_boundaries (bool): Whether to include cell boundaries or not.
    nucleus_boundaries (bool): Whether to include nucleus boundaries or not.
    cells_labels (bool): Whether to include cell labels or not.
    nucleus_labels (bool): Whether to include nucleus labels or not.
    transcripts (bool): Whether to include transcript locations or not.
    morphology_mip (bool): Whether to include morphology MIP or not.
    morphology_focus (bool): Whether to include morphology focus or not.
    aligned_images (bool): Whether to include aligned images or not.
    cells_table (bool): Whether to include cells table or not.
    anndata (bool): Whether to return only the anndata object or the full spatialdata object.
    xeniumranger_dir (str): Path to xeniumranger output directory (for proseg raw only)
    sample_name (str): The sample name.

    Returns
    -------
    If anndata, returns a tuple of the sample name and anndata object.
    Otherwise, returns a tuple of the sample name and spatialdata object.

    If sample_name is None, sample_name is not returned
    """
    path = Path(path)
    kwargs = dict(
        cells_as_circles=cells_as_circles,
        cells_boundaries=cells_boundaries,
        nucleus_boundaries=nucleus_boundaries,
        cells_labels=cells_labels,
        nucleus_labels=nucleus_labels,
        transcripts=transcripts,
        morphology_mip=morphology_mip,
        morphology_focus=morphology_focus,
        aligned_images=aligned_images,
        cells_table=cells_table,
    )

    # automatically check whether path is a folder with proseg raw outputs or in xeniumranger format
    if (path / "expected-counts.csv.gz").exists():
        reader = xenium_proseg
        kwargs["cells_boundaries_layers"] = cells_boundaries_layers
        kwargs["xeniumranger_dir"] = xeniumranger_dir
    else:
        reader = spatialdata_io.xenium

    sdata = reader(path, **kwargs)

    adata = sdata["table"]
    adata.obs_names = adata.obs["cell_id"].values

    metrics_path = path / "metrics_summary.csv"
    if metrics_path.exists():
        adata.uns["metrics_summary"] = pd.read_csv(metrics_path)

    if sample_name is None:
        if anndata:
            return adata
        else:
            return sdata
    else:
        if anndata:
            return sample_name, adata
        else:
            return sample_name, sdata


def read_xenium_samples(
    data_dirs,
    cells_as_circles=False,
    cells_boundaries=False,
    cells_boundaries_layers=False,
    nucleus_boundaries=False,
    cells_labels=False,
    nucleus_labels=False,
    transcripts=False,
    morphology_mip=False,
    morphology_focus=False,
    aligned_images=False,
    cells_table=True,
    anndata=False,
    sample_name_as_key=True,
    xeniumranger_dir=None,
    max_workers=None,
    pool_mode="thread",
):
    """
    Reads in a dictionary of sample directories and returns a dictionary of
    AnnData objects or spatialdata objects depending on the anndata flag.

    Parameters
    ----------
    data_dirs : dict or list
        A dictionary of sample directories or a list of paths to sample directories.
    cells_as_circles : bool, optional
        Whether to include cell boundary data as circles, by default False
    cells_boundaries : bool, optional
        Whether to include cell boundary data, by default False
    cells_boundaries : bool, optional
        Whether to include cell boundary layers data (for proseg raw only), by default False
    nucleus_boundaries : bool, optional
        Whether to include nucleus boundary data, by default False
    cells_labels : bool, optional
        Whether to include cell labels, by default False
    nucleus_labels : bool, optional
        Whether to include nucleus labels, by default False
    transcripts : bool, optional
        Whether to include transcript data, by default False
    morphology_mip : bool, optional
        Whether to include morphology data at the maximum intensity projection, by default False
    morphology_focus : bool, optional
        Whether to include morphology data at the focus, by default False
    aligned_images : bool, optional
        Whether to include aligned images, by default False
    cells_table (bool):
        Whether to include cells table or not, by default True
    anndata : bool, optional
        Whether to only return an AnnData object, by default False
    sample_name_as_key: bool, optional
        Whether to use the sample name as the key in the return dictionary,
        otherwise returns full path as key
    xeniumranger_dir: str, optional
        Path to xeniumranger output dir (for proseg raw only)
    max_workers : int, optional
        Maximum number of workers to use for parallel processing, by default None
    pool_mode : str, optional
        Pool mode for parallel processing, "thread" or "process", by default "thread"

    Returns
    -------
    dict
        A dictionary of sample names mapped to AnnData objects or spatialdata objects.
    """
    if isinstance(data_dirs, list):
        sample_names = [Path(path).stem if sample_name_as_key else path for path in data_dirs]
        data_dirs = {sample_name: path for sample_name, path in zip(sample_names, data_dirs)}

    # Parallel processing
    if pool_mode == "process":
        pool = ProcessPoolExecutor
    elif pool_mode == "thread":
        pool = ThreadPoolExecutor

    sdatas = {}
    with pool(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                read_xenium_sample,
                path,
                cells_as_circles,
                cells_boundaries,
                cells_boundaries_layers,
                nucleus_boundaries,
                cells_labels,
                nucleus_labels,
                transcripts,
                morphology_mip,
                morphology_focus,
                aligned_images,
                cells_table,
                anndata,
                xeniumranger_dir,
                sample_name,
            )
            for sample_name, path in data_dirs.items()
        ]

        for future in as_completed(futures):
            try:
                sample_name, result = future.result()
                sdatas[sample_name] = result
            except Exception as e:
                print(f"Error processing {e}")

    return sdatas


def get_gene_panel_info(path):
    with open(path, "r") as f:
        gene_panel = json.load(f)["payload"]["targets"]

    gene_panel_info = pd.DataFrame(columns=["codewords"])
    for i, g in enumerate(gene_panel):
        gene_panel_info.at[i, "gene_coverage"] = g["info"]["gene_coverage"]
        gene_panel_info.at[i, "id"] = g["type"]["data"].get("id")
        gene_panel_info.at[i, "name"] = g["type"]["data"]["name"]
        gene_panel_info.at[i, "codewords"] = g["codewords"]
        gene_panel_info.at[i, "source_category"] = g["source"]["category"]
        gene_panel_info.at[i, "source_design_id"] = g["source"]["identity"]["design_id"]
        gene_panel_info.at[i, "source_name"] = g["source"]["identity"]["name"]
        gene_panel_info.at[i, "source_version"] = g["source"]["identity"].get("version")
    return gene_panel_info


######### Xenium corrected counts readers


def _read_count_correction_sample(sample_name, corrected_counts_path):
    """Reads a 10x h5 file using scanpy."""
    try:
        adata = sc.read_10x_h5(corrected_counts_path)
        return sample_name, adata
    except Exception as e:
        print(f"Error reading {corrected_counts_path}: {e}")
        return sample_name, None  # Return None in case of an error


def read_count_correction_samples(xenium_paths, correction_methods):
    """
    Reads corrected count samples in parallel using ThreadPoolExecutor.

    Args:
        xenium_paths (dict): A dictionary where keys are correction methods and values are dictionaries
                            mapping sample names to corrected counts file paths.  Assumes `xenium_paths[correction_method]`
                            is a dictionary with keys as sample_name and values as path to the .h5 file.
        correction_methods (list): A list of correction methods.
    Returns:
        dict: A dictionary where keys are correction methods, and values are dictionaries mapping sample names
              to AnnData objects (or None if reading failed).
    """

    xenium_corrected_counts = {}

    for correction_method in correction_methods:  # Skip the first correction method
        xenium_corrected_counts[correction_method] = {}

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(_read_count_correction_sample, sample_name, xenium_corr_path): (
                    correction_method,
                    sample_name,
                )
                for sample_name, xenium_corr_path in xenium_paths[correction_method].items()
            }

            # Progress bar with total number of samples
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {correction_method}"):
                try:
                    sample_name, adata = future.result()
                    if adata is not None:
                        xenium_corrected_counts[correction_method][sample_name] = adata
                    else:
                        xenium_corrected_counts[correction_method][sample_name] = None
                except Exception as e:
                    correction_method, sample_name = futures[future]
                    xenium_corrected_counts[correction_method][sample_name] = None  # Store None in case of error

    return xenium_corrected_counts


###### Xenium cell type annotation reader
def get_anndata_from_object(data_object):
    """
    Checks if the input is a SpatialData object and returns its AnnData table.
    If the input is already an AnnData object, it returns it directly.
    """
    if isinstance(data_object, spatialdata.SpatialData):
        # It's a SpatialData object, return its table
        return data_object["table"]
    elif isinstance(data_object, ad.AnnData):
        # It's already an AnnData object, return as is
        return data_object


def read_annotations(data_dict, correction_methods, xenium_annot_paths, level, max_workers=8):
    """
    Assigns cell type annotation (and filters NaN cells) for a dictionary of AnnData objects using parallel threads.

    Parameters:
    - data_dict: dict of dicts containing AnnData or SpatialData objects per correction method.
    - correction_methods: list of strings indicating the correction methods.
    - xenium_annot_paths: dict of file paths to annotation parquet files.
    - level: string, the name of the column in annotations to assign to AnnData.obs.
    - max_workers: number of threads to use.
    """

    def process_raw(k, ad):
        if ad is None:
            return (k, None)

        if k[0] == "proseg_expected":
            ad.obs_names = ad.obs_names.astype(str)
            if not ad.obs_names[0].startswith("proseg-"):
                ad.obs_names = "proseg-" + ad.obs_names

        annot_path = xenium_annot_paths["raw"].get(k)
        if annot_path and Path(annot_path).exists():
            labels = pd.read_parquet(annot_path).set_index("cell_id").iloc[:, 0]
            ad.obs[level] = labels
            ad = ad[ad.obs[level].notna()].copy()
        else:
            print(f"Could not find annotation file for {k}")

        return (k, ad)

    # Check if dict contains spatialdata or anndata
    data_object = data_dict["raw"][list(data_dict["raw"])[0]]
    if isinstance(data_object, spatialdata.SpatialData):
        is_sdata = True
    else:
        is_sdata = False

    # --- Process 'raw' in parallel ---
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if is_sdata:
            futures = [executor.submit(process_raw, k, sd["table"]) for k, sd in data_dict["raw"].items()]
        else:
            futures = [executor.submit(process_raw, k, ad) for k, ad in data_dict["raw"].items()]

        for future in as_completed(futures):
            k, processed_ad = future.result()
            if processed_ad is not None:
                if is_sdata:
                    data_dict["raw"][k]["table"] = processed_ad
                else:
                    data_dict["raw"][k] = processed_ad

    # --- Add raw annotation to corrected counts  ---
    for correction_method in correction_methods:
        if correction_method == "raw":
            continue
        for k, ad_ in data_dict[correction_method].items():
            if ad_ is None:
                continue

            raw_ad = data_dict["raw"][k]
            if level in raw_ad.obs:
                ad_.obs[level] = raw_ad.obs[level]
                ad_ = ad_[ad_.obs[level].notna()].copy()
                data_dict[correction_method][k] = ad_
            else:
                print(f"Raw annotations missing for {k} when processing {correction_method}")
