import glob
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import tifffile
import torch

from matplotlib.collections import PatchCollection, PolyCollection
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
from skimage.draw import polygon, line_aa
from skimage.measure import label, regionprops_table
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torchvision.models import resnet152
from torchvision.transforms import functional as TF
from tqdm import tqdm


def visualize_cell_segmentation(parquet_path, limit=None):
    """
    Visualize cell segmentation masks from a Parquet file.
    
    Parameters:
    - parquet_path: Path to the Parquet file
    - limit_to_first_100: If True, only shows first 100 cell IDs (default: True)
    """
    # Read the Parquet file
    df = pd.read_parquet(parquet_path)
    
    # Get unique cell IDs
    unique_cell_ids = df['cell_id'].unique()
    
    # Optionally limit to first 100 cell IDs
    if limit is not None and len(unique_cell_ids) > limit:
        print(f"Limiting display to first 100 of {len(unique_cell_ids)} total cell IDs")
        selected_cell_ids = unique_cell_ids[:limit]
        df = df[df['cell_id'].isin(selected_cell_ids)]
    
    # Group vertices by cell_id
    polygons = []
    colors = []
    
    # Create a colormap for visualization
    cmap = plt.cm.get_cmap('tab20', len(df['cell_id'].unique()))
    
    for i, (cell_id, group) in enumerate(df.groupby('cell_id')):
        # Get vertices in order
        vertices = list(zip(group['vertex_x'], group['vertex_y']))
        polygons.append(vertices)
        colors.append(cmap(i))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Add polygons to plot
    poly_collection = PolyCollection(
        polygons,
        facecolors=colors,
        edgecolors='black',
        linewidths=0.5,
        alpha=0.7
    )
    ax.add_collection(poly_collection)
    
    # Auto-scale the plot
    ax.autoscale()
    
    # Set labels and title
    ax.set_title('Cell Segmentation Visualization')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    
    # Add colorbar legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(df['cell_id'].unique())-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Cell ID')
    
    plt.show()

def create_organoid_regions(parquet_path, buffer_distance=5, min_cell_count=20, 
                           plot_results=True, outline_thickness=1.0):
    """
    Create organoid regions by merging overlapping or nearby cell polygons.
    Returns individual cell outlines without merging, with specified thickness.
    
    Parameters:
    - parquet_path: Path to the Parquet file
    - buffer_distance: Distance to expand cells for merging (in coordinate units)
    - min_cell_count: Minimum number of cells required to form an organoid (default: 20)
    - plot_results: Whether to plot the results (default: True)
    - outline_thickness: Thickness of the cell outlines (default: 1.0)
    
    Returns:
    - organoids: List of Shapely Polygon/MultiPolygon objects
    - cell_counts: List of cell counts for each organoid
    - bounding_boxes: List of bounding boxes as (min_x, min_y, max_x, max_y) tuples
    """
    # Read the Parquet file
    df = pd.read_parquet(parquet_path)
    
    # Create dictionary to store cell polygons and their buffered versions
    cell_polygons = {}
    buffered_polygons = []
    
    # Convert each cell to a Shapely polygon and create buffered versions
    for cell_id, group in df.groupby('cell_id'):
        vertices = list(zip(group['vertex_x'], group['vertex_y']))
        poly = Polygon(vertices)
        cell_polygons[cell_id] = poly
        buffered_polygons.append(poly.buffer(buffer_distance))
    
    # Build spatial index for faster queries
    tree = STRtree(buffered_polygons)
    
    # Find connected components (organoids) using union-find approach
    parent = {i: i for i in range(len(buffered_polygons))}
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    
    # Find all intersecting polygons and merge them
    for i, poly in enumerate(buffered_polygons):
        for j in tree.query(poly):
            if i != j and poly.intersects(buffered_polygons[j]):
                root_i = find(i)
                root_j = find(j)
                if root_i != root_j:
                    parent[root_j] = root_i
    
    # Group polygons by their root parent
    groups = {}
    for i in range(len(buffered_polygons)):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)
    
    # Filter groups by cell count and create organoid groups (without merging)
    organoids = []
    cell_counts = []
    bounding_boxes = []
    valid_original_cells = []  # For visualization
    organoid_cell_groups = []  # Store individual cells for each organoid
    
    for group_indices in groups.values():
        if len(group_indices) >= min_cell_count:
            # Get the original polygons (not buffered)
            original_polys = [list(cell_polygons.values())[i] for i in group_indices]
            valid_original_cells.extend(original_polys)
            organoid_cell_groups.append(original_polys)
            
            # Store individual cells instead of merging
            organoids.append(original_polys)
            cell_counts.append(len(group_indices))
            
            # Get bounding box for the entire group of cells
            all_coords = []
            for poly in original_polys:
                all_coords.extend(poly.exterior.coords)
            all_coords = np.array(all_coords)
            min_x, min_y = np.min(all_coords, axis=0)
            max_x, max_y = np.max(all_coords, axis=0)
            bounding_boxes.append((min_x, min_y, max_x, max_y))
    
    if plot_results:
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot original cells (only those that contributed to organoids)
        original_patches = []
        for poly in valid_original_cells:
            original_patches.append(plt.Polygon(np.array(poly.exterior.coords), 
                                               closed=True, fill=False, 
                                               linewidth=outline_thickness, 
                                               edgecolor='blue'))
        
        pc1 = PatchCollection(original_patches, match_original=True)
        ax1.add_collection(pc1)
        ax1.set_title(f'Original Cells ({len(valid_original_cells)} cells in organoids)')
        ax1.autoscale()
        
        # Plot organoid groups with cell count labels and bounding boxes
        for i, (cell_group, bbox) in enumerate(zip(organoid_cell_groups, bounding_boxes)):
            # Draw individual cell outlines
            cell_patches = []
            for poly in cell_group:
                cell_patches.append(plt.Polygon(np.array(poly.exterior.coords), 
                                               closed=True, fill=False, 
                                               linewidth=outline_thickness, 
                                               edgecolor='red'))
            
            pc2 = PatchCollection(cell_patches, match_original=True)
            ax2.add_collection(pc2)
            
            # Add cell count label at centroid of the group
            all_coords = []
            for poly in cell_group:
                all_coords.extend(poly.exterior.coords)
            all_coords = np.array(all_coords)
            centroid_x = np.mean(all_coords[:, 0])
            centroid_y = np.mean(all_coords[:, 1])
            
            ax2.text(centroid_x, centroid_y, str(cell_counts[i]), 
                     ha='center', va='center', fontsize=8,
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
            # Draw bounding box
            min_x, min_y, max_x, max_y = bbox
            rect = plt.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y,
                                linewidth=1, edgecolor='green', facecolor='none', 
                                linestyle='--')
            ax2.add_patch(rect)
        
        ax2.set_title(f'Organoid Groups ({len(organoids)} regions, min {min_cell_count} cells each)')
        ax2.autoscale()
        
        plt.tight_layout()
        plt.show()
    
    return organoids, cell_counts, bounding_boxes


def generate_organoid_masks_with_square_bboxes(geo_df, scale=True, output_size=(224, 224), padding_ratio=0.1, outline_thickness=1, max_pixel_side_length=None, fill=False):
    """
    Generate masks with both original and square bounding boxes.
    
    Returns:
        Tuple of (organoid_masks, organoid_bboxes, square_bboxes) where:
        - organoid_masks: Dictionary mapping organoid IDs to mask numpy arrays
        - organoid_bboxes: Dictionary mapping organoid IDs to original bounding boxes with padding
        - square_bboxes: Dictionary mapping organoid IDs to square bounding boxes that match the mask content
    """
    import numpy as np
    from skimage.draw import line_aa
    import geopandas as gpd
    
    # Get unique organoid IDs
    organoid_ids = geo_df['component_and_cluster_labels'].unique()
    num_organoids = len(organoid_ids)
    print(f"Processing {num_organoids} organoids")
    
    # Dictionaries to store masks and bounding boxes
    organoid_masks = {}
    organoid_bboxes = {}  # Original bboxes with proportional padding
    square_bboxes = {}    # Square bboxes that match mask content
    
    # Process each organoid individually
    for organoid_id in organoid_ids:
        # Filter for this specific organoid using GeoPandas
        organoid_gdf = geo_df[geo_df['component_and_cluster_labels'] == organoid_id].copy()
        
        # Skip if no geometries
        if len(organoid_gdf) == 0:
            organoid_masks[organoid_id] = np.zeros(output_size[::-1], dtype=np.uint8)
            organoid_bboxes[organoid_id] = (0, 0, 0, 0)
            square_bboxes[organoid_id] = (0, 0, 0, 0)
            continue
        
        # Get bounding box for THIS organoid using GeoPandas total_bounds
        min_x, min_y, max_x, max_y = organoid_gdf.total_bounds
        
        # Calculate content dimensions
        content_width = max_x - min_x
        content_height = max_y - min_y
        
        # Handle zero dimensions
        if content_width == 0 or content_height == 0:
            organoid_masks[organoid_id] = np.zeros(output_size[::-1], dtype=np.uint8)
            padding = output_size[0] * padding_ratio  # Use output size for zero-dim case
            organoid_bboxes[organoid_id] = (min_x-padding, min_y-padding, max_x+padding, max_y+padding)
            square_bboxes[organoid_id] = (min_x-padding, min_y-padding, max_x+padding, max_y+padding)
            continue
        
        # Calculate scaling and offset (same as before)
        if not scale:
            final_scale = 1.0
            padding_x = content_width * padding_ratio
            padding_y = content_height * padding_ratio
            
            # Original bounding box with proportional padding
            bbox_with_padding = (
                min_x - padding_x,
                min_y - padding_y,
                max_x + padding_x,
                max_y + padding_y
            )
            
            effective_width = content_width + 2 * padding_x
            effective_height = content_height + 2 * padding_y
            final_offset_x = (output_size[0] - effective_width * final_scale) / 2 - (min_x - padding_x) * final_scale
            final_offset_y = (output_size[1] - effective_height * final_scale) / 2 - (min_y - padding_y) * final_scale
            
        elif max_pixel_side_length is not None:
            scale_factor = max(output_size) / max_pixel_side_length
            final_scale = scale_factor
            
            padding_x = content_width * padding_ratio
            padding_y = content_height * padding_ratio
            
            bbox_with_padding = (
                min_x - padding_x,
                min_y - padding_y,
                max_x + padding_x,
                max_y + padding_y
            )
            
            scaled_width = content_width * final_scale
            scaled_height = content_height * final_scale
            final_offset_x = (output_size[0] - scaled_width) / 2 - min_x * final_scale
            final_offset_y = (output_size[1] - scaled_height) / 2 - min_y * final_scale
            
        else:
            width_scale = (output_size[0] * (1 - 2*padding_ratio)) / content_width
            height_scale = (output_size[1] * (1 - 2*padding_ratio)) / content_height
            final_scale = min(width_scale, height_scale)
            
            padding_x = (content_width * padding_ratio) / (1 - 2*padding_ratio)
            padding_y = (content_height * padding_ratio) / (1 - 2*padding_ratio)
            
            bbox_with_padding = (
                min_x - padding_x,
                min_y - padding_y,
                max_x + padding_x,
                max_y + padding_y
            )
            
            final_offset_x = (output_size[0] - content_width * final_scale) / 2 - min_x * final_scale
            final_offset_y = (output_size[1] - content_height * final_scale) / 2 - min_y * final_scale
        
        # Store the original bounding box with proportional padding
        organoid_bboxes[organoid_id] = bbox_with_padding
        
        # Calculate SQUARE bounding box that matches the mask content
        # The mask content is centered and scaled to fit the output_size with padding
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # Calculate the actual content area that appears in the square mask
        if not scale:
            # For scale=False, we use the larger dimension plus padding
            larger_dim = max(content_width, content_height)
            square_half_size = (larger_dim * (1 + 2 * padding_ratio)) / 2
        elif max_pixel_side_length is not None:
            # Scale to fit max_pixel_side_length
            larger_dim = max(content_width, content_height)
            square_half_size = (larger_dim * (1 + 2 * padding_ratio)) / 2
        else:
            # Scale to fit output_size with padding
            # The effective content area in original coordinates
            effective_pixel_width = output_size[0] * (1 - 2 * padding_ratio)
            effective_pixel_height = output_size[1] * (1 - 2 * padding_ratio)
            
            # Convert back to original coordinates using the actual scale used
            effective_width = effective_pixel_width / final_scale
            effective_height = effective_pixel_height / final_scale
            
            # Use the larger dimension to ensure squareness
            larger_effective_dim = max(effective_width, effective_height)
            square_half_size = larger_effective_dim / 2
        
        square_bbox = (
            center_x - square_half_size,
            center_y - square_half_size,
            center_x + square_half_size,
            center_y + square_half_size
        )
        
        # Store the square bounding box
        square_bboxes[organoid_id] = square_bbox
        
        # Create empty mask (same as before)
        mask = np.zeros(output_size[::-1], dtype=np.uint8)
        
        # Function to draw thick line
        def draw_thick_line(mask, r0, c0, r1, c1, thickness):
            if thickness == 1:
                rr, cc, val = line_aa(r0, c0, r1, c1)
                mask[rr, cc] = np.maximum(mask[rr, cc], (val * 255).astype(np.uint8))
            else:
                for t in range(-thickness//2, thickness//2 + 1):
                    rr, cc, val = line_aa(r0 + t, c0 + t, r1 + t, c1 + t)
                    valid = (rr >= 0) & (rr < mask.shape[0]) & (cc >= 0) & (cc < mask.shape[1])
                    mask[rr[valid], cc[valid]] = np.maximum(mask[rr[valid], cc[valid]], 
                                                           (val[valid] * 255).astype(np.uint8))

        # Process each geometry in the organoid
        for idx, row in organoid_gdf.iterrows():
            geom = row['geometry']
            
            def process_polygon(poly):
                exterior_coords = np.array(poly.exterior.coords)
                xs = (exterior_coords[:, 0] * final_scale + final_offset_x).astype(int)
                ys = (exterior_coords[:, 1] * final_scale + final_offset_y).astype(int)

                xs = np.clip(xs, 0, output_size[0] - 1)
                ys = np.clip(ys, 0, output_size[1] - 1)

                # Fill if requested
                if fill:
                    rr, cc = polygon(ys, xs, mask.shape)
                    mask[rr, cc] = 255

                # Draw outline
                for i in range(len(xs)):
                    draw_thick_line(mask, ys[i], xs[i], ys[(i + 1) % len(xs)], xs[(i + 1) % len(xs)], outline_thickness)

                # Interiors (holes)
                for interior in poly.interiors:
                    interior_coords = np.array(interior.coords)
                    xs_i = (interior_coords[:, 0] * final_scale + final_offset_x).astype(int)
                    ys_i = (interior_coords[:, 1] * final_scale + final_offset_y).astype(int)

                    xs_i = np.clip(xs_i, 0, output_size[0] - 1)
                    ys_i = np.clip(ys_i, 0, output_size[1] - 1)

                    if fill:
                        rr, cc = polygon(ys_i, xs_i, mask.shape)
                        mask[rr, cc] = 0  # cut out the hole

                    for i in range(len(xs_i)):
                        draw_thick_line(mask, ys_i[i], xs_i[i], ys_i[(i + 1) % len(xs_i)], xs_i[(i + 1) % len(xs_i)], outline_thickness)

            if geom.geom_type == 'Polygon':
                process_polygon(geom)
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    process_polygon(poly)
        
        # Convert to binary mask
        mask = (mask > 0).astype(np.uint8)
        organoid_masks[organoid_id] = mask
    
    # Assert we have exactly one of each per organoid ID
    assert len(organoid_masks) == num_organoids
    assert len(organoid_bboxes) == num_organoids
    assert len(square_bboxes) == num_organoids
    
    print(f"Successfully generated {len(organoid_masks)} masks and bounding boxes")
    return organoid_masks, organoid_bboxes, square_bboxes


def polygon_to_mask(polygons, scale=True, output_size=(224, 224), padding_ratio=0.1, outline_thickness=1, max_pixel_side_length=None):
    """
    Convert Shapely Polygon(s) to an outline mask with specified thickness.
    
    Args:
        polygons: Shapely Polygon or list of Polygons
        scale: If False, use original content dimensions without scaling (only centers with padding)
        output_size: Tuple of (width, height) for output mask in pixels
        padding_ratio: Ratio of padding around content (0.1 = 10% padding)
        outline_thickness: Thickness of the outline in pixels
        max_pixel_side_length: Maximum side length in pixels for the largest organoid.
                               If provided, induces scale factor of max(output_size) / max_pixel_side_length.
                               Ignored when scale=False.
    
    Returns:
        Outline mask as numpy array (0=background, 1=foreground)
    """
    # Handle single polygon case
    if isinstance(polygons, (Polygon, MultiPolygon)):
        polygons = [polygons]
    
    # Convert MultiPolygons to individual Polygons
    all_polygons = []
    for poly in polygons:
        if isinstance(poly, MultiPolygon):
            all_polygons.extend(list(poly.geoms))
        elif isinstance(poly, Polygon):
            all_polygons.append(poly)
    
    # Check if we have any valid polygons
    if not all_polygons:
        return np.zeros(output_size[::-1], dtype=np.uint8)
    
    # Get combined bounds of all polygons
    min_x, min_y, max_x, max_y = MultiPolygon(all_polygons).bounds
    
    # Calculate content dimensions
    content_width = max_x - min_x
    content_height = max_y - min_y
    
    # Handle case where content has zero dimensions
    if content_width == 0 or content_height == 0:
        return np.zeros(output_size[::-1], dtype=np.uint8)
    
    # Calculate scaling and offset
    if not scale:
        # No scaling - use original dimensions, but apply padding and center
        final_scale = 1.0
        
        # Calculate padding in world coordinates (based on output size and padding ratio)
        padding_x = output_size[0] * padding_ratio / final_scale
        padding_y = output_size[1] * padding_ratio / final_scale
        
        # Calculate offset to center the content with padding
        # The effective content area is reduced by padding on both sides
        effective_width = content_width + 2 * padding_x
        effective_height = content_height + 2 * padding_y
        
        # Center the content within the padded area
        final_offset_x = (output_size[0] - effective_width * final_scale) / 2 - (min_x - padding_x) * final_scale
        final_offset_y = (output_size[1] - effective_height * final_scale) / 2 - (min_y - padding_y) * final_scale
        
    elif max_pixel_side_length is not None:
        # Scale factor: max(output_size) / max_pixel_side_length
        scale_factor = max(output_size) / max_pixel_side_length
        
        # Calculate the final scale
        final_scale = scale_factor
        
        # Calculate offset to center in output
        scaled_width = content_width * final_scale
        scaled_height = content_height * final_scale
        final_offset_x = (output_size[0] - scaled_width) / 2 - min_x * final_scale
        final_offset_y = (output_size[1] - scaled_height) / 2 - min_y * final_scale
        
    else:
        # Original behavior - scale to fit output_size with padding
        width_scale = (output_size[0] * (1 - 2*padding_ratio)) / content_width
        height_scale = (output_size[1] * (1 - 2*padding_ratio)) / content_height
        final_scale = min(width_scale, height_scale)
        
        # Calculate offset to center the content
        final_offset_x = (output_size[0] - content_width * final_scale) / 2 - min_x * final_scale
        final_offset_y = (output_size[1] - content_height * final_scale) / 2 - min_y * final_scale
    
    # Create empty mask
    mask = np.zeros(output_size[::-1], dtype=np.uint8)
    
    # Function to draw thick line
    def draw_thick_line(mask, r0, c0, r1, c1, thickness):
        if thickness == 1:
            # Single pixel line
            rr, cc, val = line_aa(r0, c0, r1, c1)
            mask[rr, cc] = np.maximum(mask[rr, cc], (val * 255).astype(np.uint8))
        else:
            # For thicker lines, draw multiple lines with offset
            for t in range(-thickness//2, thickness//2 + 1):
                rr, cc, val = line_aa(r0 + t, c0 + t, r1 + t, c1 + t)
                valid = (rr >= 0) & (rr < mask.shape[0]) & (cc >= 0) & (cc < mask.shape[1])
                mask[rr[valid], cc[valid]] = np.maximum(mask[rr[valid], cc[valid]], 
                                                       (val[valid] * 255).astype(np.uint8))
    
    # Rasterize each polygon outline
    for poly in all_polygons:
        if not poly.is_empty and poly.is_valid:
            # Process exterior
            exterior_coords = list(poly.exterior.coords)
            for i in range(len(exterior_coords)):
                x1, y1 = exterior_coords[i]
                x2, y2 = exterior_coords[(i + 1) % len(exterior_coords)]
                
                # Scale and shift coordinates
                x1_scaled = int(x1 * final_scale + final_offset_x)
                y1_scaled = int(y1 * final_scale + final_offset_y)
                x2_scaled = int(x2 * final_scale + final_offset_x)
                y2_scaled = int(y2 * final_scale + final_offset_y)
                
                # Clip coordinates to mask bounds
                x1_scaled = np.clip(x1_scaled, 0, output_size[0] - 1)
                y1_scaled = np.clip(y1_scaled, 0, output_size[1] - 1)
                x2_scaled = np.clip(x2_scaled, 0, output_size[0] - 1)
                y2_scaled = np.clip(y2_scaled, 0, output_size[1] - 1)
                
                # Draw thick line for this segment
                draw_thick_line(mask, y1_scaled, x1_scaled, y2_scaled, x2_scaled, outline_thickness)
            
            # Process interior holes
            for interior in poly.interiors:
                interior_coords = list(interior.coords)
                for i in range(len(interior_coords)):
                    x1, y1 = interior_coords[i]
                    x2, y2 = interior_coords[(i + 1) % len(interior_coords)]
                    
                    # Scale and shift coordinates
                    x1_scaled = int(x1 * final_scale + final_offset_x)
                    y1_scaled = int(y1 * final_scale + final_offset_y)
                    x2_scaled = int(x2 * final_scale + final_offset_x)
                    y2_scaled = int(y2 * final_scale + final_offset_y)
                    
                    # Clip coordinates to mask bounds
                    x1_scaled = np.clip(x1_scaled, 0, output_size[0] - 1)
                    y1_scaled = np.clip(y1_scaled, 0, output_size[1] - 1)
                    x2_scaled = np.clip(x2_scaled, 0, output_size[0] - 1)
                    y2_scaled = np.clip(y2_scaled, 0, output_size[1] - 1)
                    
                    # Draw thick line for this segment
                    draw_thick_line(mask, y1_scaled, x1_scaled, y2_scaled, x2_scaled, outline_thickness)
    
    # Convert to binary mask (0 or 1)
    mask = (mask > 0).astype(np.uint8)
    
    return mask


class NorkinOrganoidDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        scale=True,
        standardize_scale=False,
        fill=False,
        organoid_cell_mapping_path="/work/PRTNR/CHUV/DIR/rgottar1/spatial/data/norkin_organoid/results/xenium/segment_organoids/organoids_ids.parquet",
        raw_data_path='/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/norkin_organoid/data/xenium/raw/CRC_PDO',
        save_path=f'/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/data/organoid_masks_official_v3',
    ):
        """
        Args:
            raw_data_path: Path to directory containing cell_boundaries.parquet files
            save_path: Path to save/load preprocessed organoid masks

        Object params:
            - organoid_cell_mapping_path: Path to CSV mapping cell IDs to organoid IDs
            - raw_data_path: Path to directory containing cell_boundaries.parquet files
            - save_path: Path to save/load preprocessed organoid masks

        """
        super().__init__()

        if fill: 
            save_path += "_fill"
        else: 
            save_path += "_nofill"

        if scale:
            standardize_scale_suffix = f"_standardized.pkl" if standardize_scale else "_unstandardized.pkl"
            save_path += standardize_scale_suffix
        else:
            save_path += "_no_scale.pkl"

        self.fill = fill
        self.organoid_cell_mapping_path = organoid_cell_mapping_path
        self.raw_data_path = raw_data_path
        self.save_path = save_path
        self.scale = scale
        self.standardize_scale = standardize_scale

        self.organoid_dfs = {}
        self.organoid_masks = {}
        self.organoid_bboxes = {}
        self.organoid_square_bboxes = {}
        self.organoid_joint_ids = {}
        self.organoid_ids = []
        self.organoid_count_threshold = 20

        # Try to load preprocessed masks if they exist
        if os.path.exists(self.save_path) and isinstance(joblib.load(self.save_path), dict):
            print(f"Loading preprocessed masks from {self.save_path}")
            with open(self.save_path, 'rb') as f:
                data_obj = pickle.load(f)
                self.organoid_masks = data_obj['organoid_masks']
                self.organoid_bboxes = data_obj['organoid_bboxes']
                self.organoid_square_bboxes = data_obj['organoid_square_bboxes']
                self.organoid_joint_ids = data_obj['organoid_joint_ids']
                self.organoid_joint_ids_encoded = data_obj['organoid_joint_ids_encoded']
                self.organoid_dfs = data_obj['organoid_dfs']
                self.organoid_ids = data_obj['organoid_ids']
        else:
            # Process parquet files if no saved masks exist
            self._process_raw_data_from_sdata()
            self._save_masks()

    def get_organoid_df_by_id(self, patient_id=None, joint_id=None):
        if patient_id is None and joint_id is None:
            raise Exception("patient ID must be a part of joint id.")
        if patient_id is not None and joint_id is not None:
            raise Exception("only patient id or joint id must be provided.")
        if patient_id is not None:
            id = patient_id
        if joint_id is not None:
            id = joint_id
        
        for proseg_id_tuple, joined_df in self.organoid_dfs.items(): 
            complete_key = "_".join(proseg_id_tuple)
            if id in complete_key:
                return joined_df
        
        raise Exception(f"no element with id {id} found.")
    
    def _get_spatial_anndatas(self):
        import sys
        sys.path.append('/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/norkin_organoid/workflow/scripts')
        import coda
        import readwrite
        cfg = readwrite.config()

        ORGANOID_COUNT_THRESHOLD = 20

        # input params
        correction_method = 'raw'
        segmentation = 'proseg_expected'
        condition = ['CRC_PDO','CRC_PDO_CAF', 'CRC_PDO_DEV']  
        panel = 'all'

        xenium_dir = Path(cfg['xenium_processed_dir'])
        xenium_count_correction_dir = Path(cfg['xenium_count_correction_dir'])
        xenium_std_seurat_analysis_dir = Path(cfg['xenium_std_seurat_analysis_dir'])
        xenium_cell_type_annotation_dir = Path(cfg['xenium_cell_type_annotation_dir'])
        results_dir = Path(cfg['results_dir'])

        xenium_levels = ['segmentation','condition','panel','donor','sample']
        normalisation = 'lognorm'
        reference = 'GEO_GSE236581' # 'GEO_GSE178341'
        method = 'rctd_class_aware'
        level = 'Level1'

        # fixed params
        BATCH_KEY = 'dataset_id'
        SPATIAL_KEY = 'spatial'
        N_CLUSTERS_RANGE = (5,19)
        MAX_RUNS = 10
        CONVERGENCE_TOL = 0.001
        OUTPUT_LABELS = results_dir/'xenium/cellcharter/labels.parquet'
        OUTPUT_SCVI_MODEL = results_dir/'xenium/cellcharter/scvi_model'
        OUTPUT_CELLCHARTER_MODELS = results_dir/'xenium/cellcharter/cellcharter_models'
        OUTPUT_PLOT = results_dir/'xenium/cellcharter/autok_stability.png'

        # read samples
        xenium_paths, xenium_annot_paths = readwrite.discover_xenium_paths(
            analysis_dir=xenium_std_seurat_analysis_dir,
            data_dir=xenium_dir,
            annotation_dir=xenium_cell_type_annotation_dir,
            correction_dir=xenium_count_correction_dir,
            normalisation=normalisation,
            reference=reference,
            method=method,
            level=level,
            correction_methods_filter=[correction_method],
            segmentations_filter=[segmentation],
            conditions_filter=[*condition] if condition != 'all' else None,
            panels_filter=[panel] if panel != 'all' else None
        )

        # set transcripts=True to load individual transcripts positions)
        if correction_method != 'raw':
            ads = readwrite.read_count_correction_samples(xenium_paths,[correction_method])
        else:
            ads = {}
            ads['raw'] = readwrite.read_xenium_samples(
                xenium_paths['raw'],  
                anndata=False,
                cells_boundaries=True, 
                pool_mode="thread",
                max_workers=6
            )

        # add cell type annotation from raw to all correction methods
        readwrite.read_annotations(ads, [correction_method], xenium_annot_paths, level, max_workers=8)

        return ads

    def _process_raw_data_from_sdata(self):
        organoid_cell_mapping = pd.read_parquet(self.organoid_cell_mapping_path)
        organoid_cell_mapping = organoid_cell_mapping[['component_and_cluster_labels']]
        # organoid_cell_mapping = organoid_cell_mapping[organoid_cell_mapping['Organoid_ID'] > 0]

        print("Loaded anndata...")
        if not os.path.exists("/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/notebooks/ads_v2.pkl"):
            ads = self._get_spatial_anndatas()
            joblib.dump(ads, "/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/notebooks/ads_v2.pkl")
        else:
            ads = joblib.load("/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/notebooks/ads_v2.pkl")
        print("Loaded anndata.")

        max_pixel_side_length = -np.inf
        dfs = {}
        
        for proseg_key in tqdm(ads['raw'].keys(), desc="Processing geo_df samples..."):
            method = proseg_key[2]
            patient_id = proseg_key[3]
            joint_id = f"{method}__{patient_id}"

            proseg_key_str = "_".join(proseg_key) + "_proseg"
            geo_df = ads['raw'][proseg_key].shapes['cells_boundaries']
            geo_df['full_cell_id'] = geo_df['cell_id'].apply(
                lambda x: f"{proseg_key_str}-{x}"
            )
            geo_df['patient_id'] = patient_id
            geo_df['method'] = method
            geo_df['joint_id'] = joint_id

            # columns: foll_cell_id, Cell_ID, Organoid_ID, patient_id, method, geometry
            joined_df = geo_df.merge(organoid_cell_mapping, 
                        left_on='full_cell_id', 
                        right_on='full_id', 
                        how='inner')

            # Get organoid IDs with at least 20 cells
            # to be clear, "component_and_cluster_labels" is organoid id... don't ask why :P 
            organoid_counts = joined_df['component_and_cluster_labels'].value_counts()
            valid_organoids = organoid_counts[organoid_counts >= self.organoid_count_threshold].index

            # Filter the DataFrame
            joined_df = joined_df[joined_df['component_and_cluster_labels'].isin(valid_organoids)]
            joined_df = joined_df[joined_df['component_and_cluster_labels'] != 0]
            
            for organoid_id, group in joined_df.groupby('component_and_cluster_labels'):
                # Get bounding box
                coords = group.total_bounds
                            # Update max pixel side length for standardization
                dims = lambda coords: (coords[2] - coords[0], coords[3] - coords[1])
                max_pixel_side_length = max([dims(coords)[0], dims(coords)[1], max_pixel_side_length])

            dfs[proseg_key] = {
                "joined_df": joined_df,
                "max_pixel_side_length": max_pixel_side_length
            }
            
            self.organoid_dfs[proseg_key] = joined_df
            print(f"Max pixel side length across all organoids: {max_pixel_side_length}")

        max_pixel_side_length = max([obj['max_pixel_side_length'] for obj in dfs.values()])
        for joined_df_obj in tqdm(dfs.values(), desc="Generating organoid masks..."):
            joined_df = joined_df_obj['joined_df']
            organoid_masks, organoid_bboxes, organoid_square_bboxes = generate_organoid_masks_with_square_bboxes(
                joined_df,
                scale=self.scale,
                output_size=(224, 224),
                padding_ratio=0.1,
                outline_thickness=1,
                fill=self.fill,
                max_pixel_side_length=max_pixel_side_length if self.standardize_scale else None
            )
            joint_id = joined_df['joint_id'].iloc[0]
            assert joined_df['joint_id'].nunique() == 1, "Multiple joint_ids in one dataframe"

            for key in organoid_masks.keys():
                self.organoid_masks[key] = organoid_masks[key]
                self.organoid_bboxes[key] = organoid_bboxes[key]
                self.organoid_square_bboxes[key] = organoid_square_bboxes[key]
                self.organoid_joint_ids[key] = joint_id
                self.organoid_ids.append(key)

            print(f"created {len(organoid_masks)} organoids from joint id {joint_id}")

        encoded_labels = LabelEncoder().fit_transform(list(self.organoid_joint_ids.values()))
        self.organoid_joint_ids_encoded = encoded_labels

    def _process_raw_data_from_parquet(self):
        """Process all parquet files to generate organoid masks"""
        all_cell_boundary_parquet_files = glob.glob(
            os.path.join(self.raw_data_path, '**', 'cell_boundaries.parquet'), 
            recursive=True
        )

        all_organoids = []
        
        max_pixel_side_length = -np.inf
        for cell_boundary_pth in tqdm(all_cell_boundary_parquet_files, desc="Generating organoid polygons..."):
            organoids, _, boundary_boxes = create_organoid_regions(
                cell_boundary_pth, 
                buffer_distance=5, 
                min_cell_count=20, 
                plot_results=False
            )
            all_organoids.extend(organoids)

            # Extract patient ID from path and add to patient id list
            # path example: /work/PRTNR/CHUV/DIR/rgottar1/spatial/data/norkin_organoid/data/xenium/raw/CRC_PDO/hImmune_v1_mm/1DCI/output-XETG00059__0021741__1DCL__20250319__172035/cell_boundaries.parquet
            patient_id = cell_boundary_pth.split('/')[-3]
            self.organoid_patient_ids += [patient_id] * len(organoids)
            print(f"created {len(organoids)} organoids from patient {patient_id}")

            # Update max pixel side length for standardization
            dims = lambda coords: (coords[2] - coords[0], coords[3] - coords[1])
            all_dims = lambda boundary_boxes: [dims(b) for b in boundary_boxes]
            max_dim = lambda boundary_boxes: np.max(np.array(all_dims(boundary_boxes)), axis=0).max()

            max_pixel_side_length = max([max_dim(boundary_boxes), max_pixel_side_length])
            print(f"Max pixel side length across all organoids: {max_pixel_side_length}")

        for organoid in tqdm(all_organoids, desc="Generating organoid masks..."): 
            organoid_mask = polygon_to_mask(
                organoid, 
                scale=self.scale,
                output_size=(224, 224), 
                outline_thickness=1,
                padding_ratio=0.1,
                max_pixel_side_length=max_pixel_side_length if self.standardize_scale else None
            )
            self.organoid_masks.append(organoid_mask)

        self.organoid_masks = np.array(self.organoid_masks)
        # Create and fit the encoder
        encoded_labels = LabelEncoder().fit_transform(self.organoid_patient_ids)
        self.organoid_patient_ids_encoded = encoded_labels
    
    def _save_masks(self):
        """Save processed masks to disk"""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'wb') as f:
            pickle.dump({
                "organoid_masks": self.organoid_masks, 
                "organoid_bboxes": self.organoid_bboxes,
                "organoid_square_bboxes": self.organoid_square_bboxes,
                "organoid_joint_ids": self.organoid_joint_ids,
                "organoid_joint_ids_encoded": self.organoid_joint_ids_encoded,
                "organoid_dfs": self.organoid_dfs,
                "organoid_ids": self.organoid_ids,
            }, f)
        print(f"Saved {len(self.organoid_masks)} organoid masks to {self.save_path}")
    
    def __len__(self):
        return len(self.organoid_masks)
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Handle slice objects and return a batched tensor
            keys = list(self.organoid_masks.keys())
            sliced_keys = keys[idx]
            tensors = [torch.FloatTensor(self.organoid_masks[key]).unsqueeze(0) for key in sliced_keys]
            return torch.cat(tensors, dim=0)
        elif isinstance(idx, list):
            # Handle list indices and return a batched tensor
            tensors = [torch.FloatTensor(self.organoid_masks[list(self.organoid_masks.keys())[i]]).unsqueeze(0) for i in idx]
            return torch.cat(tensors, dim=0)
        else:
            # Handle single integer indices
            return torch.FloatTensor(self.organoid_masks[list(self.organoid_masks.keys())[idx]]).unsqueeze(0)

def get_morphological_features(masks):
    """
    Compute morphological features for each binary mask, measuring properties of all regions collectively.
    
    Args:
        masks: List of binary masks (2D numpy arrays)
        
    Returns:
        List of dictionaries containing morphological features for each mask
    """
    from skimage.measure import label, regionprops
    from skimage.morphology import convex_hull_image
    from tqdm import tqdm
    import numpy as np

    features = []
    for mask in tqdm(masks, desc="Calculating morphological features"):
        # Label connected components
        labeled = label(mask)
        regions = regionprops(labeled)
        
        # Initialize aggregated features
        mask_features = {
            'area': 0,
            'perimeter': 0,
            'eccentricity': 0,
            'solidity': 0,
            'extent': 0,
            'major_axis_length': 0,
            'minor_axis_length': 0,
            'convex_hull_blank_percentage': 0
        }
        
        if len(regions) > 0:
            # Calculate convex hull for all regions together
            combined_mask = labeled > 0
            convex_hull = convex_hull_image(combined_mask)
            
            # Calculate blank pixels in convex hull
            blank_pixels = np.sum(convex_hull & ~combined_mask)
            hull_pixels = np.sum(convex_hull)
            blank_percentage = (blank_pixels / hull_pixels) * 100 if hull_pixels > 0 else 0
            
            # Aggregate properties across all regions
            mask_features = {
                'area': sum(r.area for r in regions),
                'perimeter': sum(r.perimeter for r in regions),
                'eccentricity': np.mean([r.eccentricity for r in regions]),
                'solidity': np.mean([r.solidity for r in regions]),
                'extent': np.mean([r.extent for r in regions]),
                'major_axis_length': np.mean([r.major_axis_length for r in regions]),
                'minor_axis_length': np.mean([r.minor_axis_length for r in regions]),
                'convex_hull_blank_percentage': blank_percentage
            }
        
        features.append(mask_features)
    
    # Verify output length matches input length
    assert len(features) == len(masks), \
        f"Output length {len(features)} doesn't match input length {len(masks)}"
    
    return features

def get_morphological_features(masks):
    """
    Compute morphological features for each binary mask, measuring properties of all regions collectively.
    
    Args:
        masks: List of binary masks (2D numpy arrays)
        
    Returns:
        List of dictionaries containing morphological features for each mask
    """
    from skimage.measure import label, regionprops
    from skimage.morphology import convex_hull_image
    from scipy import ndimage as ndi
    from tqdm import tqdm
    import numpy as np

    features = []
    for mask in tqdm(masks, desc="Calculating morphological features"):
        # Label connected components
        labeled = label(mask)
        regions = regionprops(labeled)
        
        # Initialize aggregated features
        mask_features = {
            'area': 0,
            'perimeter': 0,
            'eccentricity': 0,
            'solidity': 0,
            'extent': 0,
            'major_axis_length': 0,
            'minor_axis_length': 0,
            'convex_hull_blank_percentage': 0,
            'perimeter_sharpness': 0,
            'median_distance_to_edge': 0,
            'num_holes': 0,
            'interior_holes_percentage': 0
        }
        
        if len(regions) > 0:
            # Calculate convex hull for all regions together
            combined_mask = labeled > 0
            convex_hull = convex_hull_image(combined_mask)
            
            # Calculate blank pixels in convex hull
            blank_pixels = np.sum(convex_hull & ~combined_mask)
            hull_pixels = np.sum(convex_hull)
            blank_percentage = (blank_pixels / hull_pixels) * 100 if hull_pixels > 0 else 0
            
            # Calculate perimeter sharpness using gradient magnitude
            sobel_x = ndi.sobel(combined_mask.astype(float), axis=0)
            sobel_y = ndi.sobel(combined_mask.astype(float), axis=1)
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Perimeter sharpness: mean gradient magnitude at the boundary
            perimeter_mask = ndi.binary_dilation(combined_mask) & ~combined_mask
            perimeter_sharpness = np.mean(gradient_magnitude[perimeter_mask]) if np.any(perimeter_mask) else 0
            
            # Calculate median distance to edge using distance transform
            distance_transform = ndi.distance_transform_edt(combined_mask)
            median_distance = np.median(distance_transform[combined_mask]) if np.any(combined_mask) else 0
            
            # Calculate number of holes (inverted regions in convex hull)
            inverted_in_hull = convex_hull & ~combined_mask
            labeled_holes = label(inverted_in_hull)
            num_holes = np.max(labeled_holes)  # Number of connected hole regions
            
            # Calculate interior holes percentage - black points completely surrounded by white
            # Fill all holes in the binary mask
            filled_mask = ndi.binary_fill_holes(combined_mask)
            
            # Interior holes are the difference between filled mask and original mask
            interior_holes = filled_mask & ~combined_mask
            interior_holes_count = np.sum(interior_holes)
            total_object_pixels = np.sum(combined_mask)
            
            # Calculate percentage of interior holes relative to total object area
            if total_object_pixels > 0:
                interior_holes_percentage = (interior_holes_count / total_object_pixels) * 100
            else:
                interior_holes_percentage = 0
            
            # Aggregate properties across all regions
            mask_features = {
                'area': sum(r.area for r in regions),
                'perimeter': sum(r.perimeter for r in regions),
                'eccentricity': np.mean([r.eccentricity for r in regions]),
                'solidity': np.mean([r.solidity for r in regions]),
                'extent': np.mean([r.extent for r in regions]),
                'major_axis_length': np.mean([r.major_axis_length for r in regions]),
                'minor_axis_length': np.mean([r.minor_axis_length for r in regions]),
                'convex_hull_blank_percentage': blank_percentage,
                'perimeter_sharpness': perimeter_sharpness,
                'median_distance_to_edge': median_distance,
                'num_holes': num_holes,
                'interior_holes_percentage': interior_holes_percentage
            }
        
        features.append(mask_features)
    
    return features

def get_resnet152_embeddings(X, morphological_features=None, fine_tune=False, 
                           num_epochs=5, learning_rate=1e-4, batch_size=32,
                           validation_split=0.2, early_stopping_patience=None):
    """
    Get embeddings from a ResNet-152 model, with optional fine-tuning on morphological features.

    Args:
        X: Input tensor of shape (N, C, H, W) for ResNet-152 (C=3, H=W=224).
        morphological_features: Optional morphological features for fine-tuning (N, feature_dim).
        fine_tune: Whether to fine-tune the model on morphological features.
        num_epochs: Number of epochs for fine-tuning.
        learning_rate: Learning rate for fine-tuning.
        batch_size: Batch size for processing (default: 64).
        validation_split: Fraction of data to use for validation (default: 0.2).
        early_stopping_patience: Number of epochs to wait before early stopping (None to disable).

    Returns:
        embeddings: Tensor of shape (N, 2048) with ResNet-152 embeddings.
    """
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import torch
    import numpy as np
    from sklearn.model_selection import train_test_split

    # Load pre-trained ResNet-152 model
    resnet_model = resnet152(pretrained=True)
    resnet_model.fc = nn.Identity()  # Remove the classification head to get embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_model = resnet_model.to(device)

    if fine_tune and morphological_features is not None:
        # Convert morphological features to float tensor if needed
        if not isinstance(morphological_features, torch.Tensor):
            morphological_features = torch.FloatTensor(morphological_features)
        
        # Create TensorDataset
        dataset = torch.utils.data.TensorDataset(X, morphological_features)
        
        # Split data into training and validation sets
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [1 - validation_split, validation_split],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create DataLoaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        # Add a small MLP head for fine-tuning
        feature_dim = morphological_features.shape[1]
        mlp_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, feature_dim)
        ).to(device)
        model = nn.Sequential(resnet_model, mlp_head)
        model.train()

        # Define optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()


        # Training variables
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        # Fine-tune the model
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            for batch_X, batch_features in train_loader:
                batch_X = batch_X.to(device)
                batch_features = batch_features.to(device)
                
                # Apply random rotation augmentation (0-360 degrees)
                angles = torch.rand(batch_X.size(0), device=device) * 360  # Uniform 0-360 degrees
                rotated_batch = torch.zeros_like(batch_X)
                for i in range(batch_X.size(0)):
                    rotated_batch[i] = TF.rotate(batch_X[i].unsqueeze(0), angles[i].item()).squeeze(0)
                
                optimizer.zero_grad()
                embeddings = resnet_model(rotated_batch)  # Use augmented images
                predictions = mlp_head(embeddings)
                loss = criterion(predictions, batch_features)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
            
            train_loss /= len(train_dataset)
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_features in val_loader:
                    batch_X = batch_X.to(device)
                    batch_features = batch_features.to(device)
                    
                    embeddings = resnet_model(batch_X)
                    predictions = mlp_head(embeddings)
                    loss = criterion(predictions, batch_features)
                    val_loss += loss.item() * batch_X.size(0)
            
            val_loss /= len(val_dataset)

            # Print progress
            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Val Loss: {val_loss:.4f}")

            # Early stopping check
            if early_stopping_patience is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    # Save best model weights
                    best_model_state = {
                        'resnet': resnet_model.state_dict(),
                        'mlp_head': mlp_head.state_dict()
                    }
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        # Restore best model
                        resnet_model.load_state_dict(best_model_state['resnet'])
                        mlp_head.load_state_dict(best_model_state['mlp_head'])
                        break

        # Return model to eval mode
        resnet_model.eval()

    # Get embeddings for all data
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size].to(device)
            batch_emb = resnet_model(batch_X)
            embeddings.append(batch_emb.cpu())
    embeddings = torch.cat(embeddings, dim=0)

    return F.normalize(embeddings, dim=-1)  # Normalize embeddings for consistency

if __name__ == "__main__":
    # Example usage
    dataset = NorkinOrganoidDataset(standardize_scale=False, scale=True, fill=True)
    print(f"Loaded {len(dataset)} organoid masks")

    print("Getting morph features...")
    masks = dataset[:]
    morphological_features = get_morphological_features(masks)
    masks = masks.unsqueeze(1).repeat(1, 3, 1, 1)

    # Morph features
    morph_data_df = pd.DataFrame(morphological_features)
    morph_data_matrix = morph_data_df.values

    # Apply MinMaxScaler to scale each column to the range [0, 1]
    scaler = MinMaxScaler()
    morph_data_matrix = scaler.fit_transform(morph_data_matrix)
    morph_data_matrix.shape

    print("training...")
    embeddings = get_resnet152_embeddings(masks, morphological_features=morph_data_matrix, fine_tune=True)
    with open("embeddings_resnet152_v3_filled.pkl", "wb") as f:
        pickle.dump(embeddings, f)