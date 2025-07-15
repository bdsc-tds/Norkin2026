import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import tifffile
import torch

from matplotlib.collections import PolyCollection
from skimage.draw import polygon
from skimage.measure import label, regionprops_table
from sklearn.preprocessing import MinMaxScaler
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
from tqdm import tqdm
from torchvision.models import resnet152
from torchvision.transforms import functional as TF


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

def create_organoid_regions(parquet_path, buffer_distance=5, min_cell_count=20, plot_results=True):
    """
    Create organoid regions by merging overlapping or nearby cell polygons.
    
    Parameters:
    - parquet_path: Path to the Parquet file
    - buffer_distance: Distance to expand cells for merging (in coordinate units)
    - min_cell_count: Minimum number of cells required to form an organoid (default: 20)
    - plot_results: Whether to plot the results (default: True)
    
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
    
    # Filter groups by cell count and create merged organoids
    organoids = []
    cell_counts = []
    bounding_boxes = []
    valid_original_cells = []  # For visualization
    
    for group_indices in groups.values():
        if len(group_indices) >= min_cell_count:
            # Get the original polygons (not buffered)
            original_polys = [list(cell_polygons.values())[i] for i in group_indices]
            valid_original_cells.extend(original_polys)
            
            # Merge the buffered versions
            to_merge = [buffered_polygons[i] for i in group_indices]
            merged = unary_union(to_merge)
            
            # Sometimes buffer creates small artifacts - we can simplify
            merged = merged.simplify(buffer_distance/2)
            organoids.append(merged)
            cell_counts.append(len(group_indices))
            
            # Get bounding box (min_x, min_y, max_x, max_y)
            bounding_boxes.append(merged.bounds)
    
    if plot_results:
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot original cells (only those that contributed to organoids)
        original_patches = []
        for poly in valid_original_cells:
            original_patches.append(plt.Polygon(np.array(poly.exterior.coords), closed=True))
        
        pc1 = PatchCollection(original_patches, facecolors='blue', edgecolors='black', 
                             alpha=0.3, linewidths=0.5)
        ax1.add_collection(pc1)
        ax1.set_title(f'Original Cells ({len(valid_original_cells)} cells in organoids)')
        ax1.autoscale()
        
        # Plot merged organoids with cell count labels and bounding boxes
        organoid_patches = []
        for i, (organoid, bbox) in enumerate(zip(organoids, bounding_boxes)):
            if isinstance(organoid, Polygon):
                patch = plt.Polygon(np.array(organoid.exterior.coords), closed=True)
                organoid_patches.append(patch)
                
                # Add cell count label at centroid
                centroid = organoid.centroid
                ax2.text(centroid.x, centroid.y, str(cell_counts[i]), 
                         ha='center', va='center', fontsize=8,
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                
                # Draw bounding box
                min_x, min_y, max_x, max_y = bbox
                rect = plt.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y,
                                    linewidth=1, edgecolor='green', facecolor='none', linestyle='--')
                ax2.add_patch(rect)
        
        pc2 = PatchCollection(organoid_patches, facecolors='red', edgecolors='black', 
                             alpha=0.5, linewidths=1)
        ax2.add_collection(pc2)
        ax2.set_title(f'Merged Organoids ({len(organoids)} regions, min {min_cell_count} cells each)')
        ax2.autoscale()
        
        plt.tight_layout()
        plt.show()
    
    return organoids, cell_counts, bounding_boxes


def polygon_to_mask(polygons, output_size=(224, 224), padding_ratio=0.1):
    """
    Convert Shapely Polygon(s) to a binary mask with padding.
    
    Args:
        polygons: Shapely Polygon or list of Polygons
        output_size: Tuple of (width, height) for output mask
        padding_ratio: Ratio of padding around content (0.1 = 10% padding)
    
    Returns:
        Binary mask as numpy array (0=background, 1=foreground)
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
    
    # Get combined bounds of all polygons
    min_x, min_y, max_x, max_y = MultiPolygon(all_polygons).bounds
    
    # Calculate content dimensions
    content_width = max_x - min_x
    content_height = max_y - min_y
    
    # Calculate scaling factor to fit content + padding
    width_scale = (output_size[0] * (1 - 2*padding_ratio)) / content_width
    height_scale = (output_size[1] * (1 - 2*padding_ratio)) / content_height
    scale = min(width_scale, height_scale)
    
    # Calculate offset to center the content
    offset_x = (output_size[0] - content_width * scale) / 2 - min_x * scale
    offset_y = (output_size[1] - content_height * scale) / 2 - min_y * scale
    
    # Create empty mask
    mask = np.zeros(output_size[::-1], dtype=np.uint8)  # Note: (height, width)
    
    # Rasterize each polygon
    for poly in all_polygons:
        if not poly.is_empty:
            # Get exterior coordinates
            x, y = poly.exterior.coords.xy
            
            # Scale and shift coordinates
            x_scaled = (np.array(x) * scale + offset_x).astype(int)
            y_scaled = (np.array(y) * scale + offset_y).astype(int)
            
            # Create polygon mask
            rr, cc = polygon(y_scaled, x_scaled, shape=output_size[::-1])
            mask[rr, cc] = 1
            
            # Handle holes if they exist
            for interior in poly.interiors:
                x, y = interior.coords.xy
                x_scaled = (np.array(x) * scale + offset_x).astype(int)
                y_scaled = (np.array(y) * scale + offset_y).astype(int)
                rr, cc = polygon(y_scaled, x_scaled, shape=output_size[::-1])
                mask[rr, cc] = 0
    
    return mask

class NorkinOrganoidDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data_path='/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/norkin_organoid/data/xenium/raw/CRC_PDO',
                 save_path='/work/PRTNR/CHUV/DIR/rgottar1/spatial/env/lmcconn1/norkin_organoid/code/organoid_masks.pkl'):
        """
        Args:
            raw_data_path: Path to directory containing cell_boundaries.parquet files
            save_path: Path to save/load preprocessed organoid masks
        """
        super(NorkinOrganoidDataset, self).__init__()
        self.raw_data_path = raw_data_path
        self.save_path = save_path
        self.organoid_masks = []
        
        # Try to load preprocessed masks if they exist
        if os.path.exists(self.save_path):
            print(f"Loading preprocessed masks from {self.save_path}")
            with open(self.save_path, 'rb') as f:
                self.organoid_masks = pickle.load(f)
        else:
            # Process parquet files if no saved masks exist
            self._process_raw_data()
            self._save_masks()
    
    def _process_raw_data(self):
        """Process all parquet files to generate organoid masks"""
        all_cell_boundary_parquet_files = glob.glob(
            os.path.join(self.raw_data_path, '**', 'cell_boundaries.parquet'), 
            recursive=True
        )
        
        for cell_boundary_pth in tqdm(all_cell_boundary_parquet_files, desc="Processing organoids"):
            organoids, _, _ = create_organoid_regions(
                cell_boundary_pth, 
                buffer_distance=5, 
                min_cell_count=20, 
                plot_results=False
            )
            for organoid in organoids: 
                organoid_mask = polygon_to_mask(
                    organoid, 
                    output_size=(224, 224), 
                    padding_ratio=0.1
                )
                self.organoid_masks.append(organoid_mask)
        
        self.organoid_masks = np.array(self.organoid_masks)
    
    def _save_masks(self):
        """Save processed masks to disk"""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'wb') as f:
            pickle.dump(self.organoid_masks, f)
        print(f"Saved {len(self.organoid_masks)} organoid masks to {self.save_path}")
    
    def __len__(self):
        return len(self.organoid_masks)
    
    def __getitem__(self, idx):
        # Convert numpy array to torch tensor and add channel dimension
        return torch.from_numpy(self.organoid_masks[idx]).float().unsqueeze(0)


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
                    rotated_batch[i] = TF.rotate(batch_X[i].unsqueeze(0), angles[i]).squeeze(0)
                
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
    dataset = NorkinOrganoidDataset()
    print(f"Loaded {len(dataset)} organoid masks")

    masks = dataset[:]
    morphological_features = get_morphological_features(masks[0])
    masks = masks.swapaxes(0, 1).repeat(1, 3, 1, 1)

    # Morph features
    morph_data_df = pd.DataFrame(morphological_features)
    morph_data_matrix = morph_data_df.values

    # Apply MinMaxScaler to scale each column to the range [0, 1]
    scaler = MinMaxScaler()
    morph_data_matrix = scaler.fit_transform(morph_data_matrix)
    morph_data_matrix.shape

    embeddings = get_resnet152_embeddings(masks, morphological_features=morph_data_matrix, fine_tune=True)
    with open("embeddings_resnet152.pkl", "wb") as f:
        pickle.dump(embeddings, f)