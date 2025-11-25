from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import transform, polygonize, unary_union
from shapely.validation import make_valid
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from tqdm import tqdm
import geopandas as gpd
import colorcet as cc


def get_colors(gdf, col_key, cmap=cc.cm.glasbey_light.colors):
    map_colors = {}
    i = 0
    for label in gdf[col_key].unique():
        map_colors[label] = cmap[i]
        i += 1
        if i == len(cmap):
            i = 0
    colors = [map_colors[v] for v in gdf[col_key]]
    return colors


def create_sample_gdf_with_hole():
    polygons = []
    clusters = []
    size = 10
    for i in range(size):
        for j in range(size):
            if 4 <= i <= 5 and 4 <= j <= 5:
                continue
            polygons.append(Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]))
            clusters.append(1)
    gdf = gpd.GeoDataFrame({"cluster_label": clusters, "geometry": polygons})
    gdf.crs = "EPSG:3857"
    return gdf


def to_2d(geom):
    if geom.is_empty:
        return geom
    return transform(lambda x, y, z=None: (x, y), geom)


def make_valid_and_polygonize(geom):
    """
    Strong repair: make_valid → buffer(0) → polygonize → MultiPolygon
    Ensures the final geometry is usable for unary_union().
    """

    if geom is None or geom.is_empty:
        return geom

    # Step 1 — GEOS repair
    geom = make_valid(geom)

    # Step 2 — soften cracks and micro-slivers
    geom = geom.buffer(0)

    # Step 3 — polygonize boundaries (this fixes *everything*)
    polys = list(polygonize(geom.boundary))

    if len(polys) == 0:
        return None
    elif len(polys) == 1:
        return polys[0]
    else:
        return MultiPolygon(polys)


def get_outer_boundary(geom, ignore_holes=True, boundary_buffer=0.0):
    """
    Extracts boundaries from a geometry, with a choice to include holes.

    - ignore_holes=True (Default): Uses a robust polygonize->union method to
      find the single, continuous outer boundary, "paving over" all holes and
      internal complexity. This is ideal for defining the outermost perimeter.

    - ignore_holes=False: Returns all boundary lines, including the exterior
      shell AND the boundaries of any interior holes.
    """
    status = 0
    if geom is None or geom.is_empty:
        status = 1
        return None, status

    # --- Step 1: Cleaning (applies to both modes) ---
    # A buffer(0) or morphological closing helps resolve topological issues.
    if boundary_buffer > 0:
        geom = geom.buffer(boundary_buffer, join_style=1).buffer(-boundary_buffer, join_style=1)

    # Final cleanup and validation
    geom = geom.buffer(0)
    geom = make_valid(geom)

    if geom.is_empty:
        status = 1
        return None, status

    # --- Step 2: Boundary Extraction based on user choice ---
    if ignore_holes:
        try:
            # Fill all holes and return the single outer boundary.
            polygons = list(polygonize(geom.boundary))
            if not polygons:
                status = 1
                return None, status
            boundary = unary_union(polygons).boundary
            return boundary, status

        except Exception as e:
            status = 2
            print(f"Warning: Robust boundary extraction failed with error: {e}. Falling back to union of exteriors.")
            if hasattr(geom, "geoms"):  # MultiPolygon
                boundary = unary_union([Polygon(p.exterior) for p in geom.geoms]).boundary
                return boundary, status
            else:  # Polygon
                boundary = Polygon(geom.exterior).boundary
                return boundary, status
    else:
        # Return all boundaries, including holes.
        return geom.boundary, status


def classify_polygons_by_boundary_layers(
    gdf: gpd.GeoDataFrame,
    cluster_col: str,
    max_layer: int = 3,
    use_xy_only: bool = False,
    ignore_holes: bool = False,
    boundary_buffer: float = 0.0,
    return_distance_to_boundary: bool = False,
):
    gdf = gdf.copy()
    gdf["location"] = "interior"
    gdf["geometry"] = gdf["geometry"].make_valid()

    # 2D-only mode
    if use_xy_only:
        gdf["geometry"] = gdf.geometry.apply(to_2d)

    sindex = gdf.sindex
    clusters = gdf.groupby(cluster_col, observed=False)
    total_clusters = gdf[cluster_col].nunique()

    # dissolve per cluster
    dissolved = {}
    boundaries = {}
    statuses = {}
    for cid, cluster_gdf in tqdm(clusters, total=total_clusters, desc="Merging geoms and finding boundaries"):
        dissolved[cid] = cluster_gdf.geometry.union_all()
        boundaries[cid], statuses[cid] = get_outer_boundary(
            dissolved[cid], ignore_holes=ignore_holes, boundary_buffer=boundary_buffer
        )

    # classify
    for cid, cluster_gdf in tqdm(clusters, total=total_clusters, desc="Classifying clusters into layers from boundary"):
        boundary = boundaries[cid]
        if boundary is None:
            continue

        # boundary_0 detection
        cand_idx = sindex.query(boundary, predicate="intersects")
        cands = gdf.iloc[cand_idx]
        cands = cands[cands[cluster_col] == cid]

        # strict outer-boundary touching
        boundary_0_ids = cands[cands.touches(boundary)].index.tolist()

        if boundary_0_ids:
            gdf.loc[boundary_0_ids, "location"] = "boundary_0"

        # BFS expansion
        visited = set(boundary_0_ids)
        frontier = boundary_0_ids

        for layer in range(1, max_layer + 1):
            if not frontier:
                break

            front_geom = gdf.loc[frontier].geometry.union_all()

            idx = sindex.query(front_geom, predicate="intersects")
            neigh = gdf.iloc[idx]
            neigh = neigh[neigh[cluster_col] == cid]
            neigh = neigh[neigh.intersects(front_geom)]

            new = list(set(neigh.index) - visited)
            if not new:
                break

            gdf.loc[new, "location"] = f"boundary_{layer}"
            visited.update(new)
            frontier = new

    if return_distance_to_boundary:
        gdf["distance_to_boundary"] = float("nan")

        # Group by cluster and assign distances directly using .loc
        for cid, group in tqdm(clusters, total=total_clusters, desc="Calculating distances to boundary"):
            boundary = boundaries.get(cid)
            if boundary and not boundary.is_empty:
                gdf.loc[group.index, "distance_to_boundary"] = group.geometry.distance(boundary)

    return gdf, boundaries, dissolved, statuses


def plot_cluster_layers(
    gdf,
    cluster_id=None,
    cluster_col="cluster",
    boundaries=None,
    layers_to_plot=None,
    title=None,
    figsize=(7, 7),
):
    """
    Plot selected boundary layers + geometries with a clean, correct legend.
    - layers_to_plot: ["boundary_0","boundary_1","interior"] or None
    """

    # ---- Select cluster ----
    if cluster_id is not None:
        gdf_ = gdf[gdf[cluster_col] == cluster_id]
    else:
        gdf_ = gdf

    # ---- Select layers ----
    if layers_to_plot is None:
        layers_to_plot = sorted(gdf_["location"].unique())

    gdf_ = gdf_[gdf_["location"].isin(layers_to_plot)]

    # ---- Color map ----
    cmap = plt.colormaps.get_cmap("tab20")
    layer_colors = {layer: cmap(i) for i, layer in enumerate(layers_to_plot)}

    fig, ax = plt.subplots(figsize=figsize)

    # ------------------------------------------------------------
    # PLOT POLYGONS
    # ------------------------------------------------------------
    for layer, color in layer_colors.items():
        subset = gdf_[gdf_["location"] == layer]
        if not subset.empty:
            subset.plot(ax=ax, color=color, edgecolor="black", linewidth=0.01, aspect=1)

    # ------------------------------------------------------------
    # PLOT BOUNDARY (ALL PARTS)
    # ------------------------------------------------------------
    boundary_obj = None
    if boundaries and cluster_id in boundaries:
        boundary_obj = boundaries[cluster_id]

    if boundary_obj is not None:
        if hasattr(boundary_obj, "geoms"):  # MultiLineString
            for part in boundary_obj.geoms:
                ax.plot(*part.xy, color="red", linewidth=2)
        else:
            ax.plot(*boundary_obj.xy, color="red", linewidth=2)

    # ------------------------------------------------------------
    # MANUAL LEGEND
    # ------------------------------------------------------------
    handles = []
    labels = []

    # polygon layers
    for layer, color in layer_colors.items():
        handles.append(mpatches.Patch(color=color))
        labels.append(layer)

    # boundary (only once)
    if boundary_obj is not None:
        handles.append(Line2D([0], [0], color="red", lw=2))
        labels.append("boundary")

    ax.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc="upper left")

    # ------------------------------------------------------------
    ax.set_title(title or f"Cluster {cluster_id} — selected layers")
    plt.tight_layout()

    return fig, ax
