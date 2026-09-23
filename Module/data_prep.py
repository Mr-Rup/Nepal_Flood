import json, ee, geemap, rasterio
from pathlib import Path
import numpy as np, pandas as pd
from rasterio.warp import reproject, Resampling, calculate_default_transform

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
DATA_DIR = PROJECT_ROOT / "Data"
CONFIGS_DIR = PROJECT_ROOT / "configs"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Default AOI: Bhote Koshi - Trishuli region
DEFAULT_AOI_COORDS = [84.70, 28.00, 85.00, 28.25]

def load_time_metadata(json_path=None):
    """Load time window definitions from configs/time_metadata.json."""
    if json_path is None:
        json_path = CONFIGS_DIR / "time_metadata.json"
    json_path = Path(json_path)

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "BASELINE_START": "2025-08-01",
        "BASELINE_END": "2025-08-31",
        "PRE_START": "2026-08-01",
        "PRE_END": "2026-08-25",
        "POST_START": "2026-08-27",
        "POST_END": "2026-09-05",
        "RECOVERY_START": "2026-09-06",
        "RECOVERY_END": "2026-09-30"
    }

# ---------------------------------------------------------------------------
# Earth Engine Utilities
# ---------------------------------------------------------------------------
def init_earth_engine(project="astrotourism-darksky", aoi_coords=None):
    """Authenticate and initialize Google Earth Engine."""
    try:
        ee.Initialize(project=project)
        print(f"Earth Engine initialized successfully (Project: {project}).")
    except Exception as e:
        print(f"Initial connection notice: {e}. Attempting authentication...")
        try:
            ee.Authenticate()
            ee.Initialize(project=project)
            print("Earth Engine authenticated and initialized successfully.")
        except Exception as auth_err:
            print(f"Failed to initialize Earth Engine: {auth_err}")
            raise auth_err

    if aoi_coords is None:
        aoi_coords = DEFAULT_AOI_COORDS
    aoi = ee.Geometry.Rectangle(aoi_coords)
    return aoi

def get_temporal_composites(collection, aoi, bands=None, time_metadata=None):
    """
    Filter image collection into Baseline, Pre, Post, and Recovery temporal subsets,
    and compute median composite clipped to AOI.
    
    Returns:
        dict: {
            "baseline": ee.Image,
            "pre": ee.Image,
            "post": ee.Image,
            "recovery": ee.Image,
            "counts": {"baseline": int, "pre": int, "post": int, "recovery": int}
        }
    """
    if time_metadata is None:
        time_metadata = load_time_metadata()

    b_start, b_end = time_metadata["BASELINE_START"], time_metadata["BASELINE_END"]
    p_start, p_end = time_metadata["PRE_START"], time_metadata["PRE_END"]
    po_start, po_end = time_metadata["POST_START"], time_metadata["POST_END"]
    r_start, r_end = time_metadata["RECOVERY_START"], time_metadata["RECOVERY_END"]

    col_baseline = collection.filterDate(b_start, b_end)
    col_pre = collection.filterDate(p_start, p_end)
    col_post = collection.filterDate(po_start, po_end)
    col_recovery = collection.filterDate(r_start, r_end)

    counts = {
        "baseline": col_baseline.size().getInfo(),
        "pre": col_pre.size().getInfo(),
        "post": col_post.size().getInfo(),
        "recovery": col_recovery.size().getInfo()
    }

    print("--- Temporal Composite Scene Counts ---")
    print(f"Baseline ({b_start} to {b_end}): {counts['baseline']} scenes")
    print(f"Pre-event ({p_start} to {p_end}): {counts['pre']} scenes")
    print(f"Post-event ({po_start} to {po_end}): {counts['post']} scenes")
    print(f"Recovery ({r_start} to {r_end}): {counts['recovery']} scenes")

    def _prep_composite(col):
        if bands:
            col = col.select(bands)
        return col.median().clip(aoi)

    return {
        "baseline": _prep_composite(col_baseline),
        "pre": _prep_composite(col_pre),
        "post": _prep_composite(col_post),
        "recovery": _prep_composite(col_recovery),
        "counts": counts
    }


def calculate_optical_indices(image):
    """
    Calculate NDVI, NDWI (McFeeters), and NDBI from a Sentinel-2 surface reflectance composite.
    Expected bands in image: B3 (Green), B4 (Red), B8 (NIR), B11 (SWIR1).
    """
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    ndbi = image.normalizedDifference(["B11", "B8"]).rename("NDBI")
    return ee.Image.cat([ndvi, ndwi, ndbi])


def export_geotiff(image, filename, aoi, scale=30):
    """Export an Earth Engine Image as a local GeoTIFF file via geemap."""
    target_path = Path(filename)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    geemap.ee_export_image(
        image,
        filename=str(target_path),
        scale=scale,
        region=aoi,
        file_per_band=False
    )
    if target_path.exists():
        size_mb = target_path.stat().st_size / (1024 ** 2)
        print(f"Downloaded: {target_path.name} ({size_mb:.2f} MB)")
    else:
        print(f"Download check failed: {target_path.name} not found.")

# ---------------------------------------------------------------------------
# Local Raster Reprojection & Alignment
# ---------------------------------------------------------------------------
def align_band(src_path, ref_profile, band=1, resampling=Resampling.bilinear):
    """
    Reproject and align a single band from src_path to match ref_profile geometry.
    """
    dst_shape = (ref_profile["height"], ref_profile["width"])
    dst_transform = ref_profile["transform"]
    dst_crs = ref_profile["crs"]

    with rasterio.open(src_path) as src:
        out = np.full(dst_shape, np.nan, dtype="float32")
        reproject(
            source=src.read(band).astype("float32"),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=resampling
        )
    return out

def calculate_metric_slope(dem_path, ref_profile, utm_crs="EPSG:32645", resolution=30):
    """
    Reproject DEM to a metric UTM system (default EPSG:32645 for Nepal UTM 45N),
    compute spatial elevation gradients and slope in degrees, then reproject
    back to the master reference grid.
    """
    with rasterio.open(dem_path) as src:
        transform_utm, width_utm, height_utm = calculate_default_transform(
            src.crs, utm_crs, src.width, src.height, *src.bounds, resolution=resolution
        )
        dem_utm = np.full((height_utm, width_utm), np.nan, dtype="float32")
        reproject(
            source=src.read(1).astype("float32"),
            destination=dem_utm,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=transform_utm,
            dst_crs=utm_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear
        )

    # Compute metric gradient in metres
    dy, dx = np.gradient(dem_utm, resolution, resolution)
    slope_utm = np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))

    # Reproject slope back to master reference grid
    slope = np.full((ref_profile["height"], ref_profile["width"]), np.nan, dtype="float32")
    reproject(
        source=slope_utm.astype("float32"),
        destination=slope,
        src_transform=transform_utm,
        src_crs=utm_crs,
        dst_transform=ref_profile["transform"],
        dst_crs=ref_profile["crs"],
        dst_nodata=np.nan,
        resampling=Resampling.bilinear
    )
    return slope

def create_coordinate_grids(transform, shape):
    """Generate 2D arrays of longitude and latitude pixel centroids."""
    height, width = shape
    row_grid, col_grid = np.indices((height, width))
    lon_grid = transform.c + col_grid * transform.a + row_grid * transform.b
    lat_grid = transform.f + col_grid * transform.d + row_grid * transform.e
    return lon_grid.astype("float32"), lat_grid.astype("float32")

def build_feature_table(arrays_dict, ref_transform):
    """
    Extract finite pixels across all rasters, generate coordinates, and assemble
    into a clean pandas DataFrame.
    """
    sample_key = next(iter(arrays_dict))
    grid_shape = arrays_dict[sample_key].shape

    # Joint finite mask across all feature rasters
    valid_mask = np.ones(grid_shape, dtype=bool)
    for name, arr in arrays_dict.items():
        valid_mask &= np.isfinite(arr)

    print(f"Total raster pixels: {valid_mask.size:,}")
    print(f"Valid pixels: {int(valid_mask.sum()):,} ({valid_mask.mean() * 100:.2f}%)")

    # Generate coordinate grids
    lon_grid, lat_grid = create_coordinate_grids(ref_transform, grid_shape)

    columns = ["longitude", "latitude"] + list(arrays_dict.keys())
    flat_data = [lon_grid[valid_mask], lat_grid[valid_mask]] + [
        arrays_dict[k][valid_mask] for k in arrays_dict
    ]

    feature_matrix = np.column_stack(flat_data)
    df = pd.DataFrame(feature_matrix, columns=columns)
    return df, valid_mask

# ---------------------------------------------------------------------------
# Labeling Utility (Proxy Anomaly-Based Target)
# ---------------------------------------------------------------------------
def generate_proxy_labels(df, change_vars=("dVV", "dVH", "dNDVI", "dNDWI"), quantile=0.80):
    """
    Generate multi-modal anomaly/change score and binary proxy target Y.
    
    1. Standardizes change features: Z = (X - mean) / std
    2. Computes composite anomaly magnitude: change_score = mean(|Z|)
    3. Flags top (1 - quantile)*100% as significant event impact (Y = 1).
    """
    df_out = df.copy()
    changes = df_out[list(change_vars)].copy()

    for col in changes.columns:
        col_std = changes[col].std()
        changes[col] = (changes[col] - changes[col].mean()) / (col_std if col_std != 0 else 1.0)

    df_out["change_score"] = changes.abs().mean(axis=1)
    threshold = df_out["change_score"].quantile(quantile)
    df_out["Y"] = (df_out["change_score"] >= threshold).astype(int)

    print(f"Proxy labeling completed using {list(change_vars)}.")
    print(f"Score threshold ({quantile*100:.0f}th percentile): {threshold:.4f}")
    print(f"Class distribution: {dict(df_out['Y'].value_counts())}")
    return df_out
