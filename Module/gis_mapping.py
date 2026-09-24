"""Geographic Information System (GIS) mapping and decision-support module.

Provides reusable utilities for multi-sensor change intensity derivation,
coordinate-to-grid rasterization, categorical error mapping, and GIS output auditing.
"""

import json,rasterio, numpy as np, pandas as pd
from pathlib import Path
from rasterio.transform import rowcol
from rasterio.warp import transform as warp_coords

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "config.json"


def load_config(config_path=None):
    """Load centralized project configuration from root config.json."""
    if config_path is None:
        config_path = CONFIG_PATH
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def robust_z(arr):
    """
    Compute robust Z-score standardization using median and median absolute deviation (MAD).
    """
    arr = arr.astype("float32")
    med = np.nanmedian(arr)
    mad = np.nanmedian(np.abs(arr - med))
    scale = 1.4826 * mad
    if scale == 0 or np.isnan(scale):
        scale = np.nanstd(arr)
        if scale == 0 or np.isnan(scale):
            scale = 1.0
    return np.abs(arr - med) / scale

def compute_robust_change_intensity(dVV, dVH, dNDVI, dNDWI):
    """
    Construct multi-sensor continuous change intensity surface from SAR and optical deltas.
    
    Args:
        dVV (np.ndarray): Sentinel-1 VV backscatter change.
        dVH (np.ndarray): Sentinel-1 VH backscatter change.
        dNDVI (np.ndarray): Sentinel-2 NDVI spectral change.
        dNDWI (np.ndarray): Sentinel-2 NDWI spectral change.
        
    Returns:
        np.ndarray: Continuous change intensity array scaled to [0, 1].
    """
    z_dvv = robust_z(dVV)
    z_dvh = robust_z(dVH)
    z_dndvi = robust_z(dNDVI)
    z_dndwi = robust_z(dNDWI)
    
    # Composite change magnitude (mean of robust absolute anomalies)
    change_raw = (z_dvv + z_dvh + z_dndvi + z_dndwi) / 4.0
    
    # Robust normalization using 2nd and 98th percentiles
    p2, p98 = np.nanpercentile(change_raw, [2, 98])
    if p98 > p2:
        change_intensity = np.clip((change_raw - p2) / (p98 - p2), 0.0, 1.0)
    else:
        change_intensity = np.zeros_like(change_raw)
        
    return change_intensity.astype("float32")

def tabular_to_raster(df, value_col, transform, crs, shape,
                      lon_col="longitude", lat_col="latitude",
                      fill_value=np.nan, dtype="float32"):
    """
    Map tabular coordinates (longitude, latitude in EPSG:4326) onto a master raster grid.
    
    Args:
        df (pd.DataFrame): Dataframe containing coordinates and values.
        value_col (str): Column to rasterize.
        transform (rasterio.Affine): Affine geotransform of target raster.
        crs (rasterio.CRS or str): Coordinate reference system of target raster.
        shape (tuple): (height, width) dimensions of target raster.
        lon_col (str): Column name for longitude.
        lat_col (str): Column name for latitude.
        fill_value (float): Background value for unmapped cells.
        dtype (str): Output array data type.
        
    Returns:
        np.ndarray: 2D raster array of specified shape and dtype.
    """
    height, width = shape
    raster = np.full((height, width), fill_value, dtype=dtype)
    
    # Coordinate transformation from EPSG:4326 to raster CRS
    lons = df[lon_col].to_numpy()
    lats = df[lat_col].to_numpy()
    crs_str = str(crs)
    if crs_str in ["EPSG:4326", "OGC:CRS84", "+init=epsg:4326"] or (hasattr(crs, "to_epsg") and crs.to_epsg() == 4326):
        x_coords, y_coords = lons, lats
    else:
        x_coords, y_coords = warp_coords("EPSG:4326", crs, lons, lats)
        x_coords = np.asarray(x_coords)
        y_coords = np.asarray(y_coords)
    
    rows, cols = rowcol(transform, x_coords, y_coords)
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    
    # Boundary guard to ensure all points reside inside raster dimensions
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    raster[rows[valid], cols[valid]] = df[value_col].to_numpy()[valid]
    
    return raster

def construct_spatial_error_raster(test_pred_df, transform, crs, shape,
                                   y_true_col="Y", y_pred_col="Y_pred",
                                   lon_col="longitude", lat_col="latitude"):
    """
    Generate categorical 4-class confusion/error raster on the held-out spatial test set:
      0 = True Negative (TN)  - Reference No-Change, Predicted No-Change (Gray)
      1 = False Positive (FP) - Reference No-Change, Predicted Change (Orange: False Alarm)
      2 = False Negative (FN) - Reference Change, Predicted No-Change (Light Blue: Missed Impact)
      3 = True Positive (TP)  - Reference Change, Predicted Change (Green: Confirmed Impact)
      
    Returns:
        np.ndarray: 2D float32 raster with 0, 1, 2, 3 in test regions and NaN elsewhere.
    """
    df_temp = test_pred_df.copy()
    y_true = df_temp[y_true_col].astype(int).to_numpy()
    y_pred = df_temp[y_pred_col].astype(int).to_numpy()
    
    error_class = np.select(
        [
            (y_true == 0) & (y_pred == 0),
            (y_true == 0) & (y_pred == 1),
            (y_true == 1) & (y_pred == 0),
            (y_true == 1) & (y_pred == 1)
        ],
        [0, 1, 2, 3],
        default=-1
    )
    
    df_temp["error_class"] = error_class
    
    return tabular_to_raster(
        df=df_temp,
        value_col="error_class",
        transform=transform,
        crs=crs,
        shape=shape,
        lon_col=lon_col,
        lat_col=lat_col,
        fill_value=np.nan,
        dtype="float32"
    )

def audit_gis_outputs(file_paths_dict, ref_profile=None):
    """
    Audit exported GIS GeoTIFF files for dimensions, CRS, affine transform, and validity.
    
    Args:
        file_paths_dict (dict): Dictionary mapping product descriptions to Path objects.
        ref_profile (dict, optional): Reference raster profile for alignment checks.
        
    Returns:
        pd.DataFrame: Audit summary table.
    """
    records = []
    for label, path in file_paths_dict.items():
        path = Path(path)
        exists = path.exists()
        size_mb = path.stat().st_size / (1024 ** 2) if exists else 0.0
        
        dims = "N/A"
        crs_str = "N/A"
        valid_px = 0
        aligned = False
        
        if exists:
            try:
                with rasterio.open(path) as src:
                    dims = f"{src.height} x {src.width}"
                    crs_str = str(src.crs)
                    data = src.read(1)
                    if np.issubdtype(data.dtype, np.floating):
                        valid_px = int(np.count_nonzero(~np.isnan(data)))
                    else:
                        valid_px = int(np.count_nonzero(data != src.nodata))
                        
                    if ref_profile is not None:
                        aligned = (
                            src.height == ref_profile.get("height") and
                            src.width == ref_profile.get("width") and
                            src.crs == ref_profile.get("crs")
                        )
            except Exception as e:
                dims = f"Error: {e}"
                
        records.append({
            "Product": label,
            "Filename": path.name,
            "Exists": exists,
            "Dimensions": dims,
            "CRS": crs_str,
            "Valid Pixels": f"{valid_px:,}" if exists else "0",
            "Aligned to Master": aligned if ref_profile is not None else "N/A",
            "Size (MB)": f"{size_mb:.2f}"
        })
        
    return pd.DataFrame(records)
