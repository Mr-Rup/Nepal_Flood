[← Methodology](methodology.md) | [🏠 Main README](../README.md) | [Results →](results.md)

# Data & Features

## Data Sources

| Source | Product | Resolution | Usage |
|---|---|---|---|
| **Sentinel-1 SAR** | `COPERNICUS/S1_GRD` (IW mode, VV+VH dual-pol) | 30 m | Radar backscatter composites, SAR change detection |
| **Sentinel-2 Optical** | `COPERNICUS/S2_SR_HARMONIZED` (Level-2A SR) | 30 m (resampled) | Spectral indices (NDVI, NDWI, NDBI), optical change detection |
| **SRTM DEM** | `USGS/SRTMGL1_003` | 30 m | Elevation, terrain slope (computed via UTM projection) |
| **Copernicus EMS** | EMSR927 Rapid Mapping (queried via API) | — | External event context only (not used as labels) |

All data is accessed via **Google Earth Engine** and exported as local GeoTIFFs. See [`Module/data_prep.py`](../Module/data_prep.py) for all extraction and export functions.

> [!NOTE]
> Sentinel-1 SAR imagery is cloud-penetrating and operates day-and-night — critical during monsoon season when optical imagery is frequently obscured.

---

## Area of Interest (AOI)

**Bhote Koshi – Trishuli river basin, central Nepal**

- Bounding box: `[84.70°E, 28.00°N, 85.00°E, 28.25°N]`
- Approximate dimensions: ~33 km × ~28 km
- Terrain: Steep Himalayan valleys (elevation range: 604 m to 4,443 m)
- Defined in [`Module/data_prep.py → DEFAULT_AOI_COORDS`](../Module/data_prep.py)

---

## Temporal Windows

All temporal definitions are centralized in [`config.json → TIME_WINDOWS`](../config.json):

| Window | Start | End | Purpose |
|---|---|---|---|
| **Baseline** | 2025-08-01 | 2025-08-31 | Previous year reference (same season, pre-event) |
| **Pre-Flood** | 2026-08-01 | 2026-08-25 | Immediately before the flood event |
| **Post-Flood** | 2026-08-27 | 2026-09-05 | During/after the flood event |
| **Recovery** | 2026-09-06 | 2026-09-30 | Post-event recovery phase |

Temporal composites are **median** aggregations clipped to the AOI, implemented in [`Module/data_prep.py → get_temporal_composites()`](../Module/data_prep.py).

---

## Feature Engineering

### 17 Predictive Features

All features are defined in [`config.json → FEATURE_GROUPS`](../config.json):

#### Sentinel-1 SAR Features (6)

| Feature | Description |
|---|---|
| `VV_pre` | VV-polarization backscatter, pre-flood median composite |
| `VH_pre` | VH-polarization backscatter, pre-flood median composite |
| `VV_post` | VV-polarization backscatter, post-flood median composite |
| `VH_post` | VH-polarization backscatter, post-flood median composite |
| `dVV` | VV change: `VV_post − VV_pre` |
| `dVH` | VH change: `VH_post − VH_pre` |

> [!TIP]
> SAR backscatter typically **drops** over inundated or debris-covered areas (negative dVV/dVH), making these features strong direct indicators of flood impact.

#### Sentinel-2 Optical Features (9)

| Feature | Description | Formula |
|---|---|---|
| `NDVI_pre` | Normalized Difference Vegetation Index, pre-flood | `(B8 − B4) / (B8 + B4)` |
| `NDWI_pre` | Normalized Difference Water Index (McFeeters), pre-flood | `(B3 − B8) / (B3 + B8)` |
| `NDBI_pre` | Normalized Difference Built-up Index, pre-flood | `(B11 − B8) / (B11 + B8)` |
| `NDVI_post` | NDVI, post-flood | Same formula |
| `NDWI_post` | NDWI, post-flood | Same formula |
| `NDBI_post` | NDBI, post-flood | Same formula |
| `dNDVI` | Vegetation change: `NDVI_post − NDVI_pre` | — |
| `dNDWI` | Water change: `NDWI_post − NDWI_pre` | — |
| `dNDBI` | Built-up change: `NDBI_post − NDBI_pre` | — |

Index calculation is implemented in [`Module/data_prep.py → calculate_optical_indices()`](../Module/data_prep.py).

> [!WARNING]
> Sentinel-2 uses a `CLOUDY_PIXEL_PERCENTAGE < 80` filter, which is lenient during monsoon season. Cloud contamination can degrade optical index reliability. SAR features are unaffected by clouds.

#### Terrain Features (2)

| Feature | Description |
|---|---|
| `DEM` | SRTM 30m elevation (meters above sea level) |
| `slope` | Terrain slope in degrees, computed via metric UTM projection |

Slope is computed by reprojecting the DEM to **EPSG:32645** (UTM Zone 45N, covering Nepal), calculating spatial gradients in meters, and reprojecting the resulting slope raster back to the master grid. This avoids distortion from computing gradients directly in geographic degrees.

Implementation: [`Module/data_prep.py → calculate_metric_slope()`](../Module/data_prep.py).

---

## Spatial Alignment

All rasters are aligned to a **master reference grid** defined by `S1_pre.tif`:

- CRS: EPSG:4326 (geographic)
- Pixel size: ~30 m (~0.00027°)
- Resampling: Bilinear interpolation

The 9-band Sentinel-2 index raster and the DEM are reprojected and resampled to match this master grid pixel-for-pixel.

Implementation: [`Module/data_prep.py → align_band()`](../Module/data_prep.py).

---

## Feature Table Construction

The feature table is assembled in [`Module/data_prep.py → build_feature_table()`](../Module/data_prep.py):

1. A **joint finite-pixel mask** is computed across all 17 feature rasters — only pixels with valid (non-NaN) values in every band are retained.
2. Coordinate grids (`longitude`, `latitude`) are generated from the affine transform of the master reference.
3. All arrays are stacked into a pandas DataFrame.

### Dataset Statistics

| Property | Value | Source |
|---|---|---|
| Total valid pixels | 123,154 | Notebook 02 output |
| Columns | 20 (2 coords + 17 features + 1 target) | — |
| Missing values | 0 | Data audit in Notebook 02 |
| Exact duplicate rows | 0 | Data audit in Notebook 02 |
| Y = 0 (No Change) | 98,523 (80.0%) | — |
| Y = 1 (Change) | 24,631 (20.0%) | — |
| Longitude range | 84.9614° – 84.9999° | — |
| Latitude range | 28.0002° – 28.2498° | — |
| DEM range | 604 m – 4,443 m | — |
| Slope range | 0.6° – 74.7° | — |

---

## Output Files
 
| File | Size | Description |
|---|---|---|
| [`Data/tables/EO_feature_table_unlabeled.csv`](../Data/tables/EO_feature_table_unlabeled.csv) | ~25 MB | 17 features + coordinates, no target |
| [`Data/tables/EO_feature_table_labelled.csv`](../Data/tables/EO_feature_table_labelled.csv) | ~26 MB | With proxy target `Y` |
| [`Data/tables/EO_feature_table_spatial_split.csv`](../Data/tables/EO_feature_table_spatial_split.csv) | ~29 MB | With spatial block IDs and split labels |
| [`Data/tables/RF_spatial_test_predictions.csv`](../Data/tables/RF_spatial_test_predictions.csv) | ~1.1 MB | GIS-ready: `longitude, latitude, Y, Y_pred, P_change` |
| [`Data/tables/ablation_results.csv`](../Data/tables/ablation_results.csv) | <1 KB | Multi-sensor ablation metrics |
| [`Data/models/random_forest_final.joblib`](../Data/models/random_forest_final.joblib) | ~90 MB | Serialized frozen RF model |
| [`Data/models/random_forest_final.json`](../Data/models/random_forest_final.json) | <1 KB | Model metadata (features, hyperparameters, timestamp) |

---

[← Methodology](methodology.md) | [🏠 Main README](../README.md) | [Results →](results.md)
