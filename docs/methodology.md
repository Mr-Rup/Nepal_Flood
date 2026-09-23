[← README](../README.md) | [🏠 Main README](../README.md) | [Data & Features →](data-and-features.md)

# Methodology

## Problem Formulation

The goal is **binary pixel-level landscape change detection** in Nepal's Bhote Koshi – Trishuli river basin following extreme monsoon flooding in August 2026. Each 30 m pixel is classified as:

- **Y = 0 (No Change):** Landscape remained stable between the pre-flood and post-flood observation windows.
- **Y = 1 (Change):** Significant flood or landslide impact detected.

> [!IMPORTANT]
> The target label `Y` is **not independent ground truth**. It is an EO-derived proxy — a composite anomaly score computed from standardized change features (`dVV`, `dVH`, `dNDVI`, `dNDWI`). Pixels in the top 20th percentile of the anomaly score are flagged as change (Y = 1). The model learns to replicate this EO-based rule across spatially separated regions, not to detect floods from first principles.

### Why a Proxy Label?

In rapid disaster contexts, georeferenced field labels arrive too slowly for operational mapping. Copernicus EMS (EMSR927) rapid mapping was queried as external event context (confirming the activation exists), but its vector products are not used as training labels. Instead, a multi-modal standardized anomaly score combines radar and optical change signals:

1. Z-score normalize each change feature: `Z = (X - mean) / std`
2. Composite anomaly magnitude: `change_score = mean(|Z|)` across `dVV`, `dVH`, `dNDVI`, `dNDWI`
3. Binary threshold at 80th percentile: `Y = 1` if `change_score >= threshold`

This is implemented in [`Module/data_prep.py → generate_proxy_labels()`](../Module/data_prep.py).

---

## Cost of Errors

In a post-disaster mapping context:

| Error Type | Meaning | Real-World Impact |
|---|---|---|
| **False Positive (FP)** | Model flags change where none occurred | Over-allocation of relief resources; investigation teams dispatched unnecessarily |
| **False Negative (FN)** | Model misses actual landscape change | Affected areas go unserved; delayed rescue or relief |

**False negatives are more dangerous** — a missed landslide or inundation zone can mean lives lost. The primary metric reflecting this asymmetry is **Recall** (sensitivity to actual change). F1 and IoU (Jaccard) balance precision and recall, while ROC-AUC measures overall ranking quality.

The Random Forest uses `class_weight='balanced'` to compensate for the 80/20 class imbalance and upweight the minority (change) class during training.

---

## Model Choice

Two models were compared using identical 4-Fold **Spatial GroupKFold Cross-Validation** on the 80% development set:

### Logistic Regression (Linear Baseline)
- Standardized via `StandardScaler` inside a `Pipeline` (fitted strictly per-fold on training data only).
- `max_iter=1000`, `class_weight='balanced'`, `random_state=42`.
- Provides an interpretable linear decision boundary. Serves as a sanity baseline.

### Random Forest (Nonlinear Ensemble — Selected)
- `n_estimators=300`, `min_samples_leaf=2`, `class_weight='balanced'`, `random_state=42`, `n_jobs=-1`.
- No scaling required (tree-based models are invariant to feature magnitude).
- Captures complex cross-sensor feature interactions (e.g., joint SAR + optical + terrain patterns) that a linear model cannot.

**Why Random Forest won:** Across all 4 spatial folds, Random Forest achieved F1 of **0.9479 ± 0.0048** vs. Logistic Regression's **0.4823 ± 0.0280**. The RF's low fold-to-fold variance (std = 0.0048 on F1) demonstrates consistent geographic generalization across different valley blocks.

> [!NOTE]
> All hyperparameters are centralized in [`configs/ml_config.json`](../configs/ml_config.json). No hyperparameter search was conducted — the configuration is fixed from the start.

---

## Spatial Block Cross-Validation Design

### Why Not Random Splits?

In spatial EO datasets, neighboring pixels exhibit strong **spatial autocorrelation** (Tobler's First Law of Geography). A naive random pixel-level train/test split places pixels from the same landscape patch into both sets, causing severe **optimistic bias** — the model memorizes spatial neighborhoods rather than learning generalizable patterns.

### Block Construction

Pixels are aggregated into discrete regular grid cells:

$$\text{lon\_block} = \left\lfloor \frac{\text{longitude}}{\text{BLOCK\_SIZE}} \right\rfloor, \quad \text{lat\_block} = \left\lfloor \frac{\text{latitude}}{\text{BLOCK\_SIZE}} \right\rfloor$$

$$\text{spatial\_block} = \text{lon\_block}\_\text{lat\_block}$$

**Block size = 0.02° (~2.2 km)** was selected after evaluating candidates:

| Block Size | Approx. km | Num Blocks | Median Pixels/Block | Class 1 Std |
|---|---|---|---|---|
| 0.01° | 1.11 km | 100 | 1,338 | 0.176 |
| **0.02°** | **2.22 km** | **26** | **5,025** | **0.134** |
| 0.03° | 3.33 km | 18 | 5,249 | 0.119 |

0.02° gives 26 blocks — enough for stable GroupShuffleSplit and 4-fold GroupKFold while keeping blocks large enough to break spatial autocorrelation.

### Partition Strategy (Two-Stage GroupShuffleSplit)

1. **Stage 1:** Split 26 blocks → Development (80% ≈ 20 blocks) + Test (20% ≈ 6 blocks)
2. **Stage 2:** Split Dev blocks → Train (75% of dev ≈ 15 blocks) + Validation (25% of dev ≈ 5 blocks)

Resulting split:

| Split | Rows | Share | Blocks | Y=1 Ratio |
|---|---|---|---|---|
| Train | 72,437 | 58.8% | 15 | 17.5% |
| Validation | 21,366 | 17.3% | 5 | 27.1% |
| Test | 29,351 | 23.8% | 6 | 21.0% |

**Zero spatial block overlap** between any pair of splits was programmatically verified via [`Module/spatial_ml.py → verify_split_leakage()`](../Module/spatial_ml.py).

### Cross-Validation on Development Set

4-Fold **GroupKFold** on the 20 development blocks ensures entire blocks are held out per fold. Both Logistic Regression and Random Forest were evaluated on the **exact same fold partitions**.

### Model Freeze Protocol

After cross-validation confirmed Random Forest superiority, the model architecture, features, and hyperparameters were **frozen**. The frozen RF was then fit on the **complete** 80% development set (93,803 rows, 20 blocks) and evaluated **once** on the held-out spatial test set — a true one-shot generalization estimate.

---

## Preprocessing

- **StandardScaler**: Applied only for Logistic Regression, fitted strictly on training fold data. Validation/test data transformed using train-fitted parameters.
- **Random Forest**: No scaling applied (trees are scale-invariant).
- **No imputation needed**: The feature table construction in [`Module/data_prep.py → build_feature_table()`](../Module/data_prep.py) applies a joint finite-pixel mask across all rasters — only pixels with valid (non-NaN) values across all 17 features are retained.

> [!WARNING]
> **Target leakage check**: The code explicitly asserts that `Y`, `change_score`, and `rule_based_score` are **never** included in the predictor feature set. This is verified in Notebook 02 before any model fitting.

---

[← README](../README.md) | [🏠 Main README](../README.md) | [Data & Features →](data-and-features.md)
