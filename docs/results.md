[← Data & Features](data-and-features.md) | [🏠 Main README](../README.md)

# Results

## Spatial Cross-Validation (4-Fold GroupKFold — Dev Set)

Both models were evaluated on the **exact same 4 spatial fold partitions** across the 80% development set (20 blocks, 93,803 pixels).

### Logistic Regression (with StandardScaler Pipeline)

| Fold | Train Blocks | Val Blocks | Val Rows | Accuracy | Precision | Recall | F1 | IoU | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| Fold 1 | 15 | 5 | 24,024 | 0.7940 | 0.3552 | 0.5808 | 0.4408 | 0.2827 | 0.8002 |
| Fold 2 | 15 | 5 | 22,870 | 0.6149 | 0.3468 | 0.8928 | 0.4995 | 0.3329 | 0.7451 |
| Fold 3 | 15 | 5 | 22,817 | 0.7021 | 0.4139 | 0.6018 | 0.4905 | 0.3249 | 0.7559 |
| Fold 4 | 15 | 5 | 24,092 | 0.7373 | 0.4003 | 0.6606 | 0.4985 | 0.3320 | 0.7732 |
| **Mean ± Std** | — | — | — | **0.7121 ± 0.0750** | **0.3790 ± 0.0330** | **0.6840 ± 0.1432** | **0.4823 ± 0.0280** | **0.3181 ± 0.0239** | **0.7686 ± 0.0240** |

### Random Forest

| Fold | Train Blocks | Val Blocks | Val Rows | Accuracy | Precision | Recall | F1 | IoU | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| Fold 1 | 15 | 5 | 24,024 | 0.9831 | 0.9137 | 0.9711 | 0.9416 | 0.8896 | 0.9984 |
| Fold 2 | 15 | 5 | 22,870 | 0.9781 | 0.9247 | 0.9781 | 0.9507 | 0.9059 | 0.9983 |
| Fold 3 | 15 | 5 | 22,817 | 0.9767 | 0.9275 | 0.9788 | 0.9525 | 0.9093 | 0.9983 |
| Fold 4 | 15 | 5 | 24,092 | 0.9788 | 0.9378 | 0.9561 | 0.9469 | 0.8991 | 0.9977 |
| **Mean ± Std** | — | — | — | **0.9792 ± 0.0028** | **0.9259 ± 0.0099** | **0.9710 ± 0.0105** | **0.9479 ± 0.0048** | **0.9010 ± 0.0087** | **0.9981 ± 0.0003** |

> [!IMPORTANT]
> Random Forest dominates on every metric. Critically, its fold-to-fold standard deviation is tiny (F1 std = 0.0048), indicating **robust geographic generalization** — the model performs consistently regardless of which valley blocks are held out.

---

## One-Shot Spatial Test Evaluation

The frozen Random Forest (trained on all 20 dev blocks) was evaluated **once** on the 6 held-out test blocks (29,351 pixels):

| Metric | Value |
|---|---|
| Accuracy | **0.9844** |
| Precision | **0.9411** |
| Recall | **0.9873** |
| F1 | **0.9637** |
| IoU (Jaccard) | **0.9299** |
| ROC-AUC | **0.9991** |

### Confusion Matrix (Spatial Test)

|  | Predicted No Change (0) | Predicted Change (1) |
|---|---|---|
| **Actual No Change (0)** | 22,820 | 381 |
| **Actual Change (1)** | 78 | 6,072 |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| No Change (0) | 0.9966 | 0.9836 | 0.9901 | 23,201 |
| Change (1) | 0.9411 | 0.9873 | 0.9637 | 6,150 |
| **Accuracy** | — | — | **0.9844** | 29,351 |
| **Macro Avg** | 0.9688 | 0.9855 | 0.9769 | 29,351 |
| **Weighted Avg** | 0.9850 | 0.9844 | 0.9845 | 29,351 |

**Key observation**: Recall for the change class (0.9873) is higher than precision (0.9411) — the model slightly over-detects change (381 false positives vs. only 78 false negatives). This is the preferred error direction in disaster response, where missing an affected area is more costly than a false alarm.

---

## Multi-Sensor Ablation Study

The ablation study quantifies the contribution of each sensor modality by training separate Random Forest models (same hyperparameters) on feature subsets and evaluating on the **exact same** held-out spatial test partition:

| Experiment | Num Features | Accuracy | Precision | Recall | F1 | IoU | ROC-AUC |
|---|---|---|---|---|---|---|---|
| S1 SAR Only | 6 | 0.8000 | 0.5167 | 0.6902 | 0.5910 | 0.4195 | 0.8497 |
| S1 + S2 (Optical) | 15 | 0.9827 | 0.9352 | 0.9857 | 0.9598 | 0.9227 | 0.9990 |
| **S1 + S2 + Terrain (Full)** | **17** | **0.9844** | **0.9411** | **0.9873** | **0.9637** | **0.9299** | **0.9991** |

*Source: [`Data/ablation_results.csv`](../Data/ablation_results.csv)*

### What the Ablation Tells Us

1. **SAR alone is insufficient.** With only 6 radar features, F1 drops to 0.5910 and IoU to 0.4195. SAR captures flood-related backscatter drops but cannot distinguish flood change from other landscape dynamics without optical context.

2. **Optical indices are the biggest single lift.** Adding 9 Sentinel-2 features jumps F1 from 0.5910 → 0.9598 (+0.3688). NDVI, NDWI, and their temporal changes provide critical discrimination between vegetation loss, water appearance, and debris.

3. **Terrain provides a small but consistent improvement.** DEM and slope add only 2 features but push F1 from 0.9598 → 0.9637 (+0.0039) and IoU from 0.9227 → 0.9299 (+0.0072). Terrain context helps the model identify flood-prone valley floors vs. stable ridges.

---

## Feature Importance

Random Forest feature importance (Gini importance from the final frozen model) was computed in Notebook 02. The top features are dominated by optical change variables and SAR post-flood signals, with terrain features (DEM, slope) contributing meaningful but smaller importance.

> [!CAUTION]
> TODO(owner): The exact feature importance ranking table and bar chart are generated in Notebook 02 but the specific numeric values were not extracted into a standalone file. Refer to the "Random Forest Feature Importance" section of [`Notebooks/02_EO_Spatial_ML_Change_Detection.ipynb`](../Notebooks/02_EO_Spatial_ML_Change_Detection.ipynb) for the complete importance plot and values.

---

## Failure Modes & Blind Spots

1. **Proxy label ceiling.** Since the model learns to reproduce an EO-derived anomaly rule, its performance is bounded by the quality of that rule. True flood boundaries (from field validation) may differ, especially in areas where SAR/optical signals are ambiguous (e.g., shadows, snow-covered regions).

2. **Monsoon cloud contamination.** Despite the cloud filter, Sentinel-2 median composites may include residual cloud/haze artifacts. This could create spurious optical change signals. In the ablation, SAR-only accuracy (0.80) provides a lower bound on performance in fully cloud-obscured scenarios.

3. **Fixed 80th-percentile threshold.** The proxy label threshold is global. In regions with uniformly high or low change scores, this may under- or over-label. Local adaptive thresholding was not implemented.

4. **Single event, single geography.** The model has been trained and tested on one flood event in one river basin. Performance on structurally different landscapes (coastal plains, urban areas) or different hazard types is unknown.

5. **Spatial block size sensitivity.** The 0.02° block size (~2.2 km) was chosen pragmatically. Smaller blocks (0.01°) give more folds but may not fully break autocorrelation. Larger blocks (0.03°) give fewer blocks, limiting CV fold count and split stability.

---

## Reproducibility

| Element | How It's Fixed |
|---|---|
| Random state | `RANDOM_STATE = 42` in [`configs/ml_config.json`](../configs/ml_config.json), passed to all splitters and models |
| Temporal windows | Fixed in [`configs/time_metadata.json`](../configs/time_metadata.json) |
| Feature definitions | Fixed in [`configs/ml_config.json → FEATURE_GROUPS`](../configs/ml_config.json) |
| Block size | `DEFAULT_BLOCK_SIZE = 0.02` in config |
| Split ratios | `DEV_SIZE = 0.80`, `TEST_SIZE = 0.20`, `TRAIN_SIZE = 0.75`, `VAL_SIZE = 0.25` in config |
| Model hyperparameters | Fixed in config; no search or tuning performed |
| Serialized model | [`Data/random_forest_final.joblib`](../Data/random_forest_final.joblib) with metadata in [`.json`](../Data/random_forest_final.json) |

---

[← Data & Features](data-and-features.md) | [🏠 Main README](../README.md)
