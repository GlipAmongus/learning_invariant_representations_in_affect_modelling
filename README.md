# Quantifying Uncertainty in Affect Modelling

## Project Overview

This project explores how invariant data representations derived via Slow Feature Analysis (SFA) can reduce bias in affect (emotion) modelling. The pipeline processes the RECOLA dataset through preprocessing, slow feature extraction, feature consolidation, model training, and comprehensive evaluation.

## Conda Environments

### `ml_env` — Machine Learning Environment
**Used in:** Notebooks 1, 3, 4, 5
**Create with:**
```bash
conda env create -f env_ml.yaml
conda activate ml_env
```
**Purpose:** Standard ML stack for data preprocessing, PCA, feature consolidation, model training, and evaluation.
**Key packages:** numpy, pandas, scikit-learn, matplotlib, jupyter

### `sfa_env` — Slow Feature Analysis Environment
**Used in:** Notebook 2
**Create with:**
```bash
conda env create -f env_sfa.yaml
conda activate sfa_env
```
**Purpose:** Specialized environment for SFA computation using the MDP library.
**Key packages:** numpy <1.24 (required for MDP), MDP, pandas, matplotlib
**Note:** This environment is **incompatible with scikit-learn** due to numpy version constraints. SFA features must be computed here, then loaded in `ml_env` for downstream tasks.

## Pipeline Overview

| Notebook | Environment | Task |
|----------|-------------|------|
| **1_data_preprocessing.ipynb** | `ml_env` | Load RECOLA CSV, standardize features, save raw features & labels |
| **2_deriving_slow_features.ipynb** | `sfa_env` | Extract slow linear, degree 2, and degree 3 SFA features across [5, 10, 15, 20, 25, 30] dimensions |
| **3_feature_generation.ipynb** | `ml_env` | Consolidate raw features, compute PCA baselines, load SFA features, assemble experiment data |
| **4_model_definitions.ipynb** | `ml_env` | Define metrics, model factory, cross-validation strategies, and experiment execution framework |
| **5_evaluation_and_results.ipynb** | `ml_env` | Run full experiment matrix, aggregate results, generate visualizations and summary tables |

## Quick Start

1. **Create both environments:**
   ```bash
   conda env create -f env_ml.yaml
   conda env create -f env_sfa.yaml
   ```

2. **Run Notebook 1** (in `ml_env`) — Data Preprocessing:
   ```bash
   conda activate ml_env
   jupyter notebook 1_data_preprocessing.ipynb
   ```

3. **Run Notebook 2** (in `sfa_env`) — Derive Slow Features:
   ```bash
   conda activate sfa_env
   jupyter notebook 2_deriving_slow_features.ipynb
   ```
   **Warning:** Notebook 2 is computationally intensive (~8 hours total for all feature variants). Consider running overnight or on a high-performance machine.

4. **Run Notebooks 3–5** (in `ml_env`) — Feature Consolidation, Model Definitions & Evaluation:
   ```bash
   conda activate ml_env
   jupyter notebook 3_feature_generation.ipynb
   jupyter notebook 4_model_definitions.ipynb
   jupyter notebook 5_evaluation_and_results.ipynb
   ```

## Output Structure

```
features/
├── raw_features.npy              (from Notebook 1)
├── participant_ids.npy           (from Notebook 1)
├── y_reg_arousal.npy             (from Notebook 1)
├── y_reg_valence.npy             (from Notebook 1)
├── y_clf_arousal.npy             (from Notebook 1)
├── y_clf_valence.npy             (from Notebook 1)
├── slow_linear_all_k.npy         (from Notebook 2, k ∈ [5,10,15,20,25,30])
├── slow_deg2_all_k.npy           (from Notebook 2, k ∈ [5,10,15,20,25,30])
├── slow_deg3_all_k.npy           (from Notebook 2, k ∈ [5,10,15,20,25,30])
└── experiment_data.pkl           (from Notebook 3 — consolidated data object)

results/
├── experiment_summary.csv        (from Notebook 5 — all experiments)
├── regression_summary.csv        (from Notebook 5 — regression results)
├── classification_summary.csv    (from Notebook 5 — classification results)
├── results.pkl                   (from Notebook 5 — nested results dictionary)
└── [visualizations]
    ├── regression_metrics_by_component.png
    ├── classification_metrics_by_component.png
    ├── regression_heatmaps.png
    ├── classification_heatmaps.png
    └── cv_gap_analysis.png
```
