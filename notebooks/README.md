# Research notebooks

This directory contains source-only copies of the PB-MMAE research notebooks. Their cell outputs, execution counts, embedded figures, and tabular results have been removed before version control. The notebooks do not contain raw rasters, derived patch tensors, trained checkpoints, or publication figures.

## Current status

These notebooks preserve the analytical record used for the manuscript, but they are **not yet a one-command reproducibility release**. They retain the original hard-coded local `PROJECT_ROOT` / `PROJECT` definitions and require authorized local inputs and previously generated intermediates. The next refactoring stage will replace those definitions with the repository configuration and move shared functions into `src/pbmmae/`.

## Intended run sequence

1. `Paper1_01_Data_Cube_and_QC.ipynb` — construct and audit the 14-variable data cube.
2. `Paper1_02_Spatial_CV_and_Scaling.ipynb` — define geographically blocked folds and scaling.
3. `Paper1_03_Patch_Generation.ipynb` — create 13 × 13 spatial patches.
4. `Paper1_04_Baseline_Models.ipynb` — PCA, NMF, SOM, and convolutional-autoencoder baselines.
5. `Paper1_05_PBMMAE_Model_Development.ipynb` — PB-MMAE development and representation evaluation.
6. `Paper1_10_MAE_Baseline_13x13.ipynb` and `Paper1_10B_MAE_Consensus_13x13.ipynb` — MAE baseline and consensus analysis.
7. `Paper1_11_MultiMAE_Baseline_13x13.ipynb` and `Paper1_11B_MultiMAE_Consensus_13x13.ipynb` — MultiMAE baseline and consensus analysis.
8. `Paper1_06_Fig03_Final_Domain_Stability.ipynb`, `Paper1_07_Fig04_Domain_Interpretation.ipynb`, and `Paper1_08_Fig05_Baselines_Sensitivity_Ablations.ipynb` — final manuscript figures.
9. `Paper1_09_Supplementary_Material.ipynb` — supplementary diagnostics and figures.

Use these notebooks only with authorised source data. Do not commit generated outputs, cached arrays, model weights, or any restricted datasets.
