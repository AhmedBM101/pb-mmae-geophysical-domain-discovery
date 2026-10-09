# PB-MMAE Geophysical-Domain Discovery

Public reproducibility repository for the physics-balanced multimodal masked autoencoder (PB-MMAE) developed for label-free geophysical-domain discovery in the Middle–Upper Benue Trough, Nigeria.

## Scope

PB-MMAE learns spatial representations from harmonized magnetic, gravity, radiometric, thermal, structural, and terrain observations without geological labels, geothermal-favorability labels, or prospectivity classes. It uses family-specific encoders, spatial and whole-modality masking, equal-family reconstruction weighting, geographically blocked validation, consensus clustering, and cross-fold hierarchical consolidation.

The accompanying manuscript interprets the resulting classes as reproducible regional geophysical domains with potential geothermal relevance. They are **not** geothermal-resource classes, confirmed reservoirs, or drilling targets.

## Public reproducibility status

This repository is public and contains the source code, frozen canonical configuration, source-only notebooks, workflow scripts, and data-free smoke tests for the submitted PB-MMAE analysis. Restricted source rasters, licensed map layers, personal information, credentials, checkpoints, and unapproved derived products are deliberately excluded.

A full execution requires authorised local copies of the source data. The repository records the fixed channel order, model configuration, folds, seeds, stage order, and data-access conditions needed to inspect and reproduce the workflow lawfully.

## Repository layout

```text
configs/                    Frozen reproducible run configurations
data/                       Data-access, provenance, and redistribution documentation
notebooks/                  Source-only analytical notebooks; outputs removed
scripts/                    Command-line entry points, workflow helpers, and legacy record
src/pbmmae/                 Installable PB-MMAE Python package
tests/                      Data-free smoke tests
environment.yml             Minimal Python 3.11 environment specification
requirements-notebooks.txt  Notebook-only dependencies inferred from imports
```

## Installation

Create the documented Python 3.11 environment, then install the package:

```bash
conda env create -f environment.yml
conda activate pbmmae
python -m pip install .
```

The package dependency constraints are declared in `pyproject.toml`. Hardware-specific PyTorch builds may be substituted when GPU execution is required; the frozen paper configuration defaults to CPU for portability.

## One-command workflow

```bash
cp configs/paper1.example.yaml configs/paper1.yaml
# Set paths.project_root in configs/paper1.yaml to the authorised Paper1_SSL tree.
pbmmae preflight --config configs/paper1.yaml
pbmmae run --config configs/paper1.yaml
```

`project_root` is deliberately separate from the repository. It must contain the authorised `02_data/` inputs and `03_outputs/patches/` arrays; both are ignored by Git. The frozen workflow:

1. trains outer folds 1, 2, 4, and 5; fold 3 remains development-only;
2. extracts held-out representations and forms fixed k=8/k=9 consensus labels;
3. aligns folds, derives four meta-domains, maps them, and characterizes physical signatures;
4. evaluates withheld geological, geothermal-prospectivity, and warm-spring reference information;
5. exports the numerical-results lock and provenance record.

Use `pbmmae run --config configs/paper1.yaml --dry-run` to print the exact stage order. A named contiguous subset can be run with `--stages`, for example `--stages train evaluate consensus`.

## Verification without restricted data

Run the data-free software checks:

```bash
python -m unittest discover -s tests -p "test_*.py"
pbmmae run --config configs/paper1.example.yaml --dry-run
```

The smoke tests verify the 14-channel family ordering, model input/output shapes, frozen configuration, and pipeline stage plan. They do not reproduce scientific results; those require lawful access to the original inputs.

## Data access and provenance

No restricted source grids are redistributed. The exact 14-channel order, source categories, and access conditions are documented in [data/README.md](data/README.md). Before reproducing the analysis, obtain and use data under the relevant provider terms.

## Citation

Please cite the associated manuscript and this repository. Citation metadata are provided in [CITATION.cff](CITATION.cff).
