import os
"""Join frozen k=9 consensus labels to 13x13 patch-centre metadata.

The bridge produces the Cell_ID-level tables consumed by the cross-fold
Hungarian alignment stage. It only checks and joins same-order arrays and
metadata created during patch generation; it never fits or changes a model.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ["PBMMAE_PROJECT_ROOT"]).expanduser().resolve()
PATCH_ROOT = PROJECT_ROOT / "03_outputs" / "patches" / "13x13"
CONSENSUS_ROOT = (
    PROJECT_ROOT / "03_outputs" / "models" / "PBMMAE" /
    "OuterFolds_13x13_LinearFusion64" / "outer_fold_consensus_k8_k9"
)
OUTPUT_ROOT = (
    PROJECT_ROOT / "03_outputs" / "models" / "PBMMAE" /
    "OuterFolds_13x13_LinearFusion64" / "crossfold_alignment_k9"
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
FOLDS = (1, 2, 4, 5)
REQUIRED = ("Cell_ID", "Row", "Col", "X", "Y", "Spatial_Fold")


def build_fold_table(fold: int) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for role in ("TRAIN", "VALIDATION"):
        metadata_path = PATCH_ROOT / f"Fold_{fold}" / f"Fold{fold}_{role}_13x13_metadata.csv"
        labels_path = CONSENSUS_ROOT / f"Fold{fold}" / "k9" / f"Fold{fold}_k9_consensus_{role}_labels.npy"
        if not metadata_path.exists() or not labels_path.exists():
            missing = [str(path) for path in (metadata_path, labels_path) if not path.exists()]
            raise FileNotFoundError("Missing required metadata or consensus labels:\n" + "\n".join(missing))
        metadata = pd.read_csv(metadata_path)
        if not set(REQUIRED).issubset(metadata.columns):
            raise RuntimeError(f"Metadata columns missing from {metadata_path}: {set(REQUIRED) - set(metadata.columns)}")
        labels = np.load(labels_path)
        if labels.ndim != 1 or len(labels) != len(metadata):
            raise ValueError(f"Metadata/label mismatch for Fold {fold} {role}: {len(metadata)} rows versus {labels.shape}")
        if not np.isfinite(labels).all() or not np.isin(labels, np.arange(9)).all():
            raise ValueError(f"Expected integer labels 0..8 in {labels_path}")
        part = metadata.loc[:, REQUIRED].copy()
        part["Model_Fold"] = fold
        part["Role"] = role
        part["Cluster_k9"] = labels.astype(int)
        pieces.append(part)
    table = pd.concat(pieces, ignore_index=True)
    if not table["Cell_ID"].astype(str).is_unique:
        raise RuntimeError(f"Fold {fold} contains duplicate patch-centre Cell_ID values.")
    if sorted(table["Cluster_k9"].unique()) != list(range(9)):
        raise RuntimeError(f"Fold {fold} does not contain all frozen k=9 labels.")
    return table


for fold in FOLDS:
    destination = OUTPUT_ROOT / f"Fold{fold}_k9_geographic_labels.csv"
    table = build_fold_table(fold)
    table.to_csv(destination, index=False)
    print(f"Saved {len(table)} geographic labels: {destination}")
