import os
# ============================================================
# Paper 1 — k=9 cross-fold geographic overlap audit
#
# Purpose:
#   Join consensus labels to stable geographic Cell_IDs and
#   determine how much common geographic support exists
#   between independently trained outer-fold models.
#
# This script DOES NOT align cluster labels yet.
# ============================================================

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(os.environ["PBMMAE_PROJECT_ROOT"]).expanduser().resolve()

PATCH_ROOT = (
    PROJECT_ROOT /
    "03_outputs" /
    "patches" /
    "13x13"
)

CONSENSUS_ROOT = (
    PROJECT_ROOT /
    "03_outputs" /
    "models" /
    "PBMMAE" /
    "OuterFolds_13x13_LinearFusion64" /
    "outer_fold_consensus_k8_k9"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "03_outputs" /
    "models" /
    "PBMMAE" /
    "OuterFolds_13x13_LinearFusion64" /
    "crossfold_alignment_k9"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FOLDS = [
    1,
    2,
    4,
    5
]

K = 9


# ============================================================
# 2. LOAD ONE FOLD'S GEOGRAPHIC LABEL TABLE
# ============================================================

def load_fold_table(
    fold
):

    patch_dir = (
        PATCH_ROOT /
        f"Fold_{fold}"
    )

    consensus_dir = (
        CONSENSUS_ROOT /
        f"Fold{fold}" /
        "k9"
    )


    train_meta_file = (
        patch_dir /
        f"Fold{fold}_TRAIN_13x13_metadata.csv"
    )

    val_meta_file = (
        patch_dir /
        f"Fold{fold}_VALIDATION_13x13_metadata.csv"
    )


    train_labels_file = (
        consensus_dir /
        f"Fold{fold}_k9_consensus_TRAIN_labels.npy"
    )

    val_labels_file = (
        consensus_dir /
        f"Fold{fold}_k9_consensus_VALIDATION_labels.npy"
    )


    for path in [
        train_meta_file,
        val_meta_file,
        train_labels_file,
        val_labels_file
    ]:

        if not path.exists():

            raise FileNotFoundError(
                path
            )


    train_meta = pd.read_csv(
        train_meta_file
    )

    val_meta = pd.read_csv(
        val_meta_file
    )


    train_labels = np.load(
        train_labels_file
    )

    val_labels = np.load(
        val_labels_file
    )


    assert len(
        train_meta
    ) == len(
        train_labels
    )

    assert len(
        val_meta
    ) == len(
        val_labels
    )


    train = train_meta[
        [
            "Cell_ID",
            "Row",
            "Col",
            "X",
            "Y",
            "Spatial_Fold"
        ]
    ].copy()


    train[
        "Model_Fold"
    ] = fold

    train[
        "Role"
    ] = "TRAIN"

    train[
        "Cluster_k9"
    ] = train_labels.astype(
        int
    )


    val = val_meta[
        [
            "Cell_ID",
            "Row",
            "Col",
            "X",
            "Y",
            "Spatial_Fold"
        ]
    ].copy()


    val[
        "Model_Fold"
    ] = fold

    val[
        "Role"
    ] = "VALIDATION"

    val[
        "Cluster_k9"
    ] = val_labels.astype(
        int
    )


    table = pd.concat(
        [
            train,
            val
        ],
        ignore_index=True
    )


    # --------------------------------------------------------
    # Basic integrity
    # --------------------------------------------------------

    assert table[
        "Cell_ID"
    ].is_unique


    assert table[
        "Cluster_k9"
    ].between(
        0,
        K - 1
    ).all()


    return table


# ============================================================
# 3. LOAD ALL FOUR OUTER FOLDS
# ============================================================

fold_tables = {}


for fold in FOLDS:

    table = load_fold_table(
        fold
    )

    fold_tables[
        fold
    ] = table


    output_file = (
        OUTPUT_DIR /
        f"Fold{fold}_k9_geographic_labels.csv"
    )


    table.to_csv(
        output_file,
        index=False
    )


    print(
        "\n" +
        "=" * 80
    )

    print(
        f"FOLD {fold}"
    )

    print(
        "=" * 80
    )

    print(
        "Total geographically labeled patches:",
        len(
            table
        )
    )

    print(
        "TRAIN:",
        int(
            (
                table[
                    "Role"
                ]
                ==
                "TRAIN"
            ).sum()
        )
    )

    print(
        "VALIDATION:",
        int(
            (
                table[
                    "Role"
                ]
                ==
                "VALIDATION"
            ).sum()
        )
    )

    print(
        "Unique Cell_ID:",
        table[
            "Cell_ID"
        ].nunique()
    )

    print(
        "Clusters represented:",
        sorted(
            table[
                "Cluster_k9"
            ]
            .unique()
            .tolist()
        )
    )


# ============================================================
# 4. PAIRWISE SHARED Cell_ID AUDIT
# ============================================================

overlap_records = []


for fold_a, fold_b in combinations(
    FOLDS,
    2
):

    A = fold_tables[
        fold_a
    ]

    B = fold_tables[
        fold_b
    ]


    merged = A.merge(

        B,

        on="Cell_ID",

        how="inner",

        suffixes=(
            f"_F{fold_a}",
            f"_F{fold_b}"
        )
    )


    n_shared = len(
        merged
    )


    # --------------------------------------------------------
    # Check geographic identity really matches
    # --------------------------------------------------------

    if n_shared > 0:

        row_match = np.all(
            merged[
                f"Row_F{fold_a}"
            ].to_numpy()
            ==
            merged[
                f"Row_F{fold_b}"
            ].to_numpy()
        )

        col_match = np.all(
            merged[
                f"Col_F{fold_a}"
            ].to_numpy()
            ==
            merged[
                f"Col_F{fold_b}"
            ].to_numpy()
        )

        x_match = np.allclose(
            merged[
                f"X_F{fold_a}"
            ].to_numpy(),
            merged[
                f"X_F{fold_b}"
            ].to_numpy()
        )

        y_match = np.allclose(
            merged[
                f"Y_F{fold_a}"
            ].to_numpy(),
            merged[
                f"Y_F{fold_b}"
            ].to_numpy()
        )

    else:

        row_match = False
        col_match = False
        x_match = False
        y_match = False


    # --------------------------------------------------------
    # Role combinations among shared cells
    # --------------------------------------------------------

    if n_shared > 0:

        role_counts = (
            merged
            .groupby(
                [
                    f"Role_F{fold_a}",
                    f"Role_F{fold_b}"
                ]
            )
            .size()
            .to_dict()
        )

    else:

        role_counts = {}


    train_train = int(
        role_counts.get(
            (
                "TRAIN",
                "TRAIN"
            ),
            0
        )
    )


    train_val = int(
        role_counts.get(
            (
                "TRAIN",
                "VALIDATION"
            ),
            0
        )
    )


    val_train = int(
        role_counts.get(
            (
                "VALIDATION",
                "TRAIN"
            ),
            0
        )
    )


    val_val = int(
        role_counts.get(
            (
                "VALIDATION",
                "VALIDATION"
            ),
            0
        )
    )


    overlap_records.append({

        "Fold_A":
            fold_a,

        "Fold_B":
            fold_b,

        "N_A":
            len(
                A
            ),

        "N_B":
            len(
                B
            ),

        "Shared_Cell_ID":
            n_shared,

        "Shared_fraction_A":
            (
                n_shared /
                len(
                    A
                )
            ),

        "Shared_fraction_B":
            (
                n_shared /
                len(
                    B
                )
            ),

        "TRAIN_TRAIN_shared":
            train_train,

        "TRAIN_VALIDATION_shared":
            train_val,

        "VALIDATION_TRAIN_shared":
            val_train,

        "VALIDATION_VALIDATION_shared":
            val_val,

        "Row_match":
            row_match,

        "Col_match":
            col_match,

        "X_match":
            x_match,

        "Y_match":
            y_match
    })


    # --------------------------------------------------------
    # Save shared geographic table
    # --------------------------------------------------------

    shared_file = (
        OUTPUT_DIR /
        f"Fold{fold_a}_vs_Fold{fold_b}_"
        f"k9_shared_cells.csv"
    )


    merged.to_csv(
        shared_file,
        index=False
    )


    print(
        "\n" +
        "-" * 80
    )

    print(
        f"FOLD {fold_a} vs FOLD {fold_b}"
    )

    print(
        "-" * 80
    )

    print(
        "Shared Cell_ID:",
        n_shared
    )

    print(
        "TRAIN / TRAIN:",
        train_train
    )

    print(
        "TRAIN / VALIDATION:",
        train_val
    )

    print(
        "VALIDATION / TRAIN:",
        val_train
    )

    print(
        "VALIDATION / VALIDATION:",
        val_val
    )

    print(
        "Geographic identity check:"
    )

    print(
        " Row:",
        row_match,
        "| Col:",
        col_match,
        "| X:",
        x_match,
        "| Y:",
        y_match
    )


# ============================================================
# 5. SAVE PAIRWISE OVERLAP TABLE
# ============================================================

overlap_df = pd.DataFrame(
    overlap_records
)


overlap_file = (
    OUTPUT_DIR /
    "PBMMAE_k9_crossfold_CellID_overlap_audit.csv"
)


overlap_df.to_csv(
    overlap_file,
    index=False
)


# ============================================================
# 6. GLOBAL CELL COVERAGE
# ============================================================

all_long = pd.concat(

    [
        fold_tables[
            fold
        ][
            [
                "Cell_ID",
                "Row",
                "Col",
                "X",
                "Y",
                "Spatial_Fold",
                "Model_Fold",
                "Role",
                "Cluster_k9"
            ]
        ]

        for fold
        in FOLDS
    ],

    ignore_index=True
)


coverage = (

    all_long
    .groupby(
        "Cell_ID"
    )
    .agg(

        Row=(
            "Row",
            "first"
        ),

        Col=(
            "Col",
            "first"
        ),

        X=(
            "X",
            "first"
        ),

        Y=(
            "Y",
            "first"
        ),

        Spatial_Fold=(
            "Spatial_Fold",
            "first"
        ),

        N_models_available=(
            "Model_Fold",
            "nunique"
        )
    )
    .reset_index()
)


coverage_file = (
    OUTPUT_DIR /
    "PBMMAE_k9_global_CellID_model_coverage.csv"
)


coverage.to_csv(
    coverage_file,
    index=False
)


# ============================================================
# 7. PRINT FINAL SUMMARY
# ============================================================

print(
    "\n\n" +
    "=" * 100
)

print(
    "PAIRWISE CROSS-FOLD OVERLAP SUMMARY"
)

print(
    "=" * 100
)


print(

    overlap_df[
        [
            "Fold_A",
            "Fold_B",
            "Shared_Cell_ID",
            "Shared_fraction_A",
            "Shared_fraction_B",
            "TRAIN_TRAIN_shared",
            "TRAIN_VALIDATION_shared",
            "VALIDATION_TRAIN_shared",
            "VALIDATION_VALIDATION_shared",
            "Row_match",
            "Col_match",
            "X_match",
            "Y_match"
        ]
    ]
    .round(
        4
    )
    .to_string(
        index=False
    )
)


print(
    "\nGlobal unique Cell_ID:",
    coverage[
        "Cell_ID"
    ].nunique()
)


print(
    "\nNumber of models available per Cell_ID:"
)


print(

    coverage[
        "N_models_available"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


print(
    "\nSaved overlap audit:"
)

print(
    overlap_file
)


print(
    "\nSaved global coverage:"
)

print(
    coverage_file
)


print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)


print(
    "=" * 100
)