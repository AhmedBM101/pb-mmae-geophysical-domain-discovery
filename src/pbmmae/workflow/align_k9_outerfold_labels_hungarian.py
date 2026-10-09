import os
# ============================================================
# Paper 1 — k=9 outer-fold label alignment
#
# Aligns independently learned cluster IDs using shared
# geographic Cell_IDs and Hungarian assignment.
#
# IMPORTANT:
#   Latent centroids from different folds are NOT compared.
#   Alignment is based only on how cluster labels co-occur
#   on identical geographic cells.
#
# Reference model:
#   Fold 1
#
# Other folds:
#   Fold 2, Fold 4, Fold 5
#
# After alignment:
#   - compute agreement on shared cells
#   - compute ARI / NMI before and after label mapping
#   - create global aligned label table
#   - quantify multi-model cell agreement
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    accuracy_score
)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(os.environ["PBMMAE_PROJECT_ROOT"]).expanduser().resolve()

ALIGNMENT_ROOT = (
    PROJECT_ROOT /
    "03_outputs" /
    "models" /
    "PBMMAE" /
    "OuterFolds_13x13_LinearFusion64" /
    "crossfold_alignment_k9"
)

OUTPUT_DIR = (
    ALIGNMENT_ROOT /
    "hungarian_alignment"
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

REFERENCE_FOLD = 1

K = 9


# ============================================================
# 2. LOAD GEOGRAPHIC LABEL TABLES
# ============================================================

fold_tables = {}


for fold in FOLDS:

    path = (
        ALIGNMENT_ROOT /
        f"Fold{fold}_k9_geographic_labels.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            path
        )


    table = pd.read_csv(
        path
    )


    required_columns = {

        "Cell_ID",
        "Row",
        "Col",
        "X",
        "Y",
        "Spatial_Fold",
        "Model_Fold",
        "Role",
        "Cluster_k9"
    }


    if not required_columns.issubset(
        table.columns
    ):

        raise RuntimeError(
            f"Missing required columns in {path}"
        )


    assert table[
        "Cell_ID"
    ].is_unique


    assert table[
        "Cluster_k9"
    ].between(
        0,
        K - 1
    ).all()


    fold_tables[
        fold
    ] = table.copy()


print("=" * 100)
print("k=9 HUNGARIAN CROSS-FOLD LABEL ALIGNMENT")
print("=" * 100)

print(
    "Reference fold:",
    REFERENCE_FOLD
)


# ============================================================
# 3. REFERENCE FOLD
# ============================================================

reference = (
    fold_tables[
        REFERENCE_FOLD
    ].copy()
)

reference[
    "Aligned_cluster_k9"
] = reference[
    "Cluster_k9"
].astype(
    int
)


fold_tables[
    REFERENCE_FOLD
] = reference


# Identity mapping for reference

mapping_records = []


for cluster_id in range(
    K
):

    mapping_records.append({

        "Source_Fold":
            REFERENCE_FOLD,

        "Original_cluster":
            cluster_id,

        "Aligned_cluster":
            cluster_id,

        "Reference_Fold":
            REFERENCE_FOLD,

        "Shared_cells_used":
            len(
                reference
            ),

        "Matched_overlap":
            np.nan
    })


# ============================================================
# 4. ALIGN EACH OTHER FOLD TO FOLD 1
# ============================================================

agreement_records = []


for fold in FOLDS:

    if fold == REFERENCE_FOLD:

        continue


    target = (
        fold_tables[
            fold
        ].copy()
    )


    shared = reference[
        [
            "Cell_ID",
            "Cluster_k9"
        ]
    ].merge(

        target[
            [
                "Cell_ID",
                "Cluster_k9"
            ]
        ],

        on=
            "Cell_ID",

        how=
            "inner",

        suffixes=(
            "_REF",
            "_TARGET"
        )
    )


    if len(
        shared
    ) == 0:

        raise RuntimeError(
            f"No shared cells between Fold 1 and Fold {fold}"
        )


    # --------------------------------------------------------
    # Contingency matrix
    #
    # rows    = reference clusters
    # columns = target clusters
    # --------------------------------------------------------

    contingency = np.zeros(
        (
            K,
            K
        ),
        dtype=np.int64
    )


    for ref_label, target_label in zip(

        shared[
            "Cluster_k9_REF"
        ].astype(
            int
        ),

        shared[
            "Cluster_k9_TARGET"
        ].astype(
            int
        )
    ):

        contingency[
            ref_label,
            target_label
        ] += 1


    # --------------------------------------------------------
    # Hungarian assignment
    #
    # maximize overlap => minimize negative overlap
    # --------------------------------------------------------

    row_ind, col_ind = (
        linear_sum_assignment(
            -contingency
        )
    )


    # target original -> reference label

    target_to_reference = {

        int(
            target_cluster
        ):
            int(
                reference_cluster
            )

        for reference_cluster,
            target_cluster
        in zip(
            row_ind,
            col_ind
        )
    }


    assert len(
        target_to_reference
    ) == K


    # --------------------------------------------------------
    # Record mapping
    # --------------------------------------------------------

    for target_cluster in range(
        K
    ):

        aligned_cluster = (
            target_to_reference[
                target_cluster
            ]
        )


        matched_overlap = int(

            contingency[
                aligned_cluster,
                target_cluster
            ]
        )


        mapping_records.append({

            "Source_Fold":
                fold,

            "Original_cluster":
                target_cluster,

            "Aligned_cluster":
                aligned_cluster,

            "Reference_Fold":
                REFERENCE_FOLD,

            "Shared_cells_used":
                len(
                    shared
                ),

            "Matched_overlap":
                matched_overlap
        })


    # --------------------------------------------------------
    # Apply mapping
    # --------------------------------------------------------

    target[
        "Aligned_cluster_k9"
    ] = (

        target[
            "Cluster_k9"
        ]
        .map(
            target_to_reference
        )
        .astype(
            int
        )
    )


    fold_tables[
        fold
    ] = target


    # --------------------------------------------------------
    # Agreement diagnostics on shared cells
    # --------------------------------------------------------

    aligned_shared = reference[
        [
            "Cell_ID",
            "Aligned_cluster_k9"
        ]
    ].merge(

        target[
            [
                "Cell_ID",
                "Aligned_cluster_k9"
            ]
        ],

        on=
            "Cell_ID",

        how=
            "inner",

        suffixes=(
            "_REF",
            "_TARGET"
        )
    )


    y_ref = aligned_shared[
        "Aligned_cluster_k9_REF"
    ].to_numpy()


    y_target = aligned_shared[
        "Aligned_cluster_k9_TARGET"
    ].to_numpy()


    raw_shared = shared.copy()


    raw_ari = adjusted_rand_score(

        raw_shared[
            "Cluster_k9_REF"
        ],

        raw_shared[
            "Cluster_k9_TARGET"
        ]
    )


    raw_nmi = normalized_mutual_info_score(

        raw_shared[
            "Cluster_k9_REF"
        ],

        raw_shared[
            "Cluster_k9_TARGET"
        ]
    )


    aligned_accuracy = accuracy_score(

        y_ref,

        y_target
    )


    aligned_ari = adjusted_rand_score(

        y_ref,

        y_target
    )


    aligned_nmi = normalized_mutual_info_score(

        y_ref,

        y_target
    )


    agreement_records.append({

        "Reference_Fold":
            REFERENCE_FOLD,

        "Target_Fold":
            fold,

        "Shared_cells":
            len(
                aligned_shared
            ),

        "Raw_ARI":
            raw_ari,

        "Raw_NMI":
            raw_nmi,

        "Aligned_label_accuracy":
            aligned_accuracy,

        "Aligned_ARI":
            aligned_ari,

        "Aligned_NMI":
            aligned_nmi
    })


    # --------------------------------------------------------
    # Save contingency matrix
    # --------------------------------------------------------

    contingency_df = pd.DataFrame(

        contingency,

        index=[
            f"Reference_{i}"
            for i in range(
                K
            )
        ],

        columns=[
            f"Fold{fold}_{j}"
            for j in range(
                K
            )
        ]
    )


    contingency_df.to_csv(

        OUTPUT_DIR /
        f"Fold1_vs_Fold{fold}_"
        f"k9_contingency_matrix.csv"
    )


    print(
        "\n" +
        "-" * 90
    )

    print(
        f"FOLD 1 vs FOLD {fold}"
    )

    print(
        "-" * 90
    )

    print(
        "Shared cells:",
        len(
            shared
        )
    )

    print(
        "Mapping target -> reference:"
    )

    print(
        target_to_reference
    )

    print(
        "Aligned label accuracy:",
        round(
            aligned_accuracy,
            4
        )
    )

    print(
        "ARI:",
        round(
            aligned_ari,
            4
        )
    )

    print(
        "NMI:",
        round(
            aligned_nmi,
            4
        )
    )


# ============================================================
# 5. SAVE MAPPING TABLE
# ============================================================

mapping_df = pd.DataFrame(
    mapping_records
)


mapping_file = (
    OUTPUT_DIR /
    "PBMMAE_k9_Hungarian_label_mapping.csv"
)


mapping_df.to_csv(
    mapping_file,
    index=False
)


# ============================================================
# 6. SAVE PAIRWISE REFERENCE AGREEMENT
# ============================================================

agreement_df = pd.DataFrame(
    agreement_records
)


agreement_file = (
    OUTPUT_DIR /
    "PBMMAE_k9_reference_alignment_agreement.csv"
)


agreement_df.to_csv(
    agreement_file,
    index=False
)


# ============================================================
# 7. BUILD LONG ALIGNED TABLE
# ============================================================

aligned_long = pd.concat(

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
                "Cluster_k9",
                "Aligned_cluster_k9"
            ]
        ]

        for fold
        in FOLDS
    ],

    ignore_index=True
)


aligned_long_file = (
    OUTPUT_DIR /
    "PBMMAE_k9_aligned_labels_long.csv"
)


aligned_long.to_csv(
    aligned_long_file,
    index=False
)


# ============================================================
# 8. GLOBAL CELL-LEVEL CONSENSUS
# ============================================================

global_records = []


for cell_id, group in aligned_long.groupby(
    "Cell_ID"
):

    labels = (
        group[
            "Aligned_cluster_k9"
        ]
        .astype(
            int
        )
        .to_numpy()
    )


    counts = np.bincount(
        labels,
        minlength=K
    )


    max_count = int(
        counts.max()
    )


    winners = np.where(
        counts == max_count
    )[0]


    # deterministic tie handling:
    # leave consensus undefined if tie

    if len(
        winners
    ) == 1:

        consensus_label = int(
            winners[0]
        )

        tied = False

    else:

        consensus_label = -1

        tied = True


    n_models = len(
        labels
    )


    agreement_fraction = float(

        max_count /
        n_models
    )


    first = (
        group.iloc[0]
    )


    global_records.append({

        "Cell_ID":
            cell_id,

        "Row":
            int(
                first[
                    "Row"
                ]
            ),

        "Col":
            int(
                first[
                    "Col"
                ]
            ),

        "X":
            float(
                first[
                    "X"
                ]
            ),

        "Y":
            float(
                first[
                    "Y"
                ]
            ),

        "Spatial_Fold":
            int(
                first[
                    "Spatial_Fold"
                ]
            ),

        "N_models":
            n_models,

        "Consensus_cluster_k9":
            consensus_label,

        "Consensus_tie":
            tied,

        "Agreement_count":
            max_count,

        "Agreement_fraction":
            agreement_fraction
    })


global_df = pd.DataFrame(
    global_records
)


global_file = (
    OUTPUT_DIR /
    "PBMMAE_k9_global_aligned_consensus.csv"
)


global_df.to_csv(
    global_file,
    index=False
)


# ============================================================
# 9. AGREEMENT SUMMARY
# ============================================================

summary_records = []


for n_models, subset in global_df.groupby(
    "N_models"
):

    summary_records.append({

        "N_models":
            int(
                n_models
            ),

        "N_cells":
            len(
                subset
            ),

        "Mean_agreement_fraction":
            float(
                subset[
                    "Agreement_fraction"
                ].mean()
            ),

        "Median_agreement_fraction":
            float(
                subset[
                    "Agreement_fraction"
                ].median()
            ),

        "Full_agreement_cells":
            int(
                np.sum(
                    subset[
                        "Agreement_fraction"
                    ] == 1.0
                )
            ),

        "Full_agreement_fraction":
            float(
                np.mean(
                    subset[
                        "Agreement_fraction"
                    ] == 1.0
                )
            ),

        "Tied_cells":
            int(
                subset[
                    "Consensus_tie"
                ].sum()
            )
    })


agreement_summary_df = pd.DataFrame(
    summary_records
)


agreement_summary_file = (
    OUTPUT_DIR /
    "PBMMAE_k9_global_alignment_agreement_summary.csv"
)


agreement_summary_df.to_csv(
    agreement_summary_file,
    index=False
)


# ============================================================
# 10. PRINT RESULTS
# ============================================================

print(
    "\n\n" +
    "=" * 110
)

print(
    "REFERENCE-BASED ALIGNMENT AGREEMENT"
)

print(
    "=" * 110
)


print(

    agreement_df
    .round(
        4
    )
    .to_string(
        index=False
    )
)


print(
    "\n" +
    "=" * 110
)

print(
    "GLOBAL MULTI-MODEL CELL AGREEMENT"
)

print(
    "=" * 110
)


print(

    agreement_summary_df
    .round(
        4
    )
    .to_string(
        index=False
    )
)


print(
    "\nGlobal cells:",
    len(
        global_df
    )
)


print(
    "Cells with unique consensus:",
    int(
        np.sum(
            ~global_df[
                "Consensus_tie"
            ]
        )
    )
)


print(
    "Tied cells:",
    int(
        np.sum(
            global_df[
                "Consensus_tie"
            ]
        )
    )
)


print(
    "\nMean global agreement:",
    round(
        global_df[
            "Agreement_fraction"
        ].mean(),
        4
    )
)


print(
    "Median global agreement:",
    round(
        global_df[
            "Agreement_fraction"
        ].median(),
        4
    )
)


print(
    "\nSaved mapping:"
)

print(
    mapping_file
)


print(
    "\nSaved reference agreement:"
)

print(
    agreement_file
)


print(
    "\nSaved aligned long table:"
)

print(
    aligned_long_file
)


print(
    "\nSaved global consensus:"
)

print(
    global_file
)


print(
    "\nSaved global agreement summary:"
)

print(
    agreement_summary_file
)


print(
    "=" * 110
)