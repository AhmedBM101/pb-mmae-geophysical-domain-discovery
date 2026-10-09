import os
# ============================================================
# Paper 1 — 9x9 k=9 meta-domain sensitivity analysis
#
# Uses aligned k=9 labels from the 9x9 PB-MMAE sensitivity run.
#
# Goal:
#   Derive broader meta-domains from cross-model same-cell
#   cluster substitution and test meta-domain counts 2...8.
#
# This mirrors the validated 13x13 analysis.
# ============================================================

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import (
    linkage,
    fcluster
)

from scipy.spatial.distance import squareform

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score
)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(os.environ["PBMMAE_PROJECT_ROOT"]).expanduser().resolve()

ALIGNMENT_DIR = (
    PROJECT_ROOT /
    "03_outputs" /
    "models" /
    "PBMMAE" /
    "OuterFolds_9x9_LinearFusion64_Sensitivity" /
    "crossfold_alignment_k9" /
    "hungarian_alignment"
)

INPUT_FILE = (
    ALIGNMENT_DIR /
    "PBMMAE_9x9_k9_aligned_labels_long.csv"
)

OUTPUT_DIR = (
    ALIGNMENT_DIR /
    "metadomain_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

K_FINE = 9

OUTER_FOLDS = [
    1,
    2,
    4,
    5
]


# ============================================================
# 2. LOAD DATA
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        INPUT_FILE
    )


df = pd.read_csv(
    INPUT_FILE
)


required_columns = {

    "Cell_ID",
    "Model_Fold",
    "Aligned_cluster_k9"
}


if not required_columns.issubset(
    df.columns
):

    raise RuntimeError(
        "Required columns are missing."
    )


print("=" * 100)
print("9x9 k=9 META-DOMAIN SENSITIVITY ANALYSIS")
print("=" * 100)

print(
    "Rows:",
    len(df)
)

print(
    "Unique Cell_ID:",
    df[
        "Cell_ID"
    ].nunique()
)


# ============================================================
# 3. SAME-CELL CROSS-MODEL PAIRS
# ============================================================

pair_records = []


for cell_id, group in df.groupby(
    "Cell_ID"
):

    if len(group) < 2:

        continue


    rows = list(
        group.itertuples(
            index=False
        )
    )


    for a, b in combinations(
        rows,
        2
    ):

        pair_records.append({

            "Cell_ID":
                cell_id,

            "Model_A":
                int(
                    a.Model_Fold
                ),

            "Model_B":
                int(
                    b.Model_Fold
                ),

            "Label_A":
                int(
                    a.Aligned_cluster_k9
                ),

            "Label_B":
                int(
                    b.Aligned_cluster_k9
                )
        })


pair_df = pd.DataFrame(
    pair_records
)


print(
    "Cross-model same-cell pairs:",
    len(pair_df)
)


# ============================================================
# 4. CONFUSION MATRIX
# ============================================================

confusion = np.zeros(
    (
        K_FINE,
        K_FINE
    ),
    dtype=float
)


for row in pair_df.itertuples(
    index=False
):

    i = int(
        row.Label_A
    )

    j = int(
        row.Label_B
    )


    confusion[
        i,
        j
    ] += 1

    confusion[
        j,
        i
    ] += 1


confusion_df = pd.DataFrame(

    confusion,

    index=[
        f"C{i}"
        for i in range(
            K_FINE
        )
    ],

    columns=[
        f"C{i}"
        for i in range(
            K_FINE
        )
    ]
)


confusion_file = (
    OUTPUT_DIR /
    "PBMMAE_9x9_k9_crossfold_confusion_counts.csv"
)


confusion_df.to_csv(
    confusion_file
)


# ============================================================
# 5. SUBSTITUTION SIMILARITY
# ============================================================

row_sums = confusion.sum(
    axis=1,
    keepdims=True
)


with np.errstate(
    divide="ignore",
    invalid="ignore"
):

    directional = np.divide(

        confusion,

        row_sums,

        out=np.zeros_like(
            confusion
        ),

        where=row_sums != 0
    )


substitution = (
    0.5 *
    (
        directional
        +
        directional.T
    )
)


np.fill_diagonal(
    substitution,
    1.0
)


substitution_df = pd.DataFrame(

    substitution,

    index=[
        f"C{i}"
        for i in range(
            K_FINE
        )
    ],

    columns=[
        f"C{i}"
        for i in range(
            K_FINE
        )
    ]
)


substitution_file = (
    OUTPUT_DIR /
    "PBMMAE_9x9_k9_substitution_similarity.csv"
)


substitution_df.to_csv(
    substitution_file
)


# ============================================================
# 6. EDGE LIST
# ============================================================

edge_records = []


for i in range(
    K_FINE
):

    for j in range(
        i + 1,
        K_FINE
    ):

        edge_records.append({

            "Cluster_A":
                i,

            "Cluster_B":
                j,

            "Substitution_similarity":
                float(
                    substitution[
                        i,
                        j
                    ]
                ),

            "Raw_confusion_count":
                float(
                    confusion[
                        i,
                        j
                    ]
                    +
                    confusion[
                        j,
                        i
                    ]
                )
        })


edge_df = pd.DataFrame(
    edge_records
)


edge_df = edge_df.sort_values(

    "Substitution_similarity",

    ascending=False
).reset_index(
    drop=True
)


edge_file = (
    OUTPUT_DIR /
    "PBMMAE_9x9_k9_cluster_confusion_graph_edges.csv"
)


edge_df.to_csv(
    edge_file,
    index=False
)


# ============================================================
# 7. HIERARCHICAL CLUSTERING
# ============================================================

distance = (
    1.0 -
    substitution
)


distance = np.clip(
    distance,
    0.0,
    1.0
)


np.fill_diagonal(
    distance,
    0.0
)


condensed_distance = squareform(
    distance,
    checks=False
)


Z = linkage(

    condensed_distance,

    method="average"
)


linkage_df = pd.DataFrame(

    Z,

    columns=[
        "Cluster_1",
        "Cluster_2",
        "Distance",
        "New_cluster_size"
    ]
)


linkage_file = (
    OUTPUT_DIR /
    "PBMMAE_9x9_k9_metadomain_linkage.csv"
)


linkage_df.to_csv(
    linkage_file,
    index=False
)


# ============================================================
# 8. EVALUATE META-DOMAIN MAPPING
# ============================================================

def evaluate_mapping(
    fine_to_meta,
    n_meta
):

    temp = df[
        [
            "Cell_ID",
            "Model_Fold",
            "Aligned_cluster_k9"
        ]
    ].copy()


    temp[
        "Meta_domain"
    ] = (

        temp[
            "Aligned_cluster_k9"
        ]
        .map(
            fine_to_meta
        )
        .astype(
            int
        )
    )


    # --------------------------------------------------------
    # Cell-level agreement
    # --------------------------------------------------------

    cell_records = []


    for cell_id, group in temp.groupby(
        "Cell_ID"
    ):

        labels = group[
            "Meta_domain"
        ].to_numpy()


        counts = np.bincount(
            labels,
            minlength=n_meta
        )


        max_count = int(
            counts.max()
        )


        winners = np.where(
            counts == max_count
        )[0]


        n_models = len(
            labels
        )


        tied = (
            len(winners)
            >
            1
        )


        full = (
            max_count
            ==
            n_models
        )


        cell_records.append({

            "Cell_ID":
                cell_id,

            "N_models":
                n_models,

            "Agreement_fraction":
                float(
                    max_count /
                    n_models
                ),

            "Full_agreement":
                full,

            "Tie":
                tied
        })


    cell_df = pd.DataFrame(
        cell_records
    )


    # --------------------------------------------------------
    # Pairwise model agreement
    # --------------------------------------------------------

    accuracies = []
    aris = []
    nmis = []


    for fold_a, fold_b in combinations(
        OUTER_FOLDS,
        2
    ):

        A = temp[
            temp[
                "Model_Fold"
            ] == fold_a
        ][
            [
                "Cell_ID",
                "Meta_domain"
            ]
        ]


        B = temp[
            temp[
                "Model_Fold"
            ] == fold_b
        ][
            [
                "Cell_ID",
                "Meta_domain"
            ]
        ]


        shared = A.merge(

            B,

            on="Cell_ID",

            how="inner",

            suffixes=(
                "_A",
                "_B"
            )
        )


        if len(shared) == 0:

            continue


        y_a = shared[
            "Meta_domain_A"
        ].to_numpy()


        y_b = shared[
            "Meta_domain_B"
        ].to_numpy()


        accuracies.append(
            float(
                np.mean(
                    y_a
                    ==
                    y_b
                )
            )
        )


        aris.append(
            adjusted_rand_score(
                y_a,
                y_b
            )
        )


        nmis.append(
            normalized_mutual_info_score(
                y_a,
                y_b
            )
        )


    return {

        "N_meta_domains":
            n_meta,

        "Mean_cell_agreement":
            float(
                cell_df[
                    "Agreement_fraction"
                ].mean()
            ),

        "Median_cell_agreement":
            float(
                cell_df[
                    "Agreement_fraction"
                ].median()
            ),

        "Full_agreement_fraction":
            float(
                cell_df[
                    "Full_agreement"
                ].mean()
            ),

        "Tie_fraction":
            float(
                cell_df[
                    "Tie"
                ].mean()
            ),

        "Mean_pairwise_label_accuracy":
            float(
                np.mean(
                    accuracies
                )
            ),

        "Mean_pairwise_ARI":
            float(
                np.mean(
                    aris
                )
            ),

        "Mean_pairwise_NMI":
            float(
                np.mean(
                    nmis
                )
            )
    }


# ============================================================
# 9. TEST META-DOMAIN COUNTS 2...8
# ============================================================

candidate_records = []
mapping_records = []


for n_meta in range(
    2,
    K_FINE
):

    membership = fcluster(

        Z,

        t=n_meta,

        criterion="maxclust"
    )


    membership = (
        membership
        -
        1
    )


    mapping = {

        fine_cluster:
            int(
                membership[
                    fine_cluster
                ]
            )

        for fine_cluster
        in range(
            K_FINE
        )
    }


    result = evaluate_mapping(

        mapping,

        n_meta
    )


    candidate_records.append(
        result
    )


    for fine_cluster in range(
        K_FINE
    ):

        mapping_records.append({

            "N_meta_domains":
                n_meta,

            "Fine_cluster_k9":
                fine_cluster,

            "Meta_domain":
                mapping[
                    fine_cluster
                ]
        })


candidate_df = pd.DataFrame(
    candidate_records
)


mapping_df = pd.DataFrame(
    mapping_records
)


candidate_file = (
    OUTPUT_DIR /
    "PBMMAE_9x9_k9_metadomain_candidate_agreement.csv"
)


mapping_file = (
    OUTPUT_DIR /
    "PBMMAE_9x9_k9_fine_to_metadomain_mappings.csv"
)


candidate_df.to_csv(
    candidate_file,
    index=False
)


mapping_df.to_csv(
    mapping_file,
    index=False
)


# ============================================================
# 10. PRINT RESULTS
# ============================================================

print(
    "\n" +
    "=" * 110
)

print(
    "STRONGEST 9x9 CLUSTER SUBSTITUTIONS"
)

print(
    "=" * 110
)


print(

    edge_df[
        [
            "Cluster_A",
            "Cluster_B",
            "Substitution_similarity",
            "Raw_confusion_count"
        ]
    ]
    .head(
        15
    )
    .round(
        4
    )
    .to_string(
        index=False
    )
)


print(
    "\n" +
    "=" * 120
)

print(
    "9x9 META-DOMAIN REPRODUCIBILITY CANDIDATES"
)

print(
    "=" * 120
)


print(
    candidate_df
    .round(
        4
    )
    .to_string(
        index=False
    )
)


print(
    "\n" +
    "=" * 100
)

print(
    "9x9 FINE k=9 -> META-DOMAIN MAPPINGS"
)

print(
    "=" * 100
)


for n_meta in range(
    2,
    K_FINE
):

    subset = mapping_df[
        mapping_df[
            "N_meta_domains"
        ] == n_meta
    ]


    print(
        f"\n{n_meta} meta-domains:"
    )


    for meta_domain in sorted(
        subset[
            "Meta_domain"
        ].unique()
    ):

        clusters = (

            subset[
                subset[
                    "Meta_domain"
                ] == meta_domain
            ][
                "Fine_cluster_k9"
            ]
            .astype(
                int
            )
            .tolist()
        )


        print(
            f"  M{meta_domain}: {clusters}"
        )


print(
    "\nCandidate table:"
)

print(
    candidate_file
)


print(
    "\nMappings:"
)

print(
    mapping_file
)


print(
    "\nEdges:"
)

print(
    edge_file
)


print(
    "=" * 110
)