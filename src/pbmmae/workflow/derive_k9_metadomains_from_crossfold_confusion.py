import os
# ============================================================
# Paper 1 — Derive reproducible meta-domains from aligned k=9
#
# Goal:
#   Identify which aligned k=9 clusters repeatedly substitute
#   for one another across independently trained outer folds.
#
# Inputs:
#   - PBMMAE_k9_aligned_labels_long.csv
#
# Outputs:
#   1. Pairwise cluster confusion matrix
#   2. Symmetric confusion / substitution matrix
#   3. Cluster graph edge list
#   4. Hierarchical clustering of the 9 aligned clusters
#   5. Candidate meta-domain solutions for m = 2...8
#   6. Agreement improvement after collapsing k=9 clusters
#      into broader meta-domains
#
# IMPORTANT:
#   This script does NOT use latent-centroid distances across
#   independently trained models.
#
#   Meta-domains are derived from disagreement structure on
#   identical geographic Cell_IDs.
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

ALIGNMENT_ROOT = (
    PROJECT_ROOT /
    "03_outputs" /
    "models" /
    "PBMMAE" /
    "OuterFolds_13x13_LinearFusion64" /
    "crossfold_alignment_k9" /
    "hungarian_alignment"
)

INPUT_FILE = (
    ALIGNMENT_ROOT /
    "PBMMAE_k9_aligned_labels_long.csv"
)

OUTPUT_DIR = (
    ALIGNMENT_ROOT /
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
# 2. LOAD ALIGNED LONG TABLE
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
    "Row",
    "Col",
    "X",
    "Y",
    "Spatial_Fold",
    "Model_Fold",
    "Role",
    "Cluster_k9",
    "Aligned_cluster_k9"
}


if not required_columns.issubset(
    df.columns
):

    raise RuntimeError(
        "Input aligned-label file is missing required columns."
    )


assert df[
    "Aligned_cluster_k9"
].between(
    0,
    K_FINE - 1
).all()


print("=" * 100)
print("k=9 CROSS-FOLD CONFUSION / META-DOMAIN ANALYSIS")
print("=" * 100)

print(
    "Rows:",
    len(
        df
    )
)

print(
    "Unique Cell_ID:",
    df[
        "Cell_ID"
    ].nunique()
)

print(
    "Models represented:",
    sorted(
        df[
            "Model_Fold"
        ]
        .unique()
        .tolist()
    )
)


# ============================================================
# 3. BUILD CROSS-MODEL PAIRS ON IDENTICAL CELLS
#
# Each pair of model predictions available for the same Cell_ID
# contributes one observation to the confusion structure.
# ============================================================

pair_records = []


for cell_id, group in df.groupby(
    "Cell_ID"
):

    if len(
        group
    ) < 2:

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

        label_a = int(
            a.Aligned_cluster_k9
        )

        label_b = int(
            b.Aligned_cluster_k9
        )


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
                label_a,

            "Label_B":
                label_b,

            "Agreement":
                int(
                    label_a
                    ==
                    label_b
                )
        })


pair_df = pd.DataFrame(
    pair_records
)


if len(
    pair_df
) == 0:

    raise RuntimeError(
        "No shared-cell model pairs were found."
    )


pair_file = (
    OUTPUT_DIR /
    "PBMMAE_k9_shared_cell_model_pairs.csv"
)


pair_df.to_csv(
    pair_file,
    index=False
)


print(
    "\nCross-model same-cell pairs:",
    len(
        pair_df
    )
)

print(
    "Raw pairwise agreement fraction:",
    round(
        pair_df[
            "Agreement"
        ].mean(),
        4
    )
)


# ============================================================
# 4. DIRECTED CONFUSION COUNTS
#
# confusion[i,j] =
#   number of same-cell prediction pairs involving aligned
#   cluster i in one model and cluster j in another model.
#
# We add both directions so the result is symmetric in support.
# ============================================================

confusion = np.zeros(
    (
        K_FINE,
        K_FINE
    ),
    dtype=np.float64
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
    ] += 1.0

    confusion[
        j,
        i
    ] += 1.0


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
    "PBMMAE_k9_crossfold_confusion_counts.csv"
)


confusion_df.to_csv(
    confusion_file
)


# ============================================================
# 5. NORMALIZED SUBSTITUTION MATRIX
#
# For each cluster i:
#   normalize its confusion row by all comparisons involving i.
#
# Then symmetrize:
#
#   S_ij = 0.5 * (P_ij + P_ji)
#
# High S_ij means clusters i and j frequently substitute for
# one another on identical geographic cells.
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

        where=
            row_sums != 0
    )


substitution = (
    0.5 *
    (
        directional
        +
        directional.T
    )
)


# Do not use self-agreement as inter-cluster similarity.

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
    "PBMMAE_k9_crossfold_substitution_similarity.csv"
)


substitution_df.to_csv(
    substitution_file
)


# ============================================================
# 6. EDGE LIST
#
# Off-diagonal cluster confusion strength.
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
    "PBMMAE_k9_cluster_confusion_graph_edges.csv"
)


edge_df.to_csv(
    edge_file,
    index=False
)


# ============================================================
# 7. HIERARCHICAL CLUSTERING OF FINE DOMAINS
#
# distance = 1 - substitution similarity
#
# This groups clusters that frequently substitute for each
# other across independently trained models.
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
    "PBMMAE_k9_metadomain_hierarchical_linkage.csv"
)


linkage_df.to_csv(
    linkage_file,
    index=False
)


# ============================================================
# 8. AGREEMENT FUNCTION FOR A GIVEN META-DOMAIN MAPPING
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
    # Cell-wise majority agreement
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
            len(
                winners
            )
            >
            1
        )


        if tied:

            consensus = -1

        else:

            consensus = int(
                winners[0]
            )


        cell_records.append({

            "Cell_ID":
                cell_id,

            "N_models":
                n_models,

            "Consensus_meta_domain":
                consensus,

            "Tie":
                tied,

            "Agreement_fraction":
                float(
                    max_count /
                    n_models
                ),

            "Full_agreement":
                bool(
                    max_count
                    ==
                    n_models
                )
        })


    cell_df = pd.DataFrame(
        cell_records
    )


    # --------------------------------------------------------
    # Pairwise model agreement on shared cells
    # --------------------------------------------------------

    pair_agreements = []

    pair_aris = []

    pair_nmis = []


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


        if len(
            shared
        ) == 0:

            continue


        y_a = shared[
            "Meta_domain_A"
        ].to_numpy()


        y_b = shared[
            "Meta_domain_B"
        ].to_numpy()


        pair_agreements.append(
            np.mean(
                y_a
                ==
                y_b
            )
        )


        pair_aris.append(
            adjusted_rand_score(
                y_a,
                y_b
            )
        )


        pair_nmis.append(
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
                    pair_agreements
                )
            ),

        "Mean_pairwise_ARI":
            float(
                np.mean(
                    pair_aris
                )
            ),

        "Mean_pairwise_NMI":
            float(
                np.mean(
                    pair_nmis
                )
            ),

        "Cell_table":
            cell_df
    }


# ============================================================
# 9. TEST META-DOMAIN COUNTS 2...8
#
# This is an exploratory hierarchical stability analysis.
# It does not replace the frozen outer-fold evaluation.
# ============================================================

candidate_records = []

candidate_mappings = {}


for n_meta in range(
    2,
    K_FINE
):

    cluster_membership = fcluster(

        Z,

        t=n_meta,

        criterion="maxclust"
    )


    # Convert 1-based scipy labels to 0-based.

    cluster_membership = (
        cluster_membership
        -
        1
    )


    fine_to_meta = {

        fine_cluster:
            int(
                cluster_membership[
                    fine_cluster
                ]
            )

        for fine_cluster in range(
            K_FINE
        )
    }


    candidate_mappings[
        n_meta
    ] = fine_to_meta


    result = evaluate_mapping(

        fine_to_meta,

        n_meta
    )


    candidate_records.append({

        "N_meta_domains":
            n_meta,

        "Mean_cell_agreement":
            result[
                "Mean_cell_agreement"
            ],

        "Median_cell_agreement":
            result[
                "Median_cell_agreement"
            ],

        "Full_agreement_fraction":
            result[
                "Full_agreement_fraction"
            ],

        "Tie_fraction":
            result[
                "Tie_fraction"
            ],

        "Mean_pairwise_label_accuracy":
            result[
                "Mean_pairwise_label_accuracy"
            ],

        "Mean_pairwise_ARI":
            result[
                "Mean_pairwise_ARI"
            ],

        "Mean_pairwise_NMI":
            result[
                "Mean_pairwise_NMI"
            ]
    })


candidate_df = pd.DataFrame(
    candidate_records
)


candidate_file = (
    OUTPUT_DIR /
    "PBMMAE_k9_metadomain_candidate_agreement.csv"
)


candidate_df.to_csv(
    candidate_file,
    index=False
)


# ============================================================
# 10. SAVE FINE->META MAPPING TABLES
# ============================================================

mapping_records = []


for n_meta, mapping in (
    candidate_mappings.items()
):

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


mapping_df = pd.DataFrame(
    mapping_records
)


mapping_file = (
    OUTPUT_DIR /
    "PBMMAE_k9_fine_to_metadomain_mappings.csv"
)


mapping_df.to_csv(
    mapping_file,
    index=False
)


# ============================================================
# 11. RANK CANDIDATES
#
# No automatic "winner" is imposed.
# We display reproducibility trade-offs.
# ============================================================

candidate_df[
    "Agreement_gain_vs_k9"
] = (

    candidate_df[
        "Mean_cell_agreement"
    ]
    -
    0.653
)


candidate_df[
    "Full_agreement_gain_vs_k9"
] = (

    candidate_df[
        "Full_agreement_fraction"
    ]
    -
    0.227
)


candidate_df.to_csv(
    candidate_file,
    index=False
)


# ============================================================
# 12. PRINT TOP CONFUSION EDGES
# ============================================================

print(
    "\n" +
    "=" * 100
)

print(
    "STRONGEST FINE-CLUSTER SUBSTITUTIONS"
)

print(
    "=" * 100
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


# ============================================================
# 13. PRINT META-DOMAIN CANDIDATE TABLE
# ============================================================

print(
    "\n" +
    "=" * 120
)

print(
    "META-DOMAIN REPRODUCIBILITY CANDIDATES"
)

print(
    "=" * 120
)


print(

    candidate_df
    .round(
        {
            "Mean_cell_agreement": 4,

            "Median_cell_agreement": 4,

            "Full_agreement_fraction": 4,

            "Tie_fraction": 4,

            "Mean_pairwise_label_accuracy": 4,

            "Mean_pairwise_ARI": 4,

            "Mean_pairwise_NMI": 4,

            "Agreement_gain_vs_k9": 4,

            "Full_agreement_gain_vs_k9": 4
        }
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 14. PRINT MAPPINGS
# ============================================================

print(
    "\n" +
    "=" * 100
)

print(
    "FINE k=9 -> META-DOMAIN MAPPINGS"
)

print(
    "=" * 100
)


for n_meta in range(
    2,
    K_FINE
):

    mapping = candidate_mappings[
        n_meta
    ]


    print(
        f"\n{n_meta} meta-domains:"
    )


    groups = {}


    for fine_cluster, meta_domain in (
        mapping.items()
    ):

        groups.setdefault(
            meta_domain,
            []
        ).append(
            fine_cluster
        )


    for meta_domain in sorted(
        groups
    ):

        print(
            f"  M{meta_domain}: "
            f"{groups[meta_domain]}"
        )


# ============================================================
# 15. FINAL OUTPUT
# ============================================================

print(
    "\n" +
    "=" * 100
)

print(
    "k=9 META-DOMAIN ANALYSIS COMPLETED"
)

print(
    "=" * 100
)


print(
    "\nCandidate agreement table:"
)

print(
    candidate_file
)


print(
    "\nFine-to-meta mappings:"
)

print(
    mapping_file
)


print(
    "\nCluster-confusion edges:"
)

print(
    edge_file
)


print(
    "\nSubstitution similarity matrix:"
)

print(
    substitution_file
)


print(
    "\nHierarchical linkage:"
)

print(
    linkage_file
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