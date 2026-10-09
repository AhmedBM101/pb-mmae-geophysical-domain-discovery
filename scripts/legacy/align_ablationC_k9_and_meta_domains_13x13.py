import os
# ============================================================
# Paper 1 — PB-MMAE Ablation C
# Cross-fold k=9 alignment + basin-scale meta-domain analysis
#
# Ablation:
#   MAG_LD removed (13-channel configuration)
#
# Purpose:
#   1. Combine TRAIN consensus labels and VALIDATION
#      nearest-centroid labels for each outer-fold model.
#   2. Align independently trained k=9 label systems using
#      shared Cell_ID values and Hungarian assignment.
#   3. Quantify fine-domain agreement across outer models.
#   4. Build cross-model substitution/confusion structure.
#   5. Test hierarchical meta-domain solutions from 2 to 8.
#
# IMPORTANT:
#   - Latent coordinates are NEVER compared across models.
#   - Alignment uses same-cell label correspondences only.
#   - Fold 1 is the reference label system.
# ============================================================

import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

from pathlib import Path
from itertools import combinations
from collections import Counter

import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import (
    linkage,
    fcluster
)

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score
)


# ============================================================
# 1. CONFIGURATION
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
    "ablations" /
    "C_Remove_MAG_LD_13x13" /
    "outer_fold_consensus_k8_k9"
)

OUTPUT_ROOT = (
    PROJECT_ROOT /
    "03_outputs" /
    "models" /
    "PBMMAE" /
    "ablations" /
    "C_Remove_MAG_LD_13x13" /
    "cross_fold_alignment_k9"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

FOLDS = [1, 2, 4, 5]

REFERENCE_FOLD = 1

K = 9


# ============================================================
# 2. FIND PATCH METADATA ROBUSTLY
#
# We search inside each Fold_X directory because exact
# metadata filename conventions can differ slightly.
# ============================================================

def find_metadata_file(
    fold,
    role
):

    fold_dir = (
        PATCH_ROOT /
        f"Fold_{fold}"
    )

    if not fold_dir.exists():
        raise FileNotFoundError(
            fold_dir
        )

    role_upper = role.upper()

    candidates = []

    for path in fold_dir.glob("*.csv"):

        name_upper = path.name.upper()

        if (
            role_upper in name_upper
            and
            "13X13" in name_upper
        ):
            candidates.append(
                path
            )

    if len(candidates) == 0:

        # Broader fallback.
        for path in fold_dir.glob("*.csv"):

            name_upper = path.name.upper()

            if role_upper in name_upper:
                candidates.append(
                    path
                )

    if len(candidates) == 0:

        raise FileNotFoundError(
            f"No {role} metadata CSV found in {fold_dir}"
        )

    # Prefer files explicitly mentioning metadata.
    metadata_named = [
        p
        for p in candidates
        if "META" in p.name.upper()
    ]

    if len(metadata_named) == 1:
        return metadata_named[0]

    if len(metadata_named) > 1:
        candidates = metadata_named

    # Inspect columns and retain files containing Cell_ID.
    valid = []

    for path in candidates:

        try:
            test = pd.read_csv(
                path,
                nrows=5
            )

            if "Cell_ID" in test.columns:
                valid.append(
                    path
                )

        except Exception:
            pass

    if len(valid) == 1:
        return valid[0]

    if len(valid) == 0:

        raise RuntimeError(
            f"Candidate {role} files found for Fold {fold}, "
            "but none contains Cell_ID:\n"
            +
            "\n".join(
                str(p)
                for p in candidates
            )
        )

    raise RuntimeError(
        f"Multiple possible {role} metadata files for Fold {fold}:\n"
        +
        "\n".join(
            str(p)
            for p in valid
        )
    )


# ============================================================
# 3. LOAD ONE MODEL'S LABELLED CELLS
# ============================================================

def load_fold_assignments(
    fold
):

    train_meta_file = find_metadata_file(
        fold,
        "TRAIN"
    )

    val_meta_file = find_metadata_file(
        fold,
        "VALIDATION"
    )

    train_meta = pd.read_csv(
        train_meta_file
    )

    val_meta = pd.read_csv(
        val_meta_file
    )

    required_columns = [
        "Cell_ID",
        "Row",
        "Col",
        "X",
        "Y"
    ]

    for column in required_columns:

        if column not in train_meta.columns:
            raise RuntimeError(
                f"{column} absent from {train_meta_file}"
            )

        if column not in val_meta.columns:
            raise RuntimeError(
                f"{column} absent from {val_meta_file}"
            )


    label_dir = (
        CONSENSUS_ROOT /
        f"Fold{fold}" /
        f"k{K}"
    )

    train_label_file = (
        label_dir /
        f"Fold{fold}_k{K}_consensus_TRAIN_labels.npy"
    )

    val_label_file = (
        label_dir /
        f"Fold{fold}_k{K}_consensus_VALIDATION_labels.npy"
    )

    for path in [
        train_label_file,
        val_label_file
    ]:

        if not path.exists():
            raise FileNotFoundError(
                path
            )


    train_labels = np.load(
        train_label_file
    ).astype(
        int
    )

    val_labels = np.load(
        val_label_file
    ).astype(
        int
    )


    if len(train_meta) != len(train_labels):

        raise RuntimeError(
            f"Fold {fold}: TRAIN metadata rows "
            f"{len(train_meta)} != labels {len(train_labels)}"
        )

    if len(val_meta) != len(val_labels):

        raise RuntimeError(
            f"Fold {fold}: VALIDATION metadata rows "
            f"{len(val_meta)} != labels {len(val_labels)}"
        )


    train_df = train_meta[
        required_columns
    ].copy()

    train_df["Original_label"] = (
        train_labels
    )

    train_df["Role"] = "TRAIN"


    val_df = val_meta[
        required_columns
    ].copy()

    val_df["Original_label"] = (
        val_labels
    )

    val_df["Role"] = "VALIDATION"


    combined = pd.concat(
        [
            train_df,
            val_df
        ],
        ignore_index=True
    )


    if combined[
        "Cell_ID"
    ].duplicated().any():

        duplicated = combined.loc[
            combined[
                "Cell_ID"
            ].duplicated(),
            "Cell_ID"
        ].tolist()

        raise RuntimeError(
            f"Fold {fold}: duplicate Cell_ID values "
            f"within model assignments: "
            f"{duplicated[:10]}"
        )


    combined["Model_Fold"] = fold


    print(
        f"Fold {fold}:"
        f" TRAIN={len(train_df)},"
        f" VAL={len(val_df)},"
        f" total assigned={len(combined)}"
    )

    print(
        "  TRAIN metadata:",
        train_meta_file.name
    )

    print(
        "  VAL metadata:",
        val_meta_file.name
    )


    return combined


# ============================================================
# 4. LOAD ALL OUTER-MODEL ASSIGNMENTS
# ============================================================

fold_data = {}

for fold in FOLDS:

    fold_data[
        fold
    ] = load_fold_assignments(
        fold
    )


# ============================================================
# 5. GEOGRAPHIC IDENTITY CHECK
# ============================================================

identity_records = []

for fold_a, fold_b in combinations(
    FOLDS,
    2
):

    a = fold_data[
        fold_a
    ][
        [
            "Cell_ID",
            "Row",
            "Col",
            "X",
            "Y"
        ]
    ]

    b = fold_data[
        fold_b
    ][
        [
            "Cell_ID",
            "Row",
            "Col",
            "X",
            "Y"
        ]
    ]

    shared = a.merge(
        b,
        on="Cell_ID",
        suffixes=(
            "_A",
            "_B"
        )
    )

    row_match = np.all(
        shared[
            "Row_A"
        ].to_numpy()
        ==
        shared[
            "Row_B"
        ].to_numpy()
    )

    col_match = np.all(
        shared[
            "Col_A"
        ].to_numpy()
        ==
        shared[
            "Col_B"
        ].to_numpy()
    )

    x_match = np.allclose(
        shared[
            "X_A"
        ].to_numpy(),
        shared[
            "X_B"
        ].to_numpy(),
        rtol=0,
        atol=1e-6
    )

    y_match = np.allclose(
        shared[
            "Y_A"
        ].to_numpy(),
        shared[
            "Y_B"
        ].to_numpy(),
        rtol=0,
        atol=1e-6
    )

    identity_records.append({

        "Fold_A":
            fold_a,

        "Fold_B":
            fold_b,

        "Shared_cells":
            len(
                shared
            ),

        "Row_match":
            row_match,

        "Col_match":
            col_match,

        "X_match":
            x_match,

        "Y_match":
            y_match,

        "All_identity_checks":
            (
                row_match
                and
                col_match
                and
                x_match
                and
                y_match
            )
    })


identity_df = pd.DataFrame(
    identity_records
)

identity_file = (
    OUTPUT_ROOT /
    "AblationC_k9_pairwise_geographic_identity_checks.csv"
)

identity_df.to_csv(
    identity_file,
    index=False
)


if not identity_df[
    "All_identity_checks"
].all():

    raise RuntimeError(
        "Geographic identity check failed."
    )


# ============================================================
# 6. HUNGARIAN ALIGNMENT
#
# Fold 1 labels define the reference naming system.
# Alignment is based ONLY on shared Cell_ID classifications.
# ============================================================

def contingency_matrix(
    ref_labels,
    candidate_labels,
    k
):

    matrix = np.zeros(
        (
            k,
            k
        ),
        dtype=int
    )

    for ref_label, cand_label in zip(
        ref_labels,
        candidate_labels
    ):

        matrix[
            int(ref_label),
            int(cand_label)
        ] += 1

    return matrix


def hungarian_mapping_to_reference(
    reference_df,
    candidate_df,
    k
):

    shared = reference_df[
        [
            "Cell_ID",
            "Original_label"
        ]
    ].merge(

        candidate_df[
            [
                "Cell_ID",
                "Original_label"
            ]
        ],

        on="Cell_ID",

        suffixes=(
            "_REF",
            "_CAND"
        )
    )


    matrix = contingency_matrix(

        shared[
            "Original_label_REF"
        ].to_numpy(),

        shared[
            "Original_label_CAND"
        ].to_numpy(),

        k
    )


    # Maximize agreement.
    row_ind, col_ind = (
        linear_sum_assignment(
            -matrix
        )
    )


    # candidate label -> reference label
    mapping = {

        int(cand):
            int(ref)

        for ref, cand
        in zip(
            row_ind,
            col_ind
        )
    }


    aligned_candidate = np.array(
        [
            mapping[
                int(label)
            ]
            for label
            in shared[
                "Original_label_CAND"
            ].to_numpy()
        ],
        dtype=int
    )


    reference_labels = shared[
        "Original_label_REF"
    ].to_numpy(
        dtype=int
    )


    accuracy = float(
        np.mean(
            aligned_candidate
            ==
            reference_labels
        )
    )


    ari = adjusted_rand_score(
        reference_labels,
        aligned_candidate
    )

    nmi = normalized_mutual_info_score(
        reference_labels,
        aligned_candidate
    )


    return (
        mapping,
        matrix,
        len(shared),
        accuracy,
        ari,
        nmi
    )


reference_df = fold_data[
    REFERENCE_FOLD
].copy()

alignment_records = []

alignment_maps = {

    REFERENCE_FOLD: {
        label: label
        for label in range(
            K
        )
    }
}


for fold in FOLDS:

    if fold == REFERENCE_FOLD:

        continue


    (
        mapping,
        matrix,
        n_shared,
        accuracy,
        ari,
        nmi
    ) = hungarian_mapping_to_reference(

        reference_df,

        fold_data[
            fold
        ],

        K
    )


    alignment_maps[
        fold
    ] = mapping


    alignment_records.append({

        "Reference_fold":
            REFERENCE_FOLD,

        "Candidate_fold":
            fold,

        "Shared_cells":
            n_shared,

        "Aligned_accuracy":
            accuracy,

        "ARI":
            ari,

        "NMI":
            nmi
    })


    pd.DataFrame(
        matrix,
        index=[
            f"Ref_C{i}"
            for i in range(
                K
            )
        ],
        columns=[
            f"Fold{fold}_C{i}"
            for i in range(
                K
            )
        ]
    ).to_csv(

        OUTPUT_ROOT /
        f"Fold1_vs_Fold{fold}_k9_contingency_matrix.csv"
    )


    pd.DataFrame(
        [
            {
                "Candidate_label":
                    candidate,

                "Aligned_reference_label":
                    reference
            }

            for candidate, reference
            in sorted(
                mapping.items()
            )
        ]
    ).to_csv(

        OUTPUT_ROOT /
        f"Fold{fold}_to_Fold1_k9_Hungarian_mapping.csv",

        index=False
    )


alignment_df = pd.DataFrame(
    alignment_records
)

alignment_file = (
    OUTPUT_ROOT /
    "AblationC_k9_Fold1_reference_alignment_summary.csv"
)

alignment_df.to_csv(
    alignment_file,
    index=False
)


# ============================================================
# 7. APPLY ALIGNMENT
# ============================================================

aligned_data = {}


for fold in FOLDS:

    df = fold_data[
        fold
    ].copy()

    mapping = alignment_maps[
        fold
    ]

    df[
        "Aligned_label"
    ] = df[
        "Original_label"
    ].map(
        mapping
    )

    if df[
        "Aligned_label"
    ].isna().any():

        raise RuntimeError(
            f"Unmapped labels in Fold {fold}"
        )

    df[
        "Aligned_label"
    ] = df[
        "Aligned_label"
    ].astype(
        int
    )

    aligned_data[
        fold
    ] = df


# ============================================================
# 8. GLOBAL CELL-LEVEL FINE-DOMAIN AGREEMENT
# ============================================================

all_cell_ids = sorted(
    set().union(
        *[
            set(
                aligned_data[
                    fold
                ][
                    "Cell_ID"
                ].tolist()
            )
            for fold in FOLDS
        ]
    )
)


cell_records = []


for cell_id in all_cell_ids:

    labels = []

    coordinate_record = None

    model_label_record = {}


    for fold in FOLDS:

        rows = aligned_data[
            fold
        ][
            aligned_data[
                fold
            ][
                "Cell_ID"
            ] == cell_id
        ]

        if len(rows) == 0:

            model_label_record[
                f"Fold{fold}_label"
            ] = np.nan

            continue


        row = rows.iloc[
            0
        ]

        label = int(
            row[
                "Aligned_label"
            ]
        )

        labels.append(
            label
        )

        model_label_record[
            f"Fold{fold}_label"
        ] = label


        if coordinate_record is None:

            coordinate_record = {

                "Row":
                    int(
                        row[
                            "Row"
                        ]
                    ),

                "Col":
                    int(
                        row[
                            "Col"
                        ]
                    ),

                "X":
                    float(
                        row[
                            "X"
                        ]
                    ),

                "Y":
                    float(
                        row[
                            "Y"
                        ]
                    )
            }


    support = len(
        labels
    )

    counts = Counter(
        labels
    )

    max_count = max(
        counts.values()
    )

    winners = sorted(
        [
            label
            for label, count
            in counts.items()
            if count == max_count
        ]
    )


    tied = (
        len(
            winners
        ) > 1
    )


    majority_label = (
        winners[0]
        if not tied
        else np.nan
    )


    agreement_fraction = (
        max_count
        /
        support
    )


    full_agreement = (
        len(
            counts
        ) == 1
    )


    record = {

        "Cell_ID":
            cell_id,

        **coordinate_record,

        "Model_support":
            support,

        "Majority_fine_label":
            majority_label,

        "Fine_tied":
            tied,

        "Fine_agreement_fraction":
            agreement_fraction,

        "Fine_full_agreement":
            full_agreement,

        **model_label_record
    }


    cell_records.append(
        record
    )


cell_df = pd.DataFrame(
    cell_records
)


cell_file = (
    OUTPUT_ROOT /
    "AblationC_k9_global_aligned_cell_agreement.csv"
)

cell_df.to_csv(
    cell_file,
    index=False
)


# ============================================================
# 9. SUPPORT-LEVEL SUMMARY
# ============================================================

support_records = []


for support in sorted(
    cell_df[
        "Model_support"
    ].unique()
):

    subset = cell_df[
        cell_df[
            "Model_support"
        ] == support
    ]


    support_records.append({

        "Model_support":
            int(
                support
            ),

        "N_cells":
            len(
                subset
            ),

        "Mean_agreement_fraction":
            subset[
                "Fine_agreement_fraction"
            ].mean(),

        "Median_agreement_fraction":
            subset[
                "Fine_agreement_fraction"
            ].median(),

        "Full_agreement_fraction":
            subset[
                "Fine_full_agreement"
            ].mean(),

        "Tie_fraction":
            subset[
                "Fine_tied"
            ].mean()
    })


support_df = pd.DataFrame(
    support_records
)


support_file = (
    OUTPUT_ROOT /
    "AblationC_k9_support_level_agreement_summary.csv"
)

support_df.to_csv(
    support_file,
    index=False
)


# ============================================================
# 10. CROSS-MODEL SUBSTITUTION MATRIX
#
# For every same Cell_ID classified by >=2 models, record
# every model-pair label pairing.
#
# This produces a symmetric fine-domain co-occurrence matrix.
# ============================================================

pair_counts = np.zeros(
    (
        K,
        K
    ),
    dtype=float
)


for _, row in cell_df.iterrows():

    labels = []

    for fold in FOLDS:

        value = row[
            f"Fold{fold}_label"
        ]

        if pd.notna(
            value
        ):

            labels.append(
                int(
                    value
                )
            )


    for a, b in combinations(
        labels,
        2
    ):

        pair_counts[
            a,
            b
        ] += 1

        pair_counts[
            b,
            a
        ] += 1


# ------------------------------------------------------------
# Normalize each row to conditional substitution frequencies.
# Then symmetrize.
# ------------------------------------------------------------

row_sums = pair_counts.sum(
    axis=1,
    keepdims=True
)

conditional = np.divide(

    pair_counts,

    row_sums,

    out=np.zeros_like(
        pair_counts,
        dtype=float
    ),

    where=(
        row_sums > 0
    )
)


substitution = (
    conditional
    +
    conditional.T
) / 2.0


np.fill_diagonal(
    substitution,
    1.0
)


substitution_df = pd.DataFrame(

    substitution,

    index=[
        f"C{i}"
        for i in range(
            K
        )
    ],

    columns=[
        f"C{i}"
        for i in range(
            K
        )
    ]
)


substitution_file = (
    OUTPUT_ROOT /
    "AblationC_k9_symmetric_substitution_matrix.csv"
)

substitution_df.to_csv(
    substitution_file
)


# ============================================================
# 11. STRONGEST OFF-DIAGONAL SUBSTITUTIONS
# ============================================================

substitution_records = []


for i in range(
    K
):

    for j in range(
        i + 1,
        K
    ):

        substitution_records.append({

            "Cluster_A":
                f"C{i}",

            "Cluster_B":
                f"C{j}",

            "Substitution_strength":
                substitution[
                    i,
                    j
                ]
        })


substitution_pairs_df = pd.DataFrame(
    substitution_records
).sort_values(

    "Substitution_strength",

    ascending=False
)


substitution_pairs_file = (
    OUTPUT_ROOT /
    "AblationC_k9_ranked_cluster_substitutions.csv"
)

substitution_pairs_df.to_csv(
    substitution_pairs_file,
    index=False
)


# ============================================================
# 12. HIERARCHICAL META-DOMAIN CLUSTERING
#
# Distance = 1 - substitution strength.
# Average-link hierarchical clustering.
# Candidate meta-domain counts = 2 ... 8.
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


condensed = []

for i in range(
    K
):

    for j in range(
        i + 1,
        K
    ):

        condensed.append(
            distance[
                i,
                j
            ]
        )


condensed = np.asarray(
    condensed,
    dtype=float
)


Z = linkage(
    condensed,
    method="average"
)


# ============================================================
# 13. EVALUATE META-DOMAIN CANDIDATES
# ============================================================

meta_candidate_records = []

meta_mappings = {}


for n_meta in range(
    2,
    9
):

    raw_groups = fcluster(
        Z,
        t=n_meta,
        criterion="maxclust"
    )

    # Convert to deterministic 0-based IDs.
    unique_groups = sorted(
        np.unique(
            raw_groups
        )
    )

    remap = {
        old:
            new
        for new, old
        in enumerate(
            unique_groups
        )
    }

    cluster_to_meta = {

        fine_cluster:
            remap[
                raw_groups[
                    fine_cluster
                ]
            ]

        for fine_cluster
        in range(
            K
        )
    }


    meta_mappings[
        n_meta
    ] = cluster_to_meta


    cell_agreements = []
    cell_full = []
    cell_ties = []


    for _, row in cell_df.iterrows():

        meta_labels = []

        for fold in FOLDS:

            value = row[
                f"Fold{fold}_label"
            ]

            if pd.notna(
                value
            ):

                fine_label = int(
                    value
                )

                meta_labels.append(
                    cluster_to_meta[
                        fine_label
                    ]
                )


        counts = Counter(
            meta_labels
        )

        max_count = max(
            counts.values()
        )

        winners = [
            label
            for label, count
            in counts.items()
            if count == max_count
        ]

        cell_agreements.append(
            max_count
            /
            len(
                meta_labels
            )
        )

        cell_full.append(
            len(
                counts
            ) == 1
        )

        cell_ties.append(
            len(
                winners
            ) > 1
        )


    # --------------------------------------------------------
    # Pairwise model agreement after fine -> meta mapping
    # --------------------------------------------------------

    pairwise_accuracies = []


    for fold_a, fold_b in combinations(
        FOLDS,
        2
    ):

        a = aligned_data[
            fold_a
        ][
            [
                "Cell_ID",
                "Aligned_label"
            ]
        ].copy()

        b = aligned_data[
            fold_b
        ][
            [
                "Cell_ID",
                "Aligned_label"
            ]
        ].copy()


        shared = a.merge(
            b,
            on="Cell_ID",
            suffixes=(
                "_A",
                "_B"
            )
        )


        meta_a = shared[
            "Aligned_label_A"
        ].map(
            cluster_to_meta
        ).to_numpy()

        meta_b = shared[
            "Aligned_label_B"
        ].map(
            cluster_to_meta
        ).to_numpy()


        pairwise_accuracies.append(
            np.mean(
                meta_a
                ==
                meta_b
            )
        )


    meta_candidate_records.append({

        "N_meta_domains":
            n_meta,

        "Mean_cell_agreement":
            np.mean(
                cell_agreements
            ),

        "Median_cell_agreement":
            np.median(
                cell_agreements
            ),

        "Full_agreement_fraction":
            np.mean(
                cell_full
            ),

        "Tie_fraction":
            np.mean(
                cell_ties
            ),

        "Mean_pairwise_model_accuracy":
            np.mean(
                pairwise_accuracies
            )
    })


meta_candidate_df = pd.DataFrame(
    meta_candidate_records
)


meta_candidate_file = (
    OUTPUT_ROOT /
    "AblationC_k9_meta_domain_candidate_stability_2_to_8.csv"
)

meta_candidate_df.to_csv(
    meta_candidate_file,
    index=False
)


# ============================================================
# 14. SAVE 4-META-DOMAIN MAPPING
#
# Four is NOT automatically promoted here.
# It is saved for direct comparison with the frozen
# 14-channel result.
# ============================================================

four_mapping = meta_mappings[
    4
]


four_mapping_df = pd.DataFrame(
    [
        {
            "Fine_cluster":
                f"C{fine}",

            "Meta_domain":
                f"M{meta}"
        }

        for fine, meta
        in sorted(
            four_mapping.items()
        )
    ]
)


four_mapping_file = (
    OUTPUT_ROOT /
    "AblationC_k9_4meta_domain_mapping.csv"
)


four_mapping_df.to_csv(
    four_mapping_file,
    index=False
)


# ============================================================
# 15. OVERALL FINE-DOMAIN SUMMARY
# ============================================================

overall_summary = pd.DataFrame(
    [{
        "Ablation":
            "Remove_MAG_LD",

        "Fine_k":
            K,

        "Total_unique_cells":
            len(
                cell_df
            ),

        "Mean_model_support":
            cell_df[
                "Model_support"
            ].mean(),

        "Min_model_support":
            cell_df[
                "Model_support"
            ].min(),

        "Max_model_support":
            cell_df[
                "Model_support"
            ].max(),

        "Mean_fine_agreement":
            cell_df[
                "Fine_agreement_fraction"
            ].mean(),

        "Median_fine_agreement":
            cell_df[
                "Fine_agreement_fraction"
            ].median(),

        "Fine_full_agreement_fraction":
            cell_df[
                "Fine_full_agreement"
            ].mean(),

        "Fine_tie_fraction":
            cell_df[
                "Fine_tied"
            ].mean(),

        "Fine_tied_cells":
            int(
                cell_df[
                    "Fine_tied"
                ].sum()
            )
    }]
)


overall_file = (
    OUTPUT_ROOT /
    "AblationC_k9_global_fine_domain_summary.csv"
)


overall_summary.to_csv(
    overall_file,
    index=False
)


# ============================================================
# 16. PRINT RESULTS
# ============================================================

print(
    "\n\n" +
    "=" * 120
)

print(
    "ABLATION C — k=9 CROSS-FOLD ALIGNMENT"
)

print(
    "=" * 120
)

print(
    alignment_df
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
    "GLOBAL FINE-DOMAIN AGREEMENT"
)

print(
    "=" * 120
)

print(
    overall_summary
    .round(
        4
    )
    .to_string(
        index=False
    )
)


print(
    "\nSupport-level agreement:"
)

print(
    support_df
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
    "STRONGEST FINE-DOMAIN SUBSTITUTIONS"
)

print(
    "=" * 120
)

print(
    substitution_pairs_df
    .head(
        12
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
    "META-DOMAIN STABILITY — 2 TO 8"
)

print(
    "=" * 120
)

print(
    meta_candidate_df
    .round(
        4
    )
    .to_string(
        index=False
    )
)


print(
    "\nCandidate 4-meta-domain mapping:"
)

print(
    four_mapping_df.to_string(
        index=False
    )
)


print(
    "\nOutputs saved in:"
)

print(
    OUTPUT_ROOT
)


print(
    "\nIMPORTANT:"
)

print(
    "Fold 1 is used only as a categorical reference for "
    "label naming."
)

print(
    "No latent coordinates were compared between "
    "independently trained models."
)

print(
    "The hierarchy is derived from same-cell cross-model "
    "label substitutions."
)

print(
    "=" * 120
)