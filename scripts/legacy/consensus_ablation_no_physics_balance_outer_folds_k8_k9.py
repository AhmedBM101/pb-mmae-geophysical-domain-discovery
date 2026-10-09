import os
# ============================================================
# Paper 1 — PB-MMAE Ablation A consensus clustering
#
# Ablation:
#   No physics-balanced reconstruction loss
#
# Fixed evaluation protocol:
#   - outer folds 1, 2, 4, 5
#   - 13x13 patches
#   - fixed 64-D latent representations
#   - ONLY k = 8 and k = 9
#   - 50 single-init TRAIN KMeans runs
#   - TRAIN-only consensus clustering
#   - validation assigned by nearest TRAIN consensus centroid
#
# Purpose:
#   Test whether removing equal physics-family reconstruction
#   weighting reduces clustering stability and geographic
#   transfer.
# ============================================================

import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering
)

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

from scipy.spatial.distance import cdist


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(os.environ["PBMMAE_PROJECT_ROOT"]).expanduser().resolve()

LATENT_ROOT = (
    PROJECT_ROOT /
    "03_outputs" /
    "models" /
    "PBMMAE" /
    "ablations" /
    "A_NoPhysicsBalancedLoss_13x13" /
    "representation_evaluation"
)

OUTPUT_ROOT = (
    PROJECT_ROOT /
    "03_outputs" /
    "models" /
    "PBMMAE" /
    "ablations" /
    "A_NoPhysicsBalancedLoss_13x13" /
    "outer_fold_consensus_k8_k9"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

FOLDS = [
    1,
    2,
    4,
    5
]

K_VALUES = [
    8,
    9
]

N_RUNS = 50

BASE_SEED = 42

PAC_LOW = 0.10
PAC_HIGH = 0.90


# ============================================================
# 2. INITIALIZATION STABILITY
# ============================================================

def pairwise_stability(
    labels_runs
):

    ari_values = []
    nmi_values = []


    for i, j in combinations(
        range(
            labels_runs.shape[0]
        ),
        2
    ):

        ari_values.append(
            adjusted_rand_score(
                labels_runs[i],
                labels_runs[j]
            )
        )

        nmi_values.append(
            normalized_mutual_info_score(
                labels_runs[i],
                labels_runs[j]
            )
        )


    return (
        np.asarray(
            ari_values,
            dtype=float
        ),
        np.asarray(
            nmi_values,
            dtype=float
        )
    )


# ============================================================
# 3. COASSOCIATION MATRIX
# ============================================================

def build_coassociation(
    labels_runs
):

    n_runs, n_samples = labels_runs.shape


    coassoc = np.zeros(
        (
            n_samples,
            n_samples
        ),
        dtype=np.float64
    )


    for labels in labels_runs:

        same = (
            labels[:, None]
            ==
            labels[None, :]
        )

        coassoc += same.astype(
            np.float64
        )


    coassoc /= float(
        n_runs
    )


    return coassoc


# ============================================================
# 4. PAC
# ============================================================

def calculate_pac(
    coassoc,
    low=0.10,
    high=0.90
):

    upper = coassoc[
        np.triu_indices_from(
            coassoc,
            k=1
        )
    ]


    if upper.size == 0:

        return np.nan


    ambiguous = (
        (upper > low)
        &
        (upper < high)
    )


    return float(
        np.mean(
            ambiguous
        )
    )


# ============================================================
# 5. WITHIN-CLUSTER CONSENSUS
# ============================================================

def within_cluster_consensus(
    coassoc,
    labels
):

    values = []


    for cluster_id in np.unique(
        labels
    ):

        idx = np.where(
            labels == cluster_id
        )[0]


        if len(
            idx
        ) < 2:

            continue


        block = coassoc[
            np.ix_(
                idx,
                idx
            )
        ]


        upper = block[
            np.triu_indices_from(
                block,
                k=1
            )
        ]


        if upper.size > 0:

            values.extend(
                upper.tolist()
            )


    if len(
        values
    ) == 0:

        return np.nan


    return float(
        np.mean(
            values
        )
    )


# ============================================================
# 6. CONSENSUS LABELS
# ============================================================

def consensus_labels_from_coassociation(
    coassoc,
    k
):

    distance = (
        1.0
        -
        coassoc
    )


    np.fill_diagonal(
        distance,
        0.0
    )


    model = AgglomerativeClustering(

        n_clusters=k,

        metric="precomputed",

        linkage="average"
    )


    labels = model.fit_predict(
        distance
    )


    return labels.astype(
        int
    )


# ============================================================
# 7. CENTROIDS
# ============================================================

def calculate_centroids(
    X,
    labels,
    k
):

    centroids = np.zeros(
        (
            k,
            X.shape[1]
        ),
        dtype=np.float64
    )


    for cluster_id in range(
        k
    ):

        idx = np.where(
            labels == cluster_id
        )[0]


        if len(
            idx
        ) == 0:

            raise RuntimeError(
                f"Empty TRAIN consensus cluster {cluster_id}"
            )


        centroids[
            cluster_id
        ] = X[
            idx
        ].mean(
            axis=0
        )


    return centroids


# ============================================================
# 8. VALIDATION ASSIGNMENT
# ============================================================

def assign_validation(
    X_val,
    centroids
):

    distances = cdist(

        X_val,

        centroids,

        metric="euclidean"
    )


    return np.argmin(
        distances,
        axis=1
    ).astype(
        int
    )


# ============================================================
# 9. MAIN EVALUATION LOOP
# ============================================================

records = []


for fold in FOLDS:

    fold_latent_dir = (
        LATENT_ROOT /
        f"Fold{fold}"
    )


    train_file = (
        fold_latent_dir /
        f"AblationA_Fold{fold}_13x13_TRAIN_latent64.npy"
    )


    validation_file = (
        fold_latent_dir /
        f"AblationA_Fold{fold}_13x13_VALIDATION_latent64.npy"
    )


    for path in [
        train_file,
        validation_file
    ]:

        if not path.exists():

            raise FileNotFoundError(
                path
            )


    X_train = np.load(
        train_file
    ).astype(
        np.float64
    )


    X_val = np.load(
        validation_file
    ).astype(
        np.float64
    )


    print(
        "\n" +
        "=" * 100
    )

    print(
        f"ABLATION A — FOLD {fold}"
    )

    print(
        "=" * 100
    )

    print(
        "TRAIN latent:",
        X_train.shape
    )

    print(
        "VALIDATION latent:",
        X_val.shape
    )


    for k in K_VALUES:

        print(
            "\n" +
            "-" * 80
        )

        print(
            f"Fold {fold} | k={k}"
        )

        print(
            "-" * 80
        )


        fold_k_dir = (
            OUTPUT_ROOT /
            f"Fold{fold}" /
            f"k{k}"
        )


        fold_k_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        # ----------------------------------------------------
        # 50 independent TRAIN KMeans runs
        # ----------------------------------------------------

        labels_runs = []


        for run_id in range(
            N_RUNS
        ):

            seed = (
                BASE_SEED
                +
                fold * 1000
                +
                run_id
            )


            km = KMeans(

                n_clusters=k,

                init="k-means++",

                n_init=1,

                max_iter=500,

                random_state=seed
            )


            labels = km.fit_predict(
                X_train
            )


            labels_runs.append(
                labels.astype(
                    np.int32
                )
            )


        labels_runs = np.stack(
            labels_runs,
            axis=0
        )


        np.save(

            fold_k_dir /
            f"Fold{fold}_k{k}_50_single_init_TRAIN_labels.npy",

            labels_runs
        )


        # ----------------------------------------------------
        # Stability
        # ----------------------------------------------------

        ari_values, nmi_values = pairwise_stability(
            labels_runs
        )


        # ----------------------------------------------------
        # Coassociation / consensus
        # ----------------------------------------------------

        coassoc = build_coassociation(
            labels_runs
        )


        np.save(

            fold_k_dir /
            f"Fold{fold}_k{k}_TRAIN_coassociation.npy",

            coassoc
        )


        pac = calculate_pac(
            coassoc,
            PAC_LOW,
            PAC_HIGH
        )


        train_consensus = consensus_labels_from_coassociation(
            coassoc,
            k
        )


        np.save(

            fold_k_dir /
            f"Fold{fold}_k{k}_consensus_TRAIN_labels.npy",

            train_consensus
        )


        # ----------------------------------------------------
        # Centroids / validation
        # ----------------------------------------------------

        centroids = calculate_centroids(
            X_train,
            train_consensus,
            k
        )


        np.save(

            fold_k_dir /
            f"Fold{fold}_k{k}_consensus_centroids.npy",

            centroids
        )


        val_labels = assign_validation(
            X_val,
            centroids
        )


        np.save(

            fold_k_dir /
            f"Fold{fold}_k{k}_consensus_VALIDATION_labels.npy",

            val_labels
        )


        # ----------------------------------------------------
        # Internal metrics
        # ----------------------------------------------------

        within_consensus = within_cluster_consensus(
            coassoc,
            train_consensus
        )


        silhouette = silhouette_score(
            X_train,
            train_consensus
        )


        dbi = davies_bouldin_score(
            X_train,
            train_consensus
        )


        ch = calinski_harabasz_score(
            X_train,
            train_consensus
        )


        # ----------------------------------------------------
        # Occupancy
        # ----------------------------------------------------

        train_counts = np.bincount(
            train_consensus,
            minlength=k
        )


        val_counts = np.bincount(
            val_labels,
            minlength=k
        )


        train_occupied = int(
            np.sum(
                train_counts > 0
            )
        )


        val_occupied = int(
            np.sum(
                val_counts > 0
            )
        )


        val_empty = int(
            k
            -
            val_occupied
        )


        val_nonzero = val_counts[
            val_counts > 0
        ]


        if len(
            val_nonzero
        ) > 0:

            min_val_nonzero = int(
                val_nonzero.min()
            )

        else:

            min_val_nonzero = 0


        train_dominant_fraction = float(
            train_counts.max()
            /
            len(
                train_consensus
            )
        )


        val_dominant_fraction = float(
            val_counts.max()
            /
            len(
                val_labels
            )
        )


        occupancy_df = pd.DataFrame({

            "Cluster":
                np.arange(
                    k
                ),

            "TRAIN_count":
                train_counts,

            "VALIDATION_count":
                val_counts
        })


        occupancy_df.to_csv(

            fold_k_dir /
            f"Fold{fold}_k{k}_occupancy.csv",

            index=False
        )


        # ----------------------------------------------------
        # Record
        # ----------------------------------------------------

        record = {

            "Fold":
                fold,

            "Ablation":
                "NoPhysicsBalancedLoss",

            "Patch_size":
                13,

            "k":
                k,

            "Train_N":
                len(
                    X_train
                ),

            "Validation_N":
                len(
                    X_val
                ),

            "N_KMeans_runs":
                N_RUNS,

            "ARI_mean":
                float(
                    np.mean(
                        ari_values
                    )
                ),

            "ARI_SD":
                float(
                    np.std(
                        ari_values,
                        ddof=1
                    )
                ),

            "ARI_min":
                float(
                    np.min(
                        ari_values
                    )
                ),

            "ARI_median":
                float(
                    np.median(
                        ari_values
                    )
                ),

            "NMI_mean":
                float(
                    np.mean(
                        nmi_values
                    )
                ),

            "NMI_SD":
                float(
                    np.std(
                        nmi_values,
                        ddof=1
                    )
                ),

            "NMI_min":
                float(
                    np.min(
                        nmi_values
                    )
                ),

            "NMI_median":
                float(
                    np.median(
                        nmi_values
                    )
                ),

            "PAC_0.1_0.9":
                pac,

            "Within_cluster_consensus":
                within_consensus,

            "Silhouette_TRAIN":
                silhouette,

            "DBI_TRAIN":
                dbi,

            "CH_TRAIN":
                ch,

            "TRAIN_occupied_clusters":
                train_occupied,

            "VALIDATION_occupied_clusters":
                val_occupied,

            "VALIDATION_occupancy_fraction":
                float(
                    val_occupied
                    /
                    k
                ),

            "VALIDATION_empty_clusters":
                val_empty,

            "VALIDATION_min_nonzero_cluster":
                min_val_nonzero,

            "TRAIN_dominant_fraction":
                train_dominant_fraction,

            "VALIDATION_dominant_fraction":
                val_dominant_fraction
        }


        records.append(
            record
        )


        print(
            "ARI mean:",
            round(
                record[
                    "ARI_mean"
                ],
                4
            )
        )

        print(
            "NMI mean:",
            round(
                record[
                    "NMI_mean"
                ],
                4
            )
        )

        print(
            "PAC:",
            round(
                record[
                    "PAC_0.1_0.9"
                ],
                4
            )
        )

        print(
            "Within consensus:",
            round(
                record[
                    "Within_cluster_consensus"
                ],
                4
            )
        )

        print(
            "Silhouette:",
            round(
                record[
                    "Silhouette_TRAIN"
                ],
                4
            )
        )

        print(
            "DBI:",
            round(
                record[
                    "DBI_TRAIN"
                ],
                4
            )
        )

        print(
            "VAL occupied:",
            f"{val_occupied}/{k}"
        )

        print(
            "VAL dominant fraction:",
            round(
                val_dominant_fraction,
                4
            )
        )


# ============================================================
# 10. SAVE DETAILED RESULTS
# ============================================================

results_df = pd.DataFrame(
    records
)


detailed_file = (
    OUTPUT_ROOT /
    "PBMMAE_AblationA_NoPhysicsBalance_outer_folds_consensus_k8_k9.csv"
)


results_df.to_csv(
    detailed_file,
    index=False
)


# ============================================================
# 11. SUMMARY BY k
# ============================================================

summary_records = []


for k in K_VALUES:

    subset = results_df[
        results_df[
            "k"
        ] == k
    ]


    summary_records.append({

        "Ablation":
            "NoPhysicsBalancedLoss",

        "Patch_size":
            13,

        "k":
            k,

        "N_outer_folds":
            len(
                subset
            ),

        "Mean_ARI":
            subset[
                "ARI_mean"
            ].mean(),

        "SD_ARI":
            subset[
                "ARI_mean"
            ].std(
                ddof=1
            ),

        "Mean_NMI":
            subset[
                "NMI_mean"
            ].mean(),

        "SD_NMI":
            subset[
                "NMI_mean"
            ].std(
                ddof=1
            ),

        "Mean_PAC":
            subset[
                "PAC_0.1_0.9"
            ].mean(),

        "SD_PAC":
            subset[
                "PAC_0.1_0.9"
            ].std(
                ddof=1
            ),

        "Mean_within_cluster_consensus":
            subset[
                "Within_cluster_consensus"
            ].mean(),

        "Mean_Silhouette":
            subset[
                "Silhouette_TRAIN"
            ].mean(),

        "Mean_DBI":
            subset[
                "DBI_TRAIN"
            ].mean(),

        "Mean_CH":
            subset[
                "CH_TRAIN"
            ].mean(),

        "Mean_validation_occupancy_fraction":
            subset[
                "VALIDATION_occupancy_fraction"
            ].mean(),

        "Total_validation_empty_clusters":
            int(
                subset[
                    "VALIDATION_empty_clusters"
                ].sum()
            ),

        "Folds_with_full_validation_occupancy":
            int(
                np.sum(
                    subset[
                        "VALIDATION_empty_clusters"
                    ] == 0
                )
            ),

        "Mean_validation_dominant_fraction":
            subset[
                "VALIDATION_dominant_fraction"
            ].mean(),

        "Worst_validation_dominant_fraction":
            subset[
                "VALIDATION_dominant_fraction"
            ].max(),

        "Minimum_validation_nonzero_cluster":
            subset[
                "VALIDATION_min_nonzero_cluster"
            ].min()
    })


summary_df = pd.DataFrame(
    summary_records
)


summary_file = (
    OUTPUT_ROOT /
    "PBMMAE_AblationA_NoPhysicsBalance_outer_folds_consensus_k8_k9_summary.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# 12. PRINT FINAL SUMMARY
# ============================================================

print(
    "\n\n" +
    "=" * 120
)

print(
    "ABLATION A — CONSENSUS CLUSTERING SUMMARY"
)

print(
    "=" * 120
)


print(
    summary_df
    .round(
        4
    )
    .to_string(
        index=False
    )
)


print(
    "\nDetailed output:"
)

print(
    detailed_file
)


print(
    "\nSummary output:"
)

print(
    summary_file
)


print(
    "\nIMPORTANT:"
)

print(
    "Only frozen k=8 and k=9 were evaluated."
)

print(
    "Consensus construction used TRAIN representations only."
)

print(
    "Validation labels were assigned by nearest TRAIN "
    "consensus centroid."
)


print(
    "=" * 120
)