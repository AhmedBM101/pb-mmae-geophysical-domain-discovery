import os
# ============================================================
# Paper 1 — Frozen PB-MMAE LinearFusion64
# outer-fold consensus evaluation for k = 8 and k = 9
#
# Outer evaluation folds:
#   1, 2, 4, 5
#
# Fold 3 was used for development and is not evaluated here.
#
# For each outer fold and each frozen candidate k in {8, 9}:
#   1. Load fixed TRAIN and held-out VALIDATION embeddings.
#   2. Run 50 independent single-init KMeans fits on TRAIN only.
#   3. Compute pairwise ARI and NMI across the 50 runs.
#   4. Construct the TRAIN co-association matrix.
#   5. Compute PAC using the interval [0.1, 0.9].
#   6. Derive consensus TRAIN labels using average-linkage
#      agglomerative clustering on distance = 1 - coassociation.
#   7. Compute consensus centroids in the fixed 64-D latent space.
#   8. Assign held-out VALIDATION to nearest TRAIN centroids.
#   9. Compute TRAIN internal metrics and validation occupancy.
#
# IMPORTANT:
#   - k is NOT searched on outer folds.
#   - Only development-fold candidates k=8 and k=9 are tested.
#   - Validation is NEVER used for training or consensus formation.
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


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(os.environ["PBMMAE_PROJECT_ROOT"]).expanduser().resolve()

MODEL_ROOT = (
    PROJECT_ROOT /
    "03_outputs" /
    "models" /
    "PBMMAE" /
    "OuterFolds_13x13_LinearFusion64"
)

REPRESENTATION_DIR = (
    MODEL_ROOT /
    "representation_evaluation"
)

CONSENSUS_DIR = (
    MODEL_ROOT /
    "outer_fold_consensus_k8_k9"
)

CONSENSUS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_CSV = (
    CONSENSUS_DIR /
    "PBMMAE_LINEAR64_outer_folds_consensus_k8_k9.csv"
)


# ============================================================
# 2. FROZEN EVALUATION CONFIGURATION
# ============================================================

OUTER_FOLDS = [
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

MAX_ITER = 1000
TOL = 1e-4

PAC_LOWER = 0.10
PAC_UPPER = 0.90

LATENT_DIM = 64


print("=" * 100)
print("PB-MMAE LINEAR-FUSION-64 — OUTER-FOLD CONSENSUS TEST")
print("=" * 100)

print("Outer folds :", OUTER_FOLDS)
print("Frozen k    :", K_VALUES)
print("Runs per k  :", N_RUNS)
print("Development fold excluded : 3")


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def cluster_counts(
    labels,
    k
):

    return np.bincount(
        labels,
        minlength=k
    )


def build_coassociation(
    label_runs
):

    """
    label_runs:
        shape = [n_runs, n_samples]

    C[i,j] =
        proportion of runs in which samples i and j
        were assigned to the same cluster.
    """

    n_runs, n_samples = (
        label_runs.shape
    )

    coassoc = np.zeros(
        (
            n_samples,
            n_samples
        ),
        dtype=np.float64
    )


    for run_id in range(
        n_runs
    ):

        labels = (
            label_runs[
                run_id
            ]
        )


        same_cluster = (
            labels[:, None]
            ==
            labels[None, :]
        )


        coassoc += (
            same_cluster.astype(
                np.float64
            )
        )


    coassoc /= float(
        n_runs
    )


    np.fill_diagonal(
        coassoc,
        1.0
    )


    return coassoc


def compute_pac(
    coassoc,
    lower=0.1,
    upper=0.9
):

    """
    Proportion of ambiguous clustering.

    Lower PAC indicates stronger consensus.
    """

    n = (
        coassoc.shape[0]
    )


    values = coassoc[
        np.triu_indices(
            n,
            k=1
        )
    ]


    ambiguous = (
        (values > lower)
        &
        (values < upper)
    )


    return float(
        np.mean(
            ambiguous
        )
    )


def compute_within_consensus(
    coassoc,
    labels
):

    """
    Mean co-association probability among all sample pairs
    belonging to the same consensus cluster.
    """

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


        sub = coassoc[
            np.ix_(
                idx,
                idx
            )
        ]


        tri = sub[
            np.triu_indices(
                len(
                    idx
                ),
                k=1
            )
        ]


        if len(
            tri
        ) > 0:

            values.extend(
                tri.tolist()
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


def agglomerative_from_precomputed(
    distance_matrix,
    k
):

    """
    Handles sklearn version differences.
    """

    try:

        model = AgglomerativeClustering(

            n_clusters=k,

            metric="precomputed",

            linkage="average"
        )

    except TypeError:

        model = AgglomerativeClustering(

            n_clusters=k,

            affinity="precomputed",

            linkage="average"
        )


    return model.fit_predict(
        distance_matrix
    )


def compute_centroids(
    data,
    labels,
    k
):

    centroids = np.zeros(
        (
            k,
            data.shape[1]
        ),
        dtype=np.float64
    )


    for cluster_id in range(
        k
    ):

        subset = data[
            labels == cluster_id
        ]


        if len(
            subset
        ) == 0:

            raise RuntimeError(
                f"Consensus cluster {cluster_id} is empty."
            )


        centroids[
            cluster_id
        ] = np.mean(
            subset,
            axis=0
        )


    return centroids


def assign_nearest_centroid(
    data,
    centroids
):

    diff = (
        data[:, None, :]
        -
        centroids[None, :, :]
    )


    squared_distance = np.sum(
        diff * diff,
        axis=2
    )


    return np.argmin(
        squared_distance,
        axis=1
    )


# ============================================================
# 4. EVALUATE ONE FOLD / ONE k
# ============================================================

def evaluate_fold_k(
    fold,
    k
):

    print(
        "\n" +
        "#" * 100
    )

    print(
        f"OUTER FOLD {fold} | k = {k}"
    )

    print(
        "#" * 100
    )


    # --------------------------------------------------------
    # Input paths
    # --------------------------------------------------------

    fold_rep_dir = (
        REPRESENTATION_DIR /
        f"Fold{fold}"
    )


    train_file = (

        fold_rep_dir /
        f"PBMMAE_LINEAR64_"
        f"Fold{fold}_13x13_"
        f"TRAIN_latent64.npy"
    )


    val_file = (

        fold_rep_dir /
        f"PBMMAE_LINEAR64_"
        f"Fold{fold}_13x13_"
        f"VALIDATION_latent64.npy"
    )


    for path in [
        train_file,
        val_file
    ]:

        print(
            "\nExists:",
            path.exists(),
            "|",
            path
        )

        if not path.exists():

            raise FileNotFoundError(
                path
            )


    # --------------------------------------------------------
    # Load fixed embeddings
    # --------------------------------------------------------

    Z_train = np.load(
        train_file
    ).astype(
        np.float64
    )


    Z_val = np.load(
        val_file
    ).astype(
        np.float64
    )


    assert Z_train.ndim == 2
    assert Z_val.ndim == 2

    assert Z_train.shape[1] == LATENT_DIM
    assert Z_val.shape[1] == LATENT_DIM

    assert np.isfinite(
        Z_train
    ).all()

    assert np.isfinite(
        Z_val
    ).all()


    N_TRAIN = (
        Z_train.shape[0]
    )

    N_VAL = (
        Z_val.shape[0]
    )


    print(
        "\nTRAIN shape:",
        Z_train.shape
    )

    print(
        "VALIDATION shape:",
        Z_val.shape
    )


    # --------------------------------------------------------
    # 50 independent single-init KMeans runs
    # --------------------------------------------------------

    label_runs = []

    inertia_runs = []


    for run_id in range(
        N_RUNS
    ):

        # Fold-specific deterministic seed sequence
        seed = (
            BASE_SEED
            +
            fold * 1000
            +
            run_id
        )


        kmeans = KMeans(

            n_clusters=k,

            init="k-means++",

            n_init=1,

            max_iter=MAX_ITER,

            tol=TOL,

            random_state=seed,

            algorithm="lloyd"
        )


        labels = kmeans.fit_predict(
            Z_train
        )


        label_runs.append(
            labels
        )


        inertia_runs.append(
            float(
                kmeans.inertia_
            )
        )


    label_runs = np.asarray(
        label_runs,
        dtype=np.int32
    )


    # --------------------------------------------------------
    # Pairwise initialization stability
    # --------------------------------------------------------

    ari_values = []

    nmi_values = []


    for i, j in combinations(
        range(
            N_RUNS
        ),
        2
    ):

        ari_values.append(

            adjusted_rand_score(

                label_runs[
                    i
                ],

                label_runs[
                    j
                ]
            )
        )


        nmi_values.append(

            normalized_mutual_info_score(

                label_runs[
                    i
                ],

                label_runs[
                    j
                ]
            )
        )


    ari_values = np.asarray(
        ari_values,
        dtype=np.float64
    )

    nmi_values = np.asarray(
        nmi_values,
        dtype=np.float64
    )


    # --------------------------------------------------------
    # Co-association matrix
    # --------------------------------------------------------

    coassoc = build_coassociation(
        label_runs
    )


    # --------------------------------------------------------
    # PAC
    # --------------------------------------------------------

    pac = compute_pac(

        coassoc,

        lower=
            PAC_LOWER,

        upper=
            PAC_UPPER
    )


    # --------------------------------------------------------
    # Consensus partition
    # --------------------------------------------------------

    distance_matrix = (
        1.0 -
        coassoc
    )


    np.fill_diagonal(
        distance_matrix,
        0.0
    )


    consensus_train_labels = (
        agglomerative_from_precomputed(

            distance_matrix,

            k
        )
    )


    # --------------------------------------------------------
    # Within-cluster consensus
    # --------------------------------------------------------

    within_consensus = (
        compute_within_consensus(

            coassoc,

            consensus_train_labels
        )
    )


    # --------------------------------------------------------
    # Consensus centroids in original latent coordinates
    # --------------------------------------------------------

    centroids = compute_centroids(

        Z_train,

        consensus_train_labels,

        k
    )


    # --------------------------------------------------------
    # Held-out validation assignment
    # --------------------------------------------------------

    consensus_val_labels = (
        assign_nearest_centroid(

            Z_val,

            centroids
        )
    )


    # --------------------------------------------------------
    # TRAIN internal metrics
    # --------------------------------------------------------

    silhouette = silhouette_score(

        Z_train,

        consensus_train_labels,

        metric="euclidean"
    )


    dbi = davies_bouldin_score(

        Z_train,

        consensus_train_labels
    )


    ch = calinski_harabasz_score(

        Z_train,

        consensus_train_labels
    )


    # --------------------------------------------------------
    # Occupancy
    # --------------------------------------------------------

    train_counts = cluster_counts(

        consensus_train_labels,

        k
    )


    val_counts = cluster_counts(

        consensus_val_labels,

        k
    )


    train_empty = int(
        np.sum(
            train_counts == 0
        )
    )


    val_empty = int(
        np.sum(
            val_counts == 0
        )
    )


    val_occupied = int(
        np.sum(
            val_counts > 0
        )
    )


    train_min = int(
        train_counts.min()
    )


    train_max = int(
        train_counts.max()
    )


    train_min_nonzero = int(
        train_counts[
            train_counts > 0
        ].min()
    )


    val_min = int(
        val_counts.min()
    )


    val_max = int(
        val_counts.max()
    )


    val_min_nonzero = int(
        val_counts[
            val_counts > 0
        ].min()
    )


    train_dominant_fraction = float(
        train_max /
        N_TRAIN
    )


    val_dominant_fraction = float(
        val_max /
        N_VAL
    )


    # --------------------------------------------------------
    # Save artifacts
    # --------------------------------------------------------

    fold_k_dir = (

        CONSENSUS_DIR /
        f"Fold{fold}" /
        f"k{k}"
    )


    fold_k_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    np.save(

        fold_k_dir /
        f"Fold{fold}_k{k}_"
        f"50_single_init_TRAIN_labels.npy",

        label_runs
    )


    np.save(

        fold_k_dir /
        f"Fold{fold}_k{k}_"
        f"TRAIN_coassociation.npy",

        coassoc
    )


    np.save(

        fold_k_dir /
        f"Fold{fold}_k{k}_"
        f"consensus_TRAIN_labels.npy",

        consensus_train_labels
    )


    np.save(

        fold_k_dir /
        f"Fold{fold}_k{k}_"
        f"consensus_VALIDATION_labels.npy",

        consensus_val_labels
    )


    np.save(

        fold_k_dir /
        f"Fold{fold}_k{k}_"
        f"consensus_centroids.npy",

        centroids
    )


    # --------------------------------------------------------
    # Result record
    # --------------------------------------------------------

    record = {

        "Fold":
            fold,

        "k":
            k,

        "Development_fold":
            3,

        "Architecture_frozen":
            True,

        "N_runs":
            N_RUNS,

        "TRAIN_patches":
            N_TRAIN,

        "VALIDATION_patches":
            N_VAL,

        # ----------------------------------------------------
        # Stability
        # ----------------------------------------------------

        "ARI_mean":
            float(
                np.mean(
                    ari_values
                )
            ),

        "ARI_std":
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

        "NMI_std":
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

        # ----------------------------------------------------
        # KMeans inertia across runs
        # ----------------------------------------------------

        "Inertia_mean":
            float(
                np.mean(
                    inertia_runs
                )
            ),

        "Inertia_std":
            float(
                np.std(
                    inertia_runs,
                    ddof=1
                )
            ),

        # ----------------------------------------------------
        # Consensus
        # ----------------------------------------------------

        "PAC_0.1_0.9":
            pac,

        "Within_cluster_consensus":
            within_consensus,

        # ----------------------------------------------------
        # Internal TRAIN metrics
        # ----------------------------------------------------

        "Consensus_Silhouette_TRAIN":
            float(
                silhouette
            ),

        "Consensus_DBI_TRAIN":
            float(
                dbi
            ),

        "Consensus_CH_TRAIN":
            float(
                ch
            ),

        # ----------------------------------------------------
        # TRAIN occupancy
        # ----------------------------------------------------

        "TRAIN_empty_clusters":
            train_empty,

        "TRAIN_min_cluster":
            train_min,

        "TRAIN_min_nonzero_cluster":
            train_min_nonzero,

        "TRAIN_max_cluster":
            train_max,

        "TRAIN_dominant_fraction":
            train_dominant_fraction,

        # ----------------------------------------------------
        # VALIDATION occupancy
        # ----------------------------------------------------

        "VALIDATION_clusters_occupied":
            val_occupied,

        "VALIDATION_empty_clusters":
            val_empty,

        "VALIDATION_occupancy_fraction":
            float(
                val_occupied /
                k
            ),

        "VALIDATION_min_cluster":
            val_min,

        "VALIDATION_min_nonzero_cluster":
            val_min_nonzero,

        "VALIDATION_max_cluster":
            val_max,

        "VALIDATION_dominant_fraction":
            val_dominant_fraction
    }


    # --------------------------------------------------------
    # Individual occupancy counts
    # --------------------------------------------------------

    for cluster_id in range(
        k
    ):

        record[
            f"TRAIN_cluster_{cluster_id}_n"
        ] = int(
            train_counts[
                cluster_id
            ]
        )


        record[
            f"VALIDATION_cluster_{cluster_id}_n"
        ] = int(
            val_counts[
                cluster_id
            ]
        )


    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print(
        f"\nARI mean / median     : "
        f"{np.mean(ari_values):.4f} / "
        f"{np.median(ari_values):.4f}"
    )

    print(
        f"NMI mean / median     : "
        f"{np.mean(nmi_values):.4f} / "
        f"{np.median(nmi_values):.4f}"
    )

    print(
        f"PAC                   : "
        f"{pac:.4f}"
    )

    print(
        f"Within consensus      : "
        f"{within_consensus:.4f}"
    )

    print(
        f"Silhouette            : "
        f"{silhouette:.4f}"
    )

    print(
        f"DBI                   : "
        f"{dbi:.4f}"
    )

    print(
        f"CH                    : "
        f"{ch:.2f}"
    )

    print(
        "TRAIN counts          :",
        train_counts.tolist()
    )

    print(
        "VALIDATION counts     :",
        val_counts.tolist()
    )

    print(
        f"VALIDATION occupied   : "
        f"{val_occupied}/{k}"
    )

    print(
        f"VALIDATION empty      : "
        f"{val_empty}"
    )

    print(
        f"VALIDATION dominant   : "
        f"{val_dominant_fraction:.4f}"
    )


    return record


# ============================================================
# 5. RUN ALL OUTER FOLD / k COMBINATIONS
# ============================================================

records = []


for fold in OUTER_FOLDS:

    for k in K_VALUES:

        record = evaluate_fold_k(
            fold,
            k
        )

        records.append(
            record
        )


# ============================================================
# 6. SAVE COMPLETE RESULTS
# ============================================================

results_df = pd.DataFrame(
    records
)


base_columns = [

    "Fold",

    "k",

    "Development_fold",

    "Architecture_frozen",

    "N_runs",

    "TRAIN_patches",

    "VALIDATION_patches",

    "ARI_mean",

    "ARI_std",

    "ARI_min",

    "ARI_median",

    "NMI_mean",

    "NMI_std",

    "NMI_min",

    "NMI_median",

    "Inertia_mean",

    "Inertia_std",

    "PAC_0.1_0.9",

    "Within_cluster_consensus",

    "Consensus_Silhouette_TRAIN",

    "Consensus_DBI_TRAIN",

    "Consensus_CH_TRAIN",

    "TRAIN_empty_clusters",

    "TRAIN_min_cluster",

    "TRAIN_min_nonzero_cluster",

    "TRAIN_max_cluster",

    "TRAIN_dominant_fraction",

    "VALIDATION_clusters_occupied",

    "VALIDATION_empty_clusters",

    "VALIDATION_occupancy_fraction",

    "VALIDATION_min_cluster",

    "VALIDATION_min_nonzero_cluster",

    "VALIDATION_max_cluster",

    "VALIDATION_dominant_fraction"
]


other_columns = [

    c

    for c
    in results_df.columns

    if c not in base_columns
]


results_df = results_df[
    base_columns +
    other_columns
]


results_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# 7. PER-k OUTER-FOLD SUMMARY
# ============================================================

summary_records = []


for k in K_VALUES:

    subset = results_df[
        results_df[
            "k"
        ] == k
    ]


    summary_records.append({

        "k":
            k,

        "N_outer_folds":
            len(
                subset
            ),

        "Mean_ARI":
            float(
                subset[
                    "ARI_mean"
                ].mean()
            ),

        "SD_ARI":
            float(
                subset[
                    "ARI_mean"
                ].std(
                    ddof=1
                )
            ),

        "Mean_NMI":
            float(
                subset[
                    "NMI_mean"
                ].mean()
            ),

        "SD_NMI":
            float(
                subset[
                    "NMI_mean"
                ].std(
                    ddof=1
                )
            ),

        "Mean_PAC":
            float(
                subset[
                    "PAC_0.1_0.9"
                ].mean()
            ),

        "SD_PAC":
            float(
                subset[
                    "PAC_0.1_0.9"
                ].std(
                    ddof=1
                )
            ),

        "Mean_within_consensus":
            float(
                subset[
                    "Within_cluster_consensus"
                ].mean()
            ),

        "Mean_Silhouette":
            float(
                subset[
                    "Consensus_Silhouette_TRAIN"
                ].mean()
            ),

        "Mean_DBI":
            float(
                subset[
                    "Consensus_DBI_TRAIN"
                ].mean()
            ),

        "Mean_CH":
            float(
                subset[
                    "Consensus_CH_TRAIN"
                ].mean()
            ),

        "Mean_VALIDATION_occupancy_fraction":
            float(
                subset[
                    "VALIDATION_occupancy_fraction"
                ].mean()
            ),

        "Min_VALIDATION_occupancy_fraction":
            float(
                subset[
                    "VALIDATION_occupancy_fraction"
                ].min()
            ),

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

        "Mean_VALIDATION_min_nonzero_cluster":
            float(
                subset[
                    "VALIDATION_min_nonzero_cluster"
                ].mean()
            ),

        "Mean_VALIDATION_dominant_fraction":
            float(
                subset[
                    "VALIDATION_dominant_fraction"
                ].mean()
            ),

        "Max_VALIDATION_dominant_fraction":
            float(
                subset[
                    "VALIDATION_dominant_fraction"
                ].max()
            )
    })


summary_df = pd.DataFrame(
    summary_records
)


SUMMARY_CSV = (
    CONSENSUS_DIR /
    "PBMMAE_LINEAR64_outer_folds_consensus_k8_k9_summary.csv"
)


summary_df.to_csv(
    SUMMARY_CSV,
    index=False
)


# ============================================================
# 8. PRINT OUTER-FOLD RESULTS
# ============================================================

display_columns = [

    "Fold",

    "k",

    "ARI_mean",

    "NMI_mean",

    "PAC_0.1_0.9",

    "Within_cluster_consensus",

    "Consensus_Silhouette_TRAIN",

    "Consensus_DBI_TRAIN",

    "Consensus_CH_TRAIN",

    "TRAIN_min_cluster",

    "VALIDATION_clusters_occupied",

    "VALIDATION_empty_clusters",

    "VALIDATION_min_nonzero_cluster",

    "VALIDATION_max_cluster",

    "VALIDATION_dominant_fraction"
]


print(
    "\n\n" +
    "=" * 120
)

print(
    "OUTER-FOLD CONSENSUS RESULTS"
)

print(
    "=" * 120
)


print(

    results_df[
        display_columns
    ]
    .round(
        {
            "ARI_mean": 4,

            "NMI_mean": 4,

            "PAC_0.1_0.9": 4,

            "Within_cluster_consensus": 4,

            "Consensus_Silhouette_TRAIN": 4,

            "Consensus_DBI_TRAIN": 4,

            "Consensus_CH_TRAIN": 2,

            "VALIDATION_dominant_fraction": 4
        }
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 9. PRINT k=8 vs k=9 SUMMARY
# ============================================================

print(
    "\n" +
    "=" * 120
)

print(
    "FROZEN k CANDIDATE SUMMARY ACROSS OUTER FOLDS"
)

print(
    "=" * 120
)


print(

    summary_df
    .round(
        {
            "Mean_ARI": 4,
            "SD_ARI": 4,

            "Mean_NMI": 4,
            "SD_NMI": 4,

            "Mean_PAC": 4,
            "SD_PAC": 4,

            "Mean_within_consensus": 4,

            "Mean_Silhouette": 4,

            "Mean_DBI": 4,

            "Mean_CH": 2,

            "Mean_VALIDATION_occupancy_fraction": 4,

            "Min_VALIDATION_occupancy_fraction": 4,

            "Mean_VALIDATION_min_nonzero_cluster": 2,

            "Mean_VALIDATION_dominant_fraction": 4,

            "Max_VALIDATION_dominant_fraction": 4
        }
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 10. FINAL OUTPUT
# ============================================================

print(
    "\n" +
    "=" * 100
)

print(
    "OUTER-FOLD k=8 / k=9 CONSENSUS EVALUATION COMPLETED"
)

print(
    "=" * 100
)


print(
    "\nDetailed results:"
)

print(
    OUTPUT_CSV
)


print(
    "\nPer-k outer-fold summary:"
)

print(
    SUMMARY_CSV
)


print(
    "\nConsensus artifact directory:"
)

print(
    CONSENSUS_DIR
)


print(
    "=" * 100
)