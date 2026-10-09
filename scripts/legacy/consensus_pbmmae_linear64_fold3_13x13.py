import os
# ============================================================
# Paper 1 — PB-MMAE LinearFusion64 consensus clustering
# Fold 3, 13x13
#
# Representation:
#   PB-MMAE
#   per-family LayerNorm
#   single linear fusion 96 -> 64
#
# Fixed embeddings:
#   TRAIN      : 455 x 64
#   VALIDATION : 392 x 64
#
# For each k = 2 ... 10:
#   1. Run 50 independent single-init KMeans fits on TRAIN.
#   2. Measure pairwise ARI and NMI between runs.
#   3. Build TRAIN co-association matrix.
#   4. Compute PAC using [0.1, 0.9].
#   5. Derive consensus TRAIN partition using
#      average-linkage agglomerative clustering on
#      distance = 1 - coassociation.
#   6. Compute consensus TRAIN centroids in the ORIGINAL
#      fixed 64-D latent space.
#   7. Assign VALIDATION to TRAIN consensus centroids.
#   8. Compute TRAIN internal metrics and occupancy.
#
# No PB-MMAE training occurs here.
# Validation is never used to construct consensus clusters.
# ============================================================

import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering

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

RUN_DIR = (
    PROJECT_ROOT /
    "03_outputs" /
    "models" /
    "PBMMAE" /
    "Fold3_13x13_LayerNorm_LinearFusion64"
)

TRAIN_LATENT_FILE = (
    RUN_DIR /
    "PBMMAE_LINEAR64_Fold3_13x13_TRAIN_latent64.npy"
)

VAL_LATENT_FILE = (
    RUN_DIR /
    "PBMMAE_LINEAR64_Fold3_13x13_VALIDATION_latent64.npy"
)

CONSENSUS_DIR = (
    RUN_DIR /
    "consensus_clustering"
)

CONSENSUS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_CSV = (
    CONSENSUS_DIR /
    "PBMMAE_LINEAR64_Fold3_13x13_consensus_k2to10.csv"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

K_VALUES = range(
    2,
    11
)

N_RUNS = 50

BASE_SEED = 42

MAX_ITER = 1000
TOL = 1e-4

PAC_LOWER = 0.10
PAC_UPPER = 0.90


print("=" * 86)
print("PB-MMAE LINEAR-FUSION-64 — CONSENSUS CLUSTERING")
print("=" * 86)

print("Runs per k :", N_RUNS)
print("k range    : 2 ... 10")
print(
    "PAC range  :",
    PAC_LOWER,
    "to",
    PAC_UPPER
)


# ============================================================
# 3. FILE CHECKS
# ============================================================

for path in [
    TRAIN_LATENT_FILE,
    VAL_LATENT_FILE
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


# ============================================================
# 4. LOAD FIXED LATENT REPRESENTATIONS
# ============================================================

Z_train = np.load(
    TRAIN_LATENT_FILE
).astype(
    np.float64
)

Z_val = np.load(
    VAL_LATENT_FILE
).astype(
    np.float64
)


print(
    "\nTRAIN shape:",
    Z_train.shape
)

print(
    "VALIDATION shape:",
    Z_val.shape
)


assert Z_train.ndim == 2
assert Z_val.ndim == 2

assert Z_train.shape[1] == 64
assert Z_val.shape[1] == 64

assert np.isfinite(
    Z_train
).all()

assert np.isfinite(
    Z_val
).all()


N_TRAIN = Z_train.shape[0]
N_VAL = Z_val.shape[0]


# ============================================================
# 5. HELPER FUNCTIONS
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

    Returns:
        coassociation matrix [n_samples, n_samples]

    C_ij = fraction of runs in which samples i,j
    belong to the same cluster.
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

    Uses upper triangular off-diagonal co-association
    probabilities.

    Lower PAC is better.
    """

    n = coassoc.shape[0]

    upper_triangle = coassoc[
        np.triu_indices(
            n,
            k=1
        )
    ]


    ambiguous = (
        (upper_triangle > lower)
        &
        (upper_triangle < upper)
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
    Mean within-cluster co-association probability,
    averaged over all within-cluster sample pairs.

    Higher is better.
    """

    values = []

    unique_labels = np.unique(
        labels
    )


    for cluster_id in unique_labels:

        idx = np.where(
            labels == cluster_id
        )[0]


        if len(idx) < 2:

            continue


        sub = coassoc[
            np.ix_(
                idx,
                idx
            )
        ]


        tri = sub[
            np.triu_indices(
                len(idx),
                k=1
            )
        ]


        if len(tri) > 0:

            values.extend(
                tri.tolist()
            )


    if len(values) == 0:

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
    Compatibility helper for different sklearn versions.
    """

    try:

        model = AgglomerativeClustering(

            n_clusters=k,

            metric="precomputed",

            linkage="average"
        )

    except TypeError:

        # Older sklearn syntax

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

        cluster_data = data[
            labels == cluster_id
        ]


        if len(cluster_data) == 0:

            raise RuntimeError(

                f"Consensus cluster "
                f"{cluster_id} is empty."
            )


        centroids[
            cluster_id
        ] = np.mean(
            cluster_data,
            axis=0
        )


    return centroids


def assign_nearest_centroid(
    data,
    centroids
):

    """
    Euclidean nearest-centroid assignment.
    """

    # [samples, clusters, dims]

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
# 6. MAIN CONSENSUS LOOP
# ============================================================

records = []


for k in K_VALUES:

    print(
        "\n" +
        "=" * 86
    )

    print(
        f"k = {k}"
    )

    print(
        "=" * 86
    )


    # --------------------------------------------------------
    # 6A. Repeated single-initialization KMeans
    # --------------------------------------------------------

    label_runs = []

    inertia_runs = []


    for run_id in range(
        N_RUNS
    ):

        seed = (
            BASE_SEED
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


    # Save repeated-run labels

    np.save(

        CONSENSUS_DIR /
        f"k{k}_50_single_init_TRAIN_labels.npy",

        label_runs
    )


    # --------------------------------------------------------
    # 6B. Pairwise ARI / NMI
    # --------------------------------------------------------

    ari_values = []

    nmi_values = []


    for i, j in combinations(
        range(
            N_RUNS
        ),
        2
    ):

        labels_i = label_runs[
            i
        ]

        labels_j = label_runs[
            j
        ]


        ari_values.append(

            adjusted_rand_score(
                labels_i,
                labels_j
            )
        )


        nmi_values.append(

            normalized_mutual_info_score(
                labels_i,
                labels_j
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
    # 6C. Co-association matrix
    # --------------------------------------------------------

    coassoc = build_coassociation(
        label_runs
    )


    np.save(

        CONSENSUS_DIR /
        f"k{k}_TRAIN_coassociation.npy",

        coassoc
    )


    # --------------------------------------------------------
    # 6D. PAC
    # --------------------------------------------------------

    pac = compute_pac(

        coassoc,

        lower=
            PAC_LOWER,

        upper=
            PAC_UPPER
    )


    # --------------------------------------------------------
    # 6E. Consensus clustering
    #
    # distance = 1 - co-association
    # --------------------------------------------------------

    distance_matrix = (
        1.0
        -
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
    # 6F. Consensus quality
    # --------------------------------------------------------

    within_consensus = (
        compute_within_consensus(

            coassoc,

            consensus_train_labels
        )
    )


    # --------------------------------------------------------
    # 6G. Consensus centroids in ORIGINAL 64-D latent space
    # --------------------------------------------------------

    centroids = compute_centroids(

        Z_train,

        consensus_train_labels,

        k
    )


    # --------------------------------------------------------
    # 6H. Assign held-out validation
    # --------------------------------------------------------

    consensus_val_labels = (
        assign_nearest_centroid(

            Z_val,

            centroids
        )
    )


    # --------------------------------------------------------
    # 6I. TRAIN internal metrics
    # --------------------------------------------------------

    train_unique = np.unique(
        consensus_train_labels
    )


    if len(
        train_unique
    ) >= 2:

        silhouette = (
            silhouette_score(

                Z_train,

                consensus_train_labels,

                metric="euclidean"
            )
        )

        dbi = (
            davies_bouldin_score(

                Z_train,

                consensus_train_labels
            )
        )

        ch = (
            calinski_harabasz_score(

                Z_train,

                consensus_train_labels
            )
        )

    else:

        silhouette = np.nan
        dbi = np.nan
        ch = np.nan


    # --------------------------------------------------------
    # 6J. Occupancy
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


    train_min = int(
        train_counts.min()
    )

    train_max = int(
        train_counts.max()
    )


    val_min = int(
        val_counts.min()
    )

    val_max = int(
        val_counts.max()
    )


    train_min_nonzero = int(
        train_counts[
            train_counts > 0
        ].min()
    )


    if np.any(
        val_counts > 0
    ):

        val_min_nonzero = int(
            val_counts[
                val_counts > 0
            ].min()
        )

    else:

        val_min_nonzero = 0


    val_occupied = int(
        np.sum(
            val_counts > 0
        )
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
    # 6K. Save consensus artifacts
    # --------------------------------------------------------

    np.save(

        CONSENSUS_DIR /
        f"k{k}_consensus_TRAIN_labels.npy",

        consensus_train_labels
    )


    np.save(

        CONSENSUS_DIR /
        f"k{k}_consensus_VALIDATION_labels.npy",

        consensus_val_labels
    )


    np.save(

        CONSENSUS_DIR /
        f"k{k}_consensus_centroids.npy",

        centroids
    )


    # --------------------------------------------------------
    # 6L. Record summary
    # --------------------------------------------------------

    record = {

        "k":
            k,

        "N_runs":
            N_RUNS,

        # ----------------------------------------------------
        # Initialization stability
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
        # Consensus diagnostics
        # ----------------------------------------------------

        "PAC_0.1_0.9":
            pac,

        "Within_cluster_consensus":
            within_consensus,

        # ----------------------------------------------------
        # Consensus TRAIN internal metrics
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


    # Individual cluster counts

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


    records.append(
        record
    )


    # --------------------------------------------------------
    # 6M. Print concise summary
    # --------------------------------------------------------

    print(
        f"ARI mean / median : "
        f"{np.mean(ari_values):.4f} / "
        f"{np.median(ari_values):.4f}"
    )

    print(
        f"NMI mean / median : "
        f"{np.mean(nmi_values):.4f} / "
        f"{np.median(nmi_values):.4f}"
    )

    print(
        f"PAC               : "
        f"{pac:.4f}"
    )

    print(
        f"Within consensus  : "
        f"{within_consensus:.4f}"
    )

    print(
        f"Silhouette        : "
        f"{silhouette:.4f}"
    )

    print(
        f"DBI               : "
        f"{dbi:.4f}"
    )

    print(
        f"CH                : "
        f"{ch:.2f}"
    )

    print(
        "TRAIN counts      :",
        train_counts.tolist()
    )

    print(
        "VALIDATION counts :",
        val_counts.tolist()
    )

    print(
        f"VAL occupied      : "
        f"{val_occupied}/{k}"
    )

    print(
        f"VAL dominant frac : "
        f"{val_dominant_fraction:.4f}"
    )


# ============================================================
# 7. SAVE SUMMARY TABLE
# ============================================================

results_df = pd.DataFrame(
    records
)


base_columns = [

    "k",

    "N_runs",

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


remaining_columns = [

    col

    for col
    in results_df.columns

    if col not in base_columns
]


results_df = results_df[
    base_columns
    +
    remaining_columns
]


results_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# 8. PRINT MAIN COMPARISON TABLE
# ============================================================

display_columns = [

    "k",

    "ARI_mean",

    "ARI_median",

    "NMI_mean",

    "NMI_median",

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
    "\n" +
    "=" * 110
)

print(
    "CONSENSUS CLUSTERING SUMMARY"
)

print(
    "=" * 110
)


print(

    results_df[
        display_columns
    ]
    .round(
        {
            "ARI_mean": 4,
            "ARI_median": 4,

            "NMI_mean": 4,
            "NMI_median": 4,

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
# 9. FULL-VALIDATION-OCCUPANCY SCREEN
# ============================================================

full_occupancy = results_df[
    results_df[
        "VALIDATION_empty_clusters"
    ] == 0
]


print(
    "\n" +
    "=" * 86
)

print(
    "CONSENSUS VALIDATION OCCUPANCY SCREEN"
)

print(
    "=" * 86
)


if len(
    full_occupancy
) == 0:

    print(
        "No k achieved complete "
        "validation occupancy under consensus."
    )

else:

    print(
        "k values with complete "
        "validation occupancy:"
    )


    screen_columns = [

        "k",

        "ARI_mean",

        "NMI_mean",

        "PAC_0.1_0.9",

        "Within_cluster_consensus",

        "Consensus_Silhouette_TRAIN",

        "Consensus_DBI_TRAIN",

        "TRAIN_min_cluster",

        "VALIDATION_min_nonzero_cluster",

        "VALIDATION_max_cluster",

        "VALIDATION_dominant_fraction"
    ]


    print(

        full_occupancy[
            screen_columns
        ]
        .round(4)
        .to_string(
            index=False
        )
    )


# ============================================================
# 10. FINAL OUTPUT
# ============================================================

print(
    "\n" +
    "=" * 86
)

print(
    "PB-MMAE LINEAR-FUSION-64 CONSENSUS ANALYSIS COMPLETED"
)

print(
    "=" * 86
)

print(
    "\nSummary CSV:"
)

print(
    OUTPUT_CSV
)

print(
    "\nConsensus artifacts:"
)

print(
    CONSENSUS_DIR
)

print(
    "=" * 86
)