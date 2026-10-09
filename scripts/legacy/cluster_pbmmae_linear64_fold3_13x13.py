import os
# ============================================================
# Paper 1 — PB-MMAE LinearFusion64 clustering evaluation
# Fold 3, 13x13
#
# Uses fixed unmasked fused embeddings:
#   TRAIN      : 455 x 64
#   VALIDATION : 392 x 64
#
# KMeans is fitted on TRAIN only for k = 2 ... 10.
# Validation is assigned to the fitted TRAIN centroids.
#
# Metrics:
#   - TRAIN Silhouette
#   - TRAIN Davies-Bouldin
#   - TRAIN Calinski-Harabasz
#   - TRAIN cluster occupancy
#   - VALIDATION cluster occupancy
#   - number of empty validation clusters
#
# No model training occurs here.
# ============================================================

import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
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

OUTPUT_CSV = (
    RUN_DIR /
    "PBMMAE_LINEAR64_Fold3_13x13_KMeans_k2to10.csv"
)

LABEL_DIR = (
    RUN_DIR /
    "kmeans_labels"
)

LABEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

K_VALUES = range(
    2,
    11
)

SEED = 42

# Keep this deterministic and directly comparable.
N_INIT = 50
MAX_ITER = 1000
TOL = 1e-4


print("=" * 82)
print("PB-MMAE LINEAR-FUSION-64 — KMEANS k=2...10")
print("=" * 82)

print("TRAIN latent file:")
print(TRAIN_LATENT_FILE)

print("\nVALIDATION latent file:")
print(VAL_LATENT_FILE)


# ============================================================
# 3. FILE CHECKS
# ============================================================

for path in [
    TRAIN_LATENT_FILE,
    VAL_LATENT_FILE
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


# ============================================================
# 4. LOAD FIXED REPRESENTATIONS
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


# ============================================================
# 5. HELPER — CLUSTER OCCUPANCY
# ============================================================

def cluster_counts(
    labels,
    k
):

    counts = np.bincount(
        labels,
        minlength=k
    )

    assert len(
        counts
    ) == k

    return counts


# ============================================================
# 6. KMEANS SWEEP
# ============================================================

records = []


for k in K_VALUES:

    print("\n" + "-" * 82)
    print(
        f"k = {k}"
    )
    print("-" * 82)


    # --------------------------------------------------------
    # Fit KMeans on TRAIN only
    # --------------------------------------------------------

    kmeans = KMeans(

        n_clusters=k,

        init="k-means++",

        n_init=N_INIT,

        max_iter=MAX_ITER,

        tol=TOL,

        random_state=SEED,

        algorithm="lloyd"
    )


    train_labels = kmeans.fit_predict(
        Z_train
    )


    # --------------------------------------------------------
    # Assign held-out validation to TRAIN centroids
    # --------------------------------------------------------

    val_labels = kmeans.predict(
        Z_val
    )


    # --------------------------------------------------------
    # TRAIN-only internal metrics
    # --------------------------------------------------------

    n_train_clusters_found = len(
        np.unique(
            train_labels
        )
    )


    if n_train_clusters_found < 2:

        silhouette = np.nan
        dbi = np.nan
        ch = np.nan

    else:

        silhouette = silhouette_score(
            Z_train,
            train_labels,
            metric="euclidean"
        )

        dbi = davies_bouldin_score(
            Z_train,
            train_labels
        )

        ch = calinski_harabasz_score(
            Z_train,
            train_labels
        )


    # --------------------------------------------------------
    # Occupancy diagnostics
    # --------------------------------------------------------

    train_counts = cluster_counts(
        train_labels,
        k
    )

    val_counts = cluster_counts(
        val_labels,
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


    train_dominant_fraction = float(
        train_max /
        len(
            train_labels
        )
    )


    val_dominant_fraction = float(
        val_max /
        len(
            val_labels
        )
    )


    val_occupied_clusters = int(
        np.sum(
            val_counts > 0
        )
    )


    val_occupancy_fraction = float(
        val_occupied_clusters /
        k
    )


    # --------------------------------------------------------
    # Record metrics
    # --------------------------------------------------------

    record = {

        "k":
            k,

        "Silhouette_TRAIN":
            silhouette,

        "DBI_TRAIN":
            dbi,

        "CH_TRAIN":
            ch,

        "KMeans_inertia_TRAIN":
            float(
                kmeans.inertia_
            ),

        "KMeans_iterations":
            int(
                kmeans.n_iter_
            ),

        "TRAIN_clusters_found":
            n_train_clusters_found,

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

        "VALIDATION_clusters_occupied":
            val_occupied_clusters,

        "VALIDATION_empty_clusters":
            val_empty,

        "VALIDATION_occupancy_fraction":
            val_occupancy_fraction,

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
    # Add individual cluster counts
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


    records.append(
        record
    )


    # --------------------------------------------------------
    # Save labels
    # --------------------------------------------------------

    np.save(

        LABEL_DIR /
        f"PBMMAE_LINEAR64_Fold3_13x13_"
        f"k{k}_TRAIN_labels.npy",

        train_labels
    )


    np.save(

        LABEL_DIR /
        f"PBMMAE_LINEAR64_Fold3_13x13_"
        f"k{k}_VALIDATION_labels.npy",

        val_labels
    )


    np.save(

        LABEL_DIR /
        f"PBMMAE_LINEAR64_Fold3_13x13_"
        f"k{k}_centroids.npy",

        kmeans.cluster_centers_
    )


    # --------------------------------------------------------
    # Print concise diagnostics
    # --------------------------------------------------------

    print(
        f"TRAIN Silhouette          : "
        f"{silhouette:.6f}"
    )

    print(
        f"TRAIN DBI                 : "
        f"{dbi:.6f}"
    )

    print(
        f"TRAIN CH                  : "
        f"{ch:.6f}"
    )


    print(
        "TRAIN counts              :",
        train_counts.tolist()
    )


    print(
        "VALIDATION counts         :",
        val_counts.tolist()
    )


    print(
        f"VALIDATION occupied       : "
        f"{val_occupied_clusters}/{k}"
    )


    print(
        f"VALIDATION empty clusters : "
        f"{val_empty}"
    )


    print(
        f"VALIDATION dominant frac  : "
        f"{val_dominant_fraction:.4f}"
    )


# ============================================================
# 7. BUILD RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    records
)


# Sort column structure for readability
base_columns = [

    "k",

    "Silhouette_TRAIN",

    "DBI_TRAIN",

    "CH_TRAIN",

    "KMeans_inertia_TRAIN",

    "KMeans_iterations",

    "TRAIN_clusters_found",

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

    for col in results_df.columns

    if col not in base_columns
]


results_df = results_df[
    base_columns +
    remaining_columns
]


# ============================================================
# 8. SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# 9. PRINT MAIN COMPARISON TABLE
# ============================================================

display_columns = [

    "k",

    "Silhouette_TRAIN",

    "DBI_TRAIN",

    "CH_TRAIN",

    "TRAIN_min_cluster",

    "VALIDATION_clusters_occupied",

    "VALIDATION_empty_clusters",

    "VALIDATION_min_cluster",

    "VALIDATION_min_nonzero_cluster",

    "VALIDATION_max_cluster",

    "VALIDATION_dominant_fraction"
]


print("\n" + "=" * 82)

print(
    "KMEANS SWEEP SUMMARY"
)

print("=" * 82)


print(

    results_df[
        display_columns
    ]
    .round(
        {
            "Silhouette_TRAIN": 4,
            "DBI_TRAIN": 4,
            "CH_TRAIN": 2,
            "VALIDATION_dominant_fraction": 4
        }
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 10. SIMPLE OCCUPANCY SCREEN
#
# This is diagnostic only, not a final k-selection rule.
# ============================================================

full_occupancy = results_df[
    results_df[
        "VALIDATION_empty_clusters"
    ] == 0
]


print("\n" + "=" * 82)

print(
    "VALIDATION OCCUPANCY SCREEN"
)

print("=" * 82)


if len(
    full_occupancy
) == 0:

    print(
        "No k in 2...10 achieved complete "
        "validation cluster occupancy."
    )

else:

    print(
        "k values with no empty "
        "validation clusters:"
    )

    print(
        full_occupancy[
            [
                "k",
                "Silhouette_TRAIN",
                "DBI_TRAIN",
                "CH_TRAIN",
                "VALIDATION_min_cluster",
                "VALIDATION_max_cluster",
                "VALIDATION_dominant_fraction"
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )


# ============================================================
# 11. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 82)

print(
    "PB-MMAE LINEAR-FUSION-64 KMEANS SWEEP COMPLETED"
)

print("=" * 82)


print(
    "\nResults CSV:"
)

print(
    OUTPUT_CSV
)


print(
    "\nLabels and centroids:"
)

print(
    LABEL_DIR
)

print("=" * 82)