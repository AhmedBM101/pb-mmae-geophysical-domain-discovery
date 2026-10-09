import os
# ============================================================
# Paper 1 — Formal GPI ↔ PB-MMAE Domain Comparison
#
# Input:
#   Paper1_PBMMAE_domain_external_validation_MASTER.csv
#
# WITHHELD variable:
#   Continuous AHP-derived GPI
#
# Tests:
#   1. Domain-specific descriptive statistics
#   2. Kruskal-Wallis H test
#   3. Epsilon-squared effect size
#   4. 10,000 label-permutation test of global H
#   5. Pairwise Mann-Whitney U tests
#   6. Holm multiple-testing correction
#   7. Cliff's delta pairwise effect sizes
#
# IMPORTANT:
#   GPI was NEVER used during PB-MMAE training,
#   representation learning, clustering, k selection,
#   or meta-domain derivation.
#
# Interpretation:
#   This analysis evaluates agreement/divergence between
#   data-driven latent geophysical organization and the
#   independently withheld expert-weighted GPI.
# ============================================================

import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from scipy.stats import (
    kruskal,
    mannwhitneyu
)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(os.environ["PBMMAE_PROJECT_ROOT"]).expanduser().resolve()

INPUT_FILE = (
    PROJECT_ROOT /
    "03_outputs" /
    "external_validation" /
    "Paper1_PBMMAE_domain_external_validation_MASTER.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT /
    "03_outputs" /
    "external_validation" /
    "GPI_domain_validation"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

DOMAIN_COLUMN = "PBMMAE_domain"

GPI_COLUMN = "GPI_continuous"

DOMAIN_ORDER = [
    "M0",
    "M1",
    "M2",
    "M3"
]

N_PERMUTATIONS = 10000

RANDOM_SEED = 42

rng = np.random.default_rng(
    RANDOM_SEED
)


# ============================================================
# 3. LOAD DATA
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        INPUT_FILE
    )


df = pd.read_csv(
    INPUT_FILE
)


required_columns = [

    DOMAIN_COLUMN,
    GPI_COLUMN
]


for column in required_columns:

    if column not in df.columns:

        raise RuntimeError(
            f"Required column absent: {column}"
        )


analysis_df = df[

    df[
        DOMAIN_COLUMN
    ].notna()

    &

    df[
        GPI_COLUMN
    ].notna()

].copy()


analysis_df[
    DOMAIN_COLUMN
] = analysis_df[
    DOMAIN_COLUMN
].astype(
    str
)


analysis_df[
    GPI_COLUMN
] = pd.to_numeric(
    analysis_df[
        GPI_COLUMN
    ],
    errors="coerce"
)


analysis_df = analysis_df[
    analysis_df[
        GPI_COLUMN
    ].notna()
].copy()


# ============================================================
# 4. AUDIT
# ============================================================

print(
    "=" * 120
)

print(
    "PAPER 1 — GPI ↔ PB-MMAE DOMAIN COMPARISON"
)

print(
    "=" * 120
)


print(
    "\nAnalysis cells:",
    len(
        analysis_df
    )
)


print(
    "Domains:",
    sorted(
        analysis_df[
            DOMAIN_COLUMN
        ].unique()
    )
)


if set(
    analysis_df[
        DOMAIN_COLUMN
    ].unique()
) != set(
    DOMAIN_ORDER
):

    raise RuntimeError(
        "Expected exactly M0, M1, M2 and M3."
    )


print(
    "Overall GPI range:",
    analysis_df[
        GPI_COLUMN
    ].min(),
    "to",
    analysis_df[
        GPI_COLUMN
    ].max()
)


# ============================================================
# 5. DESCRIPTIVE STATISTICS
# ============================================================

descriptive_records = []


for domain in DOMAIN_ORDER:

    values = analysis_df.loc[
        analysis_df[
            DOMAIN_COLUMN
        ] == domain,
        GPI_COLUMN
    ].to_numpy(
        dtype=float
    )


    q25 = np.percentile(
        values,
        25
    )

    q75 = np.percentile(
        values,
        75
    )


    descriptive_records.append({

        "PBMMAE_domain":
            domain,

        "N":
            len(
                values
            ),

        "Mean":
            np.mean(
                values
            ),

        "SD":
            np.std(
                values,
                ddof=1
            ),

        "Median":
            np.median(
                values
            ),

        "Q25":
            q25,

        "Q75":
            q75,

        "IQR":
            q75 - q25,

        "Minimum":
            np.min(
                values
            ),

        "Maximum":
            np.max(
                values
            )
    })


descriptive_df = pd.DataFrame(
    descriptive_records
)


descriptive_file = (
    OUTPUT_ROOT /
    "Paper1_GPI_by_domain_descriptive_statistics.csv"
)


descriptive_df.to_csv(
    descriptive_file,
    index=False
)


# ============================================================
# 6. DOMAIN ARRAYS
# ============================================================

groups = {

    domain:
        analysis_df.loc[
            analysis_df[
                DOMAIN_COLUMN
            ] == domain,
            GPI_COLUMN
        ].to_numpy(
            dtype=float
        )

    for domain
    in DOMAIN_ORDER
}


# ============================================================
# 7. KRUSKAL-WALLIS
# ============================================================

H_statistic, H_p = kruskal(

    *[
        groups[
            domain
        ]
        for domain
        in DOMAIN_ORDER
    ]
)


n_total = len(
    analysis_df
)

k_groups = len(
    DOMAIN_ORDER
)


# ============================================================
# 8. EPSILON-SQUARED
#
# epsilon^2 = (H - k + 1) / (n - k)
#
# Negative sampling values, if any, are clipped at zero.
# ============================================================

epsilon_squared = (

    H_statistic
    -
    k_groups
    +
    1

) / (

    n_total
    -
    k_groups
)


epsilon_squared = max(
    0.0,
    float(
        epsilon_squared
    )
)


# ============================================================
# 9. GLOBAL PERMUTATION TEST
#
# Null:
#   Domain labels and GPI values are unrelated.
#
# GPI observations remain fixed.
# Domain labels are permuted.
#
# This is a categorical label-permutation test.
# It does NOT remove spatial autocorrelation and should
# therefore be interpreted together with effect sizes.
# ============================================================

domain_labels = analysis_df[
    DOMAIN_COLUMN
].to_numpy()

gpi_values = analysis_df[
    GPI_COLUMN
].to_numpy(
    dtype=float
)


permutation_H = np.zeros(
    N_PERMUTATIONS,
    dtype=float
)


for permutation_id in range(
    N_PERMUTATIONS
):

    permuted_domains = rng.permutation(
        domain_labels
    )


    perm_groups = [

        gpi_values[
            permuted_domains == domain
        ]

        for domain
        in DOMAIN_ORDER
    ]


    perm_H, _ = kruskal(
        *perm_groups
    )


    permutation_H[
        permutation_id
    ] = perm_H


permutation_p = (

    1

    +

    np.sum(
        permutation_H
        >=
        H_statistic
    )

) / (

    N_PERMUTATIONS
    +
    1
)


permutation_file = (
    OUTPUT_ROOT /
    "Paper1_GPI_domain_permutation_H_distribution.csv"
)


pd.DataFrame({

    "Permutation":
        np.arange(
            1,
            N_PERMUTATIONS + 1
        ),

    "Kruskal_H":
        permutation_H

}).to_csv(
    permutation_file,
    index=False
)


# ============================================================
# 10. CLIFF'S DELTA
#
# delta =
#   P(X > Y) - P(X < Y)
#
# Positive delta:
#   first domain tends to have higher GPI.
#
# Negative delta:
#   second domain tends to have higher GPI.
# ============================================================

def cliffs_delta(
    x,
    y
):

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )


    # Exact pairwise comparison is acceptable here:
    # largest pair is well below one million comparisons.

    difference = (
        x[:, None]
        -
        y[None, :]
    )


    greater = np.sum(
        difference > 0
    )

    less = np.sum(
        difference < 0
    )


    delta = (

        greater
        -
        less

    ) / (

        len(x)
        *
        len(y)
    )


    return float(
        delta
    )


# ============================================================
# 11. CLIFF DELTA MAGNITUDE
#
# Conventional descriptors:
#
# |delta| < 0.147  negligible
# < 0.330          small
# < 0.474          medium
# >=0.474          large
# ============================================================

def cliff_magnitude(
    delta
):

    value = abs(
        delta
    )


    if value < 0.147:

        return "Negligible"

    elif value < 0.330:

        return "Small"

    elif value < 0.474:

        return "Medium"

    else:

        return "Large"


# ============================================================
# 12. PAIRWISE MANN-WHITNEY TESTS
# ============================================================

pairwise_records = []


for domain_a, domain_b in combinations(
    DOMAIN_ORDER,
    2
):

    x = groups[
        domain_a
    ]

    y = groups[
        domain_b
    ]


    U_statistic, raw_p = mannwhitneyu(

        x,
        y,

        alternative="two-sided",

        method="auto"
    )


    delta = cliffs_delta(
        x,
        y
    )


    pairwise_records.append({

        "Domain_A":
            domain_a,

        "Domain_B":
            domain_b,

        "N_A":
            len(
                x
            ),

        "N_B":
            len(
                y
            ),

        "Median_A":
            np.median(
                x
            ),

        "Median_B":
            np.median(
                y
            ),

        "Mann_Whitney_U":
            U_statistic,

        "Raw_p":
            raw_p,

        "Cliffs_delta_A_vs_B":
            delta,

        "Abs_Cliffs_delta":
            abs(
                delta
            ),

        "Effect_magnitude":
            cliff_magnitude(
                delta
            ),

        "Direction":
            (
                f"{domain_a} > {domain_b}"
                if delta > 0
                else
                f"{domain_b} > {domain_a}"
                if delta < 0
                else
                "No directional difference"
            )
    })


pairwise_df = pd.DataFrame(
    pairwise_records
)


# ============================================================
# 13. HOLM CORRECTION
#
# Step-down family-wise error control.
# ============================================================

def holm_adjust(
    p_values
):

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    m = len(
        p_values
    )


    order = np.argsort(
        p_values
    )


    sorted_p = p_values[
        order
    ]


    adjusted_sorted = np.zeros(
        m,
        dtype=float
    )


    running_max = 0.0


    for rank, p_value in enumerate(
        sorted_p
    ):

        multiplier = (
            m
            -
            rank
        )


        adjusted = min(
            1.0,
            multiplier
            *
            p_value
        )


        running_max = max(
            running_max,
            adjusted
        )


        adjusted_sorted[
            rank
        ] = running_max


    adjusted = np.empty(
        m,
        dtype=float
    )


    adjusted[
        order
    ] = adjusted_sorted


    return adjusted


pairwise_df[
    "Holm_adjusted_p"
] = holm_adjust(
    pairwise_df[
        "Raw_p"
    ].to_numpy()
)


pairwise_df[
    "Holm_significant_0.05"
] = (
    pairwise_df[
        "Holm_adjusted_p"
    ]
    <
    0.05
)


pairwise_file = (
    OUTPUT_ROOT /
    "Paper1_GPI_pairwise_MannWhitney_Holm_CliffsDelta.csv"
)


pairwise_df.to_csv(
    pairwise_file,
    index=False
)


# ============================================================
# 14. GLOBAL RANK STATISTICS
#
# Useful to characterize relative GPI ranking without
# imposing arbitrary categorical thresholds.
# ============================================================

analysis_df[
    "GPI_global_percentile"
] = analysis_df[
    GPI_COLUMN
].rank(
    method="average",
    pct=True
)


rank_summary = (

    analysis_df

    .groupby(
        DOMAIN_COLUMN
    )[
        "GPI_global_percentile"
    ]

    .agg(
        [
            "count",
            "mean",
            "median"
        ]
    )

    .reindex(
        DOMAIN_ORDER
    )

    .reset_index()
)


rank_summary[
    "Mean_global_percentile_x100"
] = (
    rank_summary[
        "mean"
    ]
    *
    100
)


rank_summary[
    "Median_global_percentile_x100"
] = (
    rank_summary[
        "median"
    ]
    *
    100
)


rank_summary_file = (
    OUTPUT_ROOT /
    "Paper1_GPI_domain_global_percentile_summary.csv"
)


rank_summary.to_csv(
    rank_summary_file,
    index=False
)


# ============================================================
# 15. GLOBAL SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    [{
        "N_cells":
            n_total,

        "N_domains":
            k_groups,

        "Kruskal_H":
            float(
                H_statistic
            ),

        "Kruskal_df":
            k_groups - 1,

        "Kruskal_asymptotic_p":
            float(
                H_p
            ),

        "Epsilon_squared":
            epsilon_squared,

        "Permutation_count":
            N_PERMUTATIONS,

        "Permutation_p":
            float(
                permutation_p
            ),

        "Pairwise_tests":
            len(
                pairwise_df
            ),

        "Holm_significant_pairs_0.05":
            int(
                pairwise_df[
                    "Holm_significant_0.05"
                ].sum()
            ),

        "Overall_GPI_mean":
            float(
                analysis_df[
                    GPI_COLUMN
                ].mean()
            ),

        "Overall_GPI_median":
            float(
                analysis_df[
                    GPI_COLUMN
                ].median()
            )
    }]
)


summary_file = (
    OUTPUT_ROOT /
    "Paper1_GPI_domain_global_test_summary.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# 16. SAVE ANALYSIS TABLE
# ============================================================

analysis_output_file = (
    OUTPUT_ROOT /
    "Paper1_GPI_domain_analysis_cells.csv"
)


analysis_df.to_csv(
    analysis_output_file,
    index=False
)


# ============================================================
# 17. PRINT RESULTS
# ============================================================

print(
    "\n\n" +
    "=" * 120
)

print(
    "GPI DESCRIPTIVE STATISTICS BY PB-MMAE DOMAIN"
)

print(
    "=" * 120
)


print(
    descriptive_df
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
    "GLOBAL GPI ↔ DOMAIN TEST"
)

print(
    "=" * 120
)


print(
    summary_df
    .round(
        6
    )
    .to_string(
        index=False
    )
)


print(
    "\nKruskal-Wallis H:",
    round(
        H_statistic,
        6
    )
)


print(
    "Asymptotic p:",
    H_p
)


print(
    "Epsilon-squared:",
    round(
        epsilon_squared,
        6
    )
)


print(
    "Permutation p:",
    permutation_p
)


print(
    "\n" +
    "=" * 120
)

print(
    "PAIRWISE DOMAIN DIFFERENCES"
)

print(
    "=" * 120
)


print(
    pairwise_df[
        [
            "Domain_A",
            "Domain_B",
            "Median_A",
            "Median_B",
            "Mann_Whitney_U",
            "Raw_p",
            "Holm_adjusted_p",
            "Cliffs_delta_A_vs_B",
            "Effect_magnitude",
            "Direction",
            "Holm_significant_0.05"
        ]
    ]
    .round(
        6
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
    "GLOBAL GPI PERCENTILE POSITION BY DOMAIN"
)

print(
    "=" * 120
)


print(
    rank_summary[
        [
            DOMAIN_COLUMN,
            "count",
            "Mean_global_percentile_x100",
            "Median_global_percentile_x100"
        ]
    ]
    .round(
        2
    )
    .to_string(
        index=False
    )
)


print(
    "\nFiles saved in:"
)

print(
    OUTPUT_ROOT
)


print(
    "\nIMPORTANT:"
)

print(
    "Continuous GPI was used for inferential testing."
)

print(
    "No rounded GPI class was used to determine significance."
)

print(
    "GPI remained completely withheld from PB-MMAE "
    "training and clustering."
)

print(
    "The permutation procedure is a label-permutation test "
    "and does not explicitly remove spatial autocorrelation; "
    "effect sizes therefore remain important."
)

print(
    "=" * 120
)