import os
# ============================================================
# Paper 1 — Formal Geology ↔ PB-MMAE Domain Validation
#
# Input:
#   Paper1_PBMMAE_domain_external_validation_MASTER.csv
#
# Tests:
#   1. Pearson chi-square test
#   2. Cramer's V
#   3. 10,000-permutation significance test
#   4. Expected counts
#   5. Pearson standardized residuals
#   6. Adjusted standardized residuals
#   7. Observed / expected enrichment ratios
#   8. Domain-specific geological enrichment rankings
#
# IMPORTANT:
#   Geology remained completely withheld during PB-MMAE
#   training and clustering.
# ============================================================

import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency


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
    "geology_domain_validation"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

DOMAIN_COLUMN = "PBMMAE_domain"
GEOLOGY_COLUMN = "unit_name"

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
    GEOLOGY_COLUMN,
    "Geology_boundary_ambiguous"
]


for column in required_columns:

    if column not in df.columns:

        raise RuntimeError(
            f"Required column absent: {column}"
        )


# ============================================================
# 4. STRICT ANALYSIS SUBSET
#
# Only:
#   - resolved PB-MMAE domains
#   - valid geology
#   - non-boundary-ambiguous geology
# ============================================================

analysis_df = df[

    df[
        DOMAIN_COLUMN
    ].notna()

    &

    df[
        GEOLOGY_COLUMN
    ].notna()

    &

    (
        ~df[
            "Geology_boundary_ambiguous"
        ].astype(bool)
    )

].copy()


analysis_df[
    DOMAIN_COLUMN
] = analysis_df[
    DOMAIN_COLUMN
].astype(
    str
)


analysis_df[
    GEOLOGY_COLUMN
] = analysis_df[
    GEOLOGY_COLUMN
].astype(
    str
)


print(
    "=" * 120
)

print(
    "PAPER 1 — GEOLOGY ↔ PB-MMAE DOMAIN VALIDATION"
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

print(
    "Geological units:",
    analysis_df[
        GEOLOGY_COLUMN
    ].nunique()
)


# ============================================================
# 5. CONTINGENCY TABLE
# ============================================================

observed_df = pd.crosstab(

    analysis_df[
        DOMAIN_COLUMN
    ],

    analysis_df[
        GEOLOGY_COLUMN
    ]
)


observed = observed_df.to_numpy(
    dtype=float
)


observed_file = (
    OUTPUT_ROOT /
    "Paper1_geology_domain_observed_counts.csv"
)


observed_df.to_csv(
    observed_file
)


print(
    "\nObserved contingency table:"
)

print(
    observed_df.to_string()
)


# ============================================================
# 6. PEARSON CHI-SQUARE
# ============================================================

chi2_stat, chi2_p, dof, expected = (
    chi2_contingency(
        observed,
        correction=False
    )
)


expected_df = pd.DataFrame(

    expected,

    index=observed_df.index,

    columns=observed_df.columns
)


expected_file = (
    OUTPUT_ROOT /
    "Paper1_geology_domain_expected_counts.csv"
)


expected_df.to_csv(
    expected_file
)


# ============================================================
# 7. CRAMER'S V
#
# V = sqrt( chi2 / [n * min(r-1, c-1)] )
# ============================================================

n = observed.sum()

r, c = observed.shape

denominator = (
    n
    *
    min(
        r - 1,
        c - 1
    )
)


if denominator <= 0:

    raise RuntimeError(
        "Invalid contingency dimensions for Cramer's V."
    )


cramers_v = np.sqrt(
    chi2_stat /
    denominator
)


# ============================================================
# 8. EXPECTED-COUNT AUDIT
# ============================================================

n_expected_lt5 = int(
    np.sum(
        expected < 5
    )
)

fraction_expected_lt5 = float(
    np.mean(
        expected < 5
    )
)

minimum_expected = float(
    expected.min()
)


# ============================================================
# 9. PEARSON STANDARDIZED RESIDUALS
#
# (O - E) / sqrt(E)
# ============================================================

pearson_residuals = (

    observed
    -
    expected

) / np.sqrt(
    expected
)


pearson_residuals_df = pd.DataFrame(

    pearson_residuals,

    index=observed_df.index,

    columns=observed_df.columns
)


pearson_residuals_file = (
    OUTPUT_ROOT /
    "Paper1_geology_domain_Pearson_residuals.csv"
)


pearson_residuals_df.to_csv(
    pearson_residuals_file
)


# ============================================================
# 10. ADJUSTED STANDARDIZED RESIDUALS
#
# adjusted residual:
#
# (O - E) /
# sqrt[ E * (1-row_prop) * (1-col_prop) ]
#
# Approximate interpretation:
#   |residual| >= 1.96   notable
#   |residual| >= 2.58   strong
#   |residual| >= 3.29   very strong
# ============================================================

row_totals = observed.sum(
    axis=1,
    keepdims=True
)

col_totals = observed.sum(
    axis=0,
    keepdims=True
)


row_prop = (
    row_totals /
    n
)

col_prop = (
    col_totals /
    n
)


adjusted_denominator = np.sqrt(

    expected

    *

    (
        1.0 -
        row_prop
    )

    *

    (
        1.0 -
        col_prop
    )
)


adjusted_residuals = np.divide(

    observed - expected,

    adjusted_denominator,

    out=np.zeros_like(
        observed,
        dtype=float
    ),

    where=(
        adjusted_denominator > 0
    )
)


adjusted_residuals_df = pd.DataFrame(

    adjusted_residuals,

    index=observed_df.index,

    columns=observed_df.columns
)


adjusted_residuals_file = (
    OUTPUT_ROOT /
    "Paper1_geology_domain_adjusted_standardized_residuals.csv"
)


adjusted_residuals_df.to_csv(
    adjusted_residuals_file
)


# ============================================================
# 11. ENRICHMENT RATIO
#
# O / E
#
# >1 = overrepresented
# <1 = underrepresented
# ============================================================

enrichment = np.divide(

    observed,

    expected,

    out=np.zeros_like(
        observed,
        dtype=float
    ),

    where=(
        expected > 0
    )
)


enrichment_df = pd.DataFrame(

    enrichment,

    index=observed_df.index,

    columns=observed_df.columns
)


enrichment_file = (
    OUTPUT_ROOT /
    "Paper1_geology_domain_enrichment_ratio.csv"
)


enrichment_df.to_csv(
    enrichment_file
)


# ============================================================
# 12. PERMUTATION TEST
#
# Null:
#   geological labels are unrelated to PB-MMAE domains.
#
# Domain labels remain spatially fixed.
# Geological labels are permuted across the resolved cells.
#
# Statistic:
#   Pearson chi-square
#
# NOTE:
#   This is a label-permutation test, not a spatially
#   constrained permutation. It tests categorical dependence
#   given this sampled cell set.
# ============================================================

domain_values = analysis_df[
    DOMAIN_COLUMN
].to_numpy()

geology_values = analysis_df[
    GEOLOGY_COLUMN
].to_numpy()


domain_categories = list(
    observed_df.index
)

geology_categories = list(
    observed_df.columns
)


domain_to_idx = {
    value: idx
    for idx, value
    in enumerate(
        domain_categories
    )
}


geology_to_idx = {
    value: idx
    for idx, value
    in enumerate(
        geology_categories
    )
}


domain_idx = np.array(
    [
        domain_to_idx[
            value
        ]
        for value
        in domain_values
    ],
    dtype=int
)


geology_idx_original = np.array(
    [
        geology_to_idx[
            value
        ]
        for value
        in geology_values
    ],
    dtype=int
)


permutation_chi2 = np.zeros(
    N_PERMUTATIONS,
    dtype=float
)


for permutation_id in range(
    N_PERMUTATIONS
):

    permuted_geology_idx = rng.permutation(
        geology_idx_original
    )


    perm_table = np.zeros(
        (
            r,
            c
        ),
        dtype=float
    )


    np.add.at(

        perm_table,

        (
            domain_idx,
            permuted_geology_idx
        ),

        1
    )


    # Row and column marginals are fixed under permutation,
    # so expected values are identical to the observed-table
    # expected matrix.

    permutation_chi2[
        permutation_id
    ] = np.sum(

        (
            perm_table
            -
            expected
        ) ** 2

        /

        expected
    )


# Plus-one correction.

permutation_p = (

    1

    +

    np.sum(
        permutation_chi2
        >=
        chi2_stat
    )

) / (

    N_PERMUTATIONS
    +
    1
)


permutation_file = (
    OUTPUT_ROOT /
    "Paper1_geology_domain_permutation_chi2_distribution.csv"
)


pd.DataFrame(
    {
        "Permutation":
            np.arange(
                1,
                N_PERMUTATIONS + 1
            ),

        "Chi_square":
            permutation_chi2
    }
).to_csv(
    permutation_file,
    index=False
)


# ============================================================
# 13. LONG-FORM CELL ASSOCIATION TABLE
# ============================================================

long_records = []


for i, domain in enumerate(
    observed_df.index
):

    for j, geology_unit in enumerate(
        observed_df.columns
    ):

        long_records.append({

            "PBMMAE_domain":
                domain,

            "Geology_unit":
                geology_unit,

            "Observed":
                observed[
                    i,
                    j
                ],

            "Expected":
                expected[
                    i,
                    j
                ],

            "Enrichment_ratio":
                enrichment[
                    i,
                    j
                ],

            "Pearson_residual":
                pearson_residuals[
                    i,
                    j
                ],

            "Adjusted_standardized_residual":
                adjusted_residuals[
                    i,
                    j
                ],

            "Abs_adjusted_residual":
                abs(
                    adjusted_residuals[
                        i,
                        j
                    ]
                ),

            "Direction":
                (
                    "Enriched"
                    if enrichment[
                        i,
                        j
                    ] > 1
                    else
                    "Depleted"
                )
        })


long_df = pd.DataFrame(
    long_records
)


long_file = (
    OUTPUT_ROOT /
    "Paper1_geology_domain_cellwise_association_metrics.csv"
)


long_df.to_csv(
    long_file,
    index=False
)


# ============================================================
# 14. STRONG ASSOCIATIONS
#
# Retain cells with |adjusted residual| >= 1.96.
# ============================================================

strong_df = long_df[

    long_df[
        "Abs_adjusted_residual"
    ] >= 1.96

].copy()


strong_df = strong_df.sort_values(

    [
        "PBMMAE_domain",
        "Abs_adjusted_residual"
    ],

    ascending=[
        True,
        False
    ]
)


strong_file = (
    OUTPUT_ROOT /
    "Paper1_geology_domain_significant_residual_associations.csv"
)


strong_df.to_csv(
    strong_file,
    index=False
)


# ============================================================
# 15. TOP ENRICHED GEOLOGY PER DOMAIN
#
# Require at least 5 observed cells to avoid interpreting
# tiny numerical enrichments from one-cell categories.
# ============================================================

top_enrichment_records = []


for domain in observed_df.index:

    subset = long_df[

        (
            long_df[
                "PBMMAE_domain"
            ] == domain
        )

        &

        (
            long_df[
                "Observed"
            ] >= 5
        )

    ].copy()


    subset = subset.sort_values(

        [
            "Enrichment_ratio",
            "Abs_adjusted_residual"
        ],

        ascending=[
            False,
            False
        ]
    )


    subset = subset.head(
        5
    )


    for rank, (_, row) in enumerate(
        subset.iterrows(),
        start=1
    ):

        top_enrichment_records.append({

            "PBMMAE_domain":
                domain,

            "Rank":
                rank,

            "Geology_unit":
                row[
                    "Geology_unit"
                ],

            "Observed":
                row[
                    "Observed"
                ],

            "Expected":
                row[
                    "Expected"
                ],

            "Enrichment_ratio":
                row[
                    "Enrichment_ratio"
                ],

            "Adjusted_standardized_residual":
                row[
                    "Adjusted_standardized_residual"
                ]
        })


top_enrichment_df = pd.DataFrame(
    top_enrichment_records
)


top_enrichment_file = (
    OUTPUT_ROOT /
    "Paper1_geology_domain_top_enrichments.csv"
)


top_enrichment_df.to_csv(
    top_enrichment_file,
    index=False
)


# ============================================================
# 16. GLOBAL SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    [{
        "N_cells":
            int(
                n
            ),

        "N_domains":
            int(
                r
            ),

        "N_geology_units":
            int(
                c
            ),

        "Chi_square":
            float(
                chi2_stat
            ),

        "Chi_square_df":
            int(
                dof
            ),

        "Chi_square_asymptotic_p":
            float(
                chi2_p
            ),

        "Cramers_V":
            float(
                cramers_v
            ),

        "Permutation_count":
            N_PERMUTATIONS,

        "Permutation_p":
            float(
                permutation_p
            ),

        "Minimum_expected_count":
            minimum_expected,

        "Expected_cells_lt5":
            n_expected_lt5,

        "Fraction_expected_cells_lt5":
            fraction_expected_lt5,

        "Strong_adjusted_residual_cells_abs_ge_1.96":
            int(
                len(
                    strong_df
                )
            )
    }]
)


summary_file = (
    OUTPUT_ROOT /
    "Paper1_geology_domain_association_summary.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# 17. DOMAIN OCCUPANCY BY GEOLOGY — COLUMN PERCENT
#
# Within each geological unit:
# what fraction belongs to each PB-MMAE domain?
# ============================================================

column_percent_df = (

    observed_df

    .div(
        observed_df.sum(
            axis=0
        ),
        axis=1
    )

    *
    100.0
)


column_percent_file = (
    OUTPUT_ROOT /
    "Paper1_geology_domain_column_percent.csv"
)


column_percent_df.to_csv(
    column_percent_file
)


# ============================================================
# 18. GEOLOGY COMPOSITION WITHIN DOMAIN — ROW PERCENT
# ============================================================

row_percent_df = (

    observed_df

    .div(
        observed_df.sum(
            axis=1
        ),
        axis=0
    )

    *
    100.0
)


row_percent_file = (
    OUTPUT_ROOT /
    "Paper1_geology_domain_row_percent.csv"
)


row_percent_df.to_csv(
    row_percent_file
)


# ============================================================
# 19. PRINT RESULTS
# ============================================================

print(
    "\n\n" +
    "=" * 120
)

print(
    "GLOBAL GEOLOGY ↔ DOMAIN ASSOCIATION"
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
    "\nInterpretation checks:"
)

print(
    f"Chi-square = {chi2_stat:.4f}"
)

print(
    f"df = {dof}"
)

print(
    f"Asymptotic p = {chi2_p:.8g}"
)

print(
    f"Cramer's V = {cramers_v:.4f}"
)

print(
    f"Permutation p = {permutation_p:.8g}"
)

print(
    f"Minimum expected count = {minimum_expected:.4f}"
)

print(
    f"Expected cells < 5 = "
    f"{n_expected_lt5}/{expected.size}"
)


print(
    "\n" +
    "=" * 120
)

print(
    "STRONGEST GEOLOGICAL ASSOCIATIONS"
)

print(
    "=" * 120
)


if len(
    strong_df
) == 0:

    print(
        "No |adjusted residual| >= 1.96."
    )

else:

    print(
        strong_df[
            [
                "PBMMAE_domain",
                "Geology_unit",
                "Observed",
                "Expected",
                "Enrichment_ratio",
                "Adjusted_standardized_residual"
            ]
        ]
        .head(
            40
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
    "TOP ENRICHMENTS BY DOMAIN"
)

print(
    "=" * 120
)


print(
    top_enrichment_df
    .round(
        4
    )
    .to_string(
        index=False
    )
)


print(
    "\nFiles saved to:"
)

print(
    OUTPUT_ROOT
)


print(
    "\nIMPORTANT:"
)

print(
    "Geology was not used in PB-MMAE training or clustering."
)

print(
    "Therefore this analysis measures post hoc geological "
    "consistency of the frozen domain solution."
)

print(
    "The permutation test permutes geological labels across "
    "the same resolved domain cells."
)

print(
    "=" * 120
)