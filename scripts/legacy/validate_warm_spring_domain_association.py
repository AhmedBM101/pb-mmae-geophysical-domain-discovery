import os
# ============================================================
# Paper 1 — Warm-Spring ↔ PB-MMAE Domain Spatial Validation
#
# PRIMARY TEST:
#   Exhaustive spatial translation of the entire six-spring
#   configuration across the frozen PB-MMAE consensus raster.
#
#   This preserves:
#       - relative spring geometry
#       - inter-spring spacing
#       - PB-MMAE spatial organization
#
# SECONDARY TEST:
#   Exact finite-population test conditional on the five
#   springs that fall on resolved PB-MMAE domains.
#
# IMPORTANT:
#   No spring information was used during PB-MMAE training,
#   clustering, k selection, or meta-domain construction.
#
# Consensus coding:
#       -1 = tied/unresolved
#        0 = M0
#        1 = M1
#        2 = M2
#        3 = M3
#    -9999 = raster NoData
# ============================================================


import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"


from pathlib import Path
from math import comb

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(os.environ["PBMMAE_PROJECT_ROOT"]).expanduser().resolve()


DOMAIN_RASTER = (
    PROJECT_ROOT /
    "03_outputs" /
    "models" /
    "PBMMAE" /
    "final_4domain_consensus" /
    "PBMMAE_4domain_consensus_4km.tif"
)


SPRING_MASTER = (
    PROJECT_ROOT /
    "03_outputs" /
    "external_validation" /
    "Paper1_warm_spring_external_validation_MASTER.csv"
)


SPRING_SHP = (
    PROJECT_ROOT /
    "02_data" /
    "external_validation" /
    "warm_springs" /
    "Warm_spring_XY.shp"
)


DOMAIN_MASTER = (
    PROJECT_ROOT /
    "03_outputs" /
    "external_validation" /
    "Paper1_PBMMAE_domain_external_validation_MASTER.csv"
)


OUTPUT_ROOT = (
    PROJECT_ROOT /
    "03_outputs" /
    "external_validation" /
    "warm_spring_domain_validation"
)


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. FILE CHECK
# ============================================================

for path in [
    DOMAIN_RASTER,
    SPRING_MASTER,
    SPRING_SHP,
    DOMAIN_MASTER
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


# ============================================================
# 3. LOAD EXISTING TABLES
# ============================================================

spring_table = pd.read_csv(
    SPRING_MASTER
)


domain_table = pd.read_csv(
    DOMAIN_MASTER
)


required_spring_columns = [

    "Spring_name",
    "PBMMAE_domain",
    "PBMMAE_domain_tied"
]


for column in required_spring_columns:

    if column not in spring_table.columns:

        raise RuntimeError(
            f"Missing spring column: {column}"
        )


if "PBMMAE_domain" not in domain_table.columns:

    raise RuntimeError(
        "PBMMAE_domain absent from domain master table."
    )


# ============================================================
# 4. OBSERVED SPRING CONFIGURATION
# ============================================================

resolved_springs = spring_table[

    spring_table[
        "PBMMAE_domain"
    ].notna()

].copy()


n_springs_total = len(
    spring_table
)


n_resolved_observed = len(
    resolved_springs
)


n_tied_observed = int(
    spring_table[
        "PBMMAE_domain_tied"
    ].sum()
)


observed_domain_counts = (

    resolved_springs[
        "PBMMAE_domain"
    ]

    .value_counts()

    .reindex(
        [
            "M0",
            "M1",
            "M2",
            "M3"
        ],
        fill_value=0
    )
)


observed_max_count = int(
    observed_domain_counts.max()
)


observed_dominant_domain = (
    observed_domain_counts.idxmax()
)


observed_concentration_fraction = (

    observed_max_count /
    n_resolved_observed
)


print(
    "=" * 120
)

print(
    "PAPER 1 — WARM-SPRING ↔ PB-MMAE DOMAIN VALIDATION"
)

print(
    "=" * 120
)


print(
    "\nObserved spring configuration:"
)

print(
    f"Total springs            = {n_springs_total}"
)

print(
    f"Resolved springs         = {n_resolved_observed}"
)

print(
    f"Tied/unresolved springs  = {n_tied_observed}"
)

print(
    "\nResolved spring domain counts:"
)

print(
    observed_domain_counts.to_string()
)


print(
    "\nDominant resolved domain:",
    observed_dominant_domain
)


print(
    "Maximum same-domain count:",
    observed_max_count
)


print(
    "Concentration fraction:",
    round(
        observed_concentration_fraction,
        6
    )
)


if n_springs_total != 6:

    print(
        "\nWARNING: expected six mapped warm springs."
    )


if n_resolved_observed != 5:

    print(
        "\nWARNING: expected five resolved warm springs."
    )


# ============================================================
# 5. BACKGROUND DOMAIN OCCUPANCY
# ============================================================

domain_counts = (

    domain_table[
        "PBMMAE_domain"
    ]

    .value_counts()

    .reindex(
        [
            "M0",
            "M1",
            "M2",
            "M3"
        ],
        fill_value=0
    )
)


N_RESOLVED_CELLS = int(
    domain_counts.sum()
)


background_df = pd.DataFrame({

    "PBMMAE_domain":
        domain_counts.index,

    "Resolved_cells":
        domain_counts.values
})


background_df[
    "Resolved_fraction"
] = (

    background_df[
        "Resolved_cells"
    ]
    /
    N_RESOLVED_CELLS
)


background_file = (
    OUTPUT_ROOT /
    "Paper1_warm_spring_background_domain_occupancy.csv"
)


background_df.to_csv(
    background_file,
    index=False
)


print(
    "\n" +
    "=" * 120
)

print(
    "RESOLVED DOMAIN BACKGROUND"
)

print(
    "=" * 120
)


print(
    background_df
    .round(
        6
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 6. SECONDARY EXACT FINITE-POPULATION TEST
#
# Question 1:
#
# Conditional on drawing five resolved domain cells randomly
# without replacement, what is the probability that all five
# come from M2?
#
# Question 2:
#
# Because M2 was observed after inspecting the spring result,
# a more conservative omnibus question is:
#
# What is the probability that all five randomly drawn
# resolved cells belong to ANY one of the four domains?
#
# This avoids treating M2 as if it had been specified before
# examining the spring locations.
# ============================================================

if n_resolved_observed > N_RESOLVED_CELLS:

    raise RuntimeError(
        "Resolved spring count exceeds resolved raster cells."
    )


denominator = comb(
    N_RESOLVED_CELLS,
    n_resolved_observed
)


m2_cells = int(
    domain_counts[
        "M2"
    ]
)


if m2_cells >= n_resolved_observed:

    exact_p_all_M2 = (

        comb(
            m2_cells,
            n_resolved_observed
        )
        /
        denominator
    )

else:

    exact_p_all_M2 = 0.0


# ------------------------------------------------------------
# Omnibus:
# all five in ANY one domain.
#
# These events are mutually exclusive when n > 0.
# ------------------------------------------------------------

omnibus_numerator = 0


for domain in domain_counts.index:

    count = int(
        domain_counts[
            domain
        ]
    )

    if count >= n_resolved_observed:

        omnibus_numerator += comb(
            count,
            n_resolved_observed
        )


exact_p_all_same_any_domain = (

    omnibus_numerator /
    denominator
)


# ============================================================
# 7. READ DOMAIN RASTER
# ============================================================

with rasterio.open(
    DOMAIN_RASTER
) as src:

    domain_array = src.read(
        1
    )

    raster_crs = src.crs

    transform = src.transform

    height = src.height

    width = src.width

    nodata = src.nodata


# ============================================================
# 8. LOAD ORIGINAL SPRING GEOMETRIES
# ============================================================

springs_gdf = gpd.read_file(
    SPRING_SHP
)


if springs_gdf.crs is None:

    raise RuntimeError(
        "Warm-spring shapefile has no CRS."
    )


if springs_gdf.crs != raster_crs:

    springs_gdf = springs_gdf.to_crs(
        raster_crs
    )


if len(
    springs_gdf
) != n_springs_total:

    raise RuntimeError(
        "Spring shapefile feature count does not match "
        "the spring validation table."
    )


# ============================================================
# 9. CONVERT SPRINGS TO RASTER ROW/COLUMN
# ============================================================

spring_rows = []

spring_cols = []


for geometry in springs_gdf.geometry:

    row, col = rasterio.transform.rowcol(

        transform,

        geometry.x,
        geometry.y
    )


    spring_rows.append(
        int(
            row
        )
    )

    spring_cols.append(
        int(
            col
        )
    )


spring_rows = np.asarray(
    spring_rows,
    dtype=int
)


spring_cols = np.asarray(
    spring_cols,
    dtype=int
)


print(
    "\nOriginal spring raster positions:"
)


for idx in range(
    n_springs_total
):

    spring_name = spring_table.iloc[
        idx
    ][
        "Spring_name"
    ]


    print(
        f"{spring_name:<12} "
        f"row={spring_rows[idx]:>3}, "
        f"col={spring_cols[idx]:>3}"
    )


# ============================================================
# 10. VERIFY DIRECT RASTER SAMPLING
#
# This ensures that spring row/column conversion reproduces
# the already-generated spring master table.
# ============================================================

original_raw_values = domain_array[

    spring_rows,
    spring_cols
]


print(
    "\nDirect raster values at original springs:"
)

print(
    original_raw_values
)


expected_original_pattern = {

    -1:
        n_tied_observed,

    0:
        int(
            observed_domain_counts[
                "M0"
            ]
        ),

    1:
        int(
            observed_domain_counts[
                "M1"
            ]
        ),

    2:
        int(
            observed_domain_counts[
                "M2"
            ]
        ),

    3:
        int(
            observed_domain_counts[
                "M3"
            ]
        )
}


actual_original_pattern = {

    value:
        int(
            np.sum(
                original_raw_values == value
            )
        )

    for value in [
        -1,
        0,
        1,
        2,
        3
    ]
}


print(
    "\nExpected spring pattern:",
    expected_original_pattern
)

print(
    "Raster spring pattern:  ",
    actual_original_pattern
)


if actual_original_pattern != expected_original_pattern:

    raise RuntimeError(
        "Direct raster sampling does not reproduce "
        "the spring master table."
    )


print(
    "\nSpring raster audit passed."
)


# ============================================================
# 11. SPATIAL TRANSLATION TEST
#
# We now move the ENTIRE six-spring configuration by every
# possible integer number of 4-km cells.
#
# Relative spring geometry is unchanged.
#
# A translated configuration is accepted only when:
#
#   1. all six locations remain inside the raster;
#
#   2. all six fall within the PB-MMAE consensus-support
#      region, i.e. each cell is one of:
#
#          -1, 0, 1, 2, 3
#
#   3. at least five of six locations have resolved
#      M0-M3 assignments.
#
# This makes the translated configurations comparable to
# the observed case:
#
#   6 supported springs
#   5 resolved
#   1 tied
#
# We do NOT require exactly one tie because doing so would
# make the null unnecessarily conditional on the observed
# uncertainty realization.
# ============================================================

min_row = int(
    spring_rows.min()
)

max_row = int(
    spring_rows.max()
)

min_col = int(
    spring_cols.min()
)

max_col = int(
    spring_cols.max()
)


dr_min = -min_row

dr_max = (
    height - 1
    -
    max_row
)


dc_min = -min_col

dc_max = (
    width - 1
    -
    max_col
)


valid_support_codes = np.array(
    [
        -1,
        0,
        1,
        2,
        3
    ]
)


translation_records = []


for dr in range(
    dr_min,
    dr_max + 1
):

    translated_rows = (
        spring_rows
        +
        dr
    )


    for dc in range(
        dc_min,
        dc_max + 1
    ):

        translated_cols = (
            spring_cols
            +
            dc
        )


        values = domain_array[

            translated_rows,
            translated_cols
        ]


        # ----------------------------------------------------
        # All six must fall in consensus-supported cells.
        # ----------------------------------------------------

        if not np.all(
            np.isin(
                values,
                valid_support_codes
            )
        ):

            continue


        resolved_values = values[

            np.isin(
                values,
                [
                    0,
                    1,
                    2,
                    3
                ]
            )
        ]


        n_resolved = len(
            resolved_values
        )


        n_tied = int(
            np.sum(
                values == -1
            )
        )


        # ----------------------------------------------------
        # Need at least as many resolved assignments as the
        # observed configuration.
        # ----------------------------------------------------

        if n_resolved < n_resolved_observed:

            continue


        translated_counts = [

            int(
                np.sum(
                    resolved_values == code
                )
            )

            for code in [
                0,
                1,
                2,
                3
            ]
        ]


        max_same_domain = max(
            translated_counts
        )


        concentration_fraction = (

            max_same_domain
            /
            n_resolved
        )


        dominant_code = int(
            np.argmax(
                translated_counts
            )
        )


        translation_records.append({

            "Row_shift_cells":
                dr,

            "Col_shift_cells":
                dc,

            "Row_shift_km":
                dr * 4,

            "Col_shift_km":
                dc * 4,

            "N_supported":
                6,

            "N_resolved":
                n_resolved,

            "N_tied":
                n_tied,

            "M0_count":
                translated_counts[
                    0
                ],

            "M1_count":
                translated_counts[
                    1
                ],

            "M2_count":
                translated_counts[
                    2
                ],

            "M3_count":
                translated_counts[
                    3
                ],

            "Max_same_domain_count":
                max_same_domain,

            "Dominant_domain":
                f"M{dominant_code}",

            "Concentration_fraction":
                concentration_fraction,

            "Is_original_position":
                bool(
                    dr == 0
                    and
                    dc == 0
                )
        })


translation_df = pd.DataFrame(
    translation_records
)


if len(
    translation_df
) == 0:

    raise RuntimeError(
        "No valid translated spring configurations found."
    )


# ============================================================
# 12. CONFIRM ORIGINAL TRANSLATION EXISTS
# ============================================================

original_translation = translation_df[

    translation_df[
        "Is_original_position"
    ]

]


if len(
    original_translation
) != 1:

    raise RuntimeError(
        "Original spring configuration was not uniquely "
        "identified among valid translations."
    )


# ============================================================
# 13. PRIMARY SPATIAL TRANSLATION P-VALUE
#
# Statistic:
#
# maximum proportion of resolved springs occupying one
# PB-MMAE domain.
#
# Observed = 5/5 = 1.0.
#
# We compare against all NON-ORIGINAL translations.
#
# Plus-one correction:
#
# p = (1 + N_null >= observed) / (1 + N_null)
# ============================================================

null_translation_df = translation_df[

    ~translation_df[
        "Is_original_position"
    ]

].copy()


n_translation_null = len(
    null_translation_df
)


n_translation_extreme = int(

    np.sum(

        null_translation_df[
            "Concentration_fraction"
        ]

        >=

        observed_concentration_fraction
    )
)


spatial_translation_p = (

    1
    +
    n_translation_extreme

) / (

    1
    +
    n_translation_null
)


# ============================================================
# 14. SECONDARY M2-SPECIFIC TRANSLATION RESULT
#
# Descriptive/sensitivity result only because M2 was observed
# after examining the spring allocation.
#
# Question:
#   How often do translated configurations place at least
#   five springs in M2?
# ============================================================

n_m2_extreme = int(

    np.sum(

        null_translation_df[
            "M2_count"
        ]

        >=

        observed_domain_counts[
            "M2"
        ]
    )
)


m2_translation_p = (

    1
    +
    n_m2_extreme

) / (

    1
    +
    n_translation_null
)


# ============================================================
# 15. TRANSLATION DISTRIBUTION TABLE
# ============================================================

translation_file = (
    OUTPUT_ROOT /
    "Paper1_warm_spring_spatial_translation_results.csv"
)


translation_df.to_csv(
    translation_file,
    index=False
)


# ============================================================
# 16. TRANSLATION SUMMARY BY CONCENTRATION
# ============================================================

translation_distribution_df = (

    null_translation_df

    .groupby(
        [
            "N_resolved",
            "Max_same_domain_count",
            "Concentration_fraction"
        ]
    )

    .size()

    .reset_index(
        name="Translation_count"
    )

    .sort_values(
        [
            "N_resolved",
            "Concentration_fraction"
        ]
    )
)


translation_distribution_file = (
    OUTPUT_ROOT /
    "Paper1_warm_spring_translation_concentration_distribution.csv"
)


translation_distribution_df.to_csv(
    translation_distribution_file,
    index=False
)


# ============================================================
# 17. EXACT TEST SUMMARY
# ============================================================

exact_summary_df = pd.DataFrame(
    [
        {
            "Test":
                "All 5 resolved springs in M2",

            "N_resolved_springs":
                n_resolved_observed,

            "Resolved_background_cells":
                N_RESOLVED_CELLS,

            "Relevant_background_cells":
                m2_cells,

            "Exact_probability":
                exact_p_all_M2,

            "Interpretation_scope":
                "M2-specific; secondary"
        },

        {
            "Test":
                "All 5 resolved springs in any one domain",

            "N_resolved_springs":
                n_resolved_observed,

            "Resolved_background_cells":
                N_RESOLVED_CELLS,

            "Relevant_background_cells":
                np.nan,

            "Exact_probability":
                exact_p_all_same_any_domain,

            "Interpretation_scope":
                "Omnibus finite-population concentration"
        }
    ]
)


exact_summary_file = (
    OUTPUT_ROOT /
    "Paper1_warm_spring_exact_finite_population_tests.csv"
)


exact_summary_df.to_csv(
    exact_summary_file,
    index=False
)


# ============================================================
# 18. GLOBAL SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    [{
        "Total_springs":
            n_springs_total,

        "Resolved_springs":
            n_resolved_observed,

        "Tied_springs":
            n_tied_observed,

        "Observed_dominant_domain":
            observed_dominant_domain,

        "Observed_max_same_domain":
            observed_max_count,

        "Observed_concentration_fraction":
            observed_concentration_fraction,

        "Resolved_background_cells":
            N_RESOLVED_CELLS,

        "M0_background_cells":
            int(
                domain_counts[
                    "M0"
                ]
            ),

        "M1_background_cells":
            int(
                domain_counts[
                    "M1"
                ]
            ),

        "M2_background_cells":
            int(
                domain_counts[
                    "M2"
                ]
            ),

        "M3_background_cells":
            int(
                domain_counts[
                    "M3"
                ]
            ),

        "Exact_p_all5_M2":
            exact_p_all_M2,

        "Exact_p_all5_same_any_domain":
            exact_p_all_same_any_domain,

        "Valid_spatial_translations_including_original":
            len(
                translation_df
            ),

        "Null_spatial_translations":
            n_translation_null,

        "Null_translations_equally_or_more_concentrated":
            n_translation_extreme,

        "Spatial_translation_p_omnibus":
            spatial_translation_p,

        "Null_translations_with_at_least5_M2":
            n_m2_extreme,

        "Spatial_translation_p_M2_specific":
            m2_translation_p
    }]
)


summary_file = (
    OUTPUT_ROOT /
    "Paper1_warm_spring_domain_validation_summary.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# 19. SAVE SPRING ASSIGNMENTS
# ============================================================

spring_assignments_file = (
    OUTPUT_ROOT /
    "Paper1_warm_spring_observed_domain_assignments.csv"
)


spring_table.to_csv(
    spring_assignments_file,
    index=False
)


# ============================================================
# 20. PRINT RESULTS
# ============================================================

print(
    "\n\n" +
    "=" * 120
)

print(
    "EXACT FINITE-POPULATION RESULTS"
)

print(
    "=" * 120
)


print(
    exact_summary_df
    .round(
        8
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
    "SPATIAL TRANSLATION TEST"
)

print(
    "=" * 120
)


print(
    "Valid translations including original:",
    len(
        translation_df
    )
)


print(
    "Null translations:",
    n_translation_null
)


print(
    "Null translations equally/more concentrated:",
    n_translation_extreme
)


print(
    "Primary omnibus spatial-translation p:",
    spatial_translation_p
)


print(
    "\nM2-specific translated configurations with >=5 M2:",
    n_m2_extreme
)


print(
    "Secondary M2-specific translation p:",
    m2_translation_p
)


print(
    "\n" +
    "=" * 120
)

print(
    "FINAL SUMMARY"
)

print(
    "=" * 120
)


print(
    summary_df
    .round(
        8
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
    "\nIMPORTANT INTERPRETATION:"
)

print(
    "The primary result is the omnibus spatial-translation "
    "test, because it preserves the six-spring spatial "
    "configuration and does not pre-select M2."
)

print(
    "The finite-population exact tests are complementary "
    "non-spatial benchmarks."
)

print(
    "The small spring inventory must remain an explicit "
    "limitation regardless of statistical significance."
)

print(
    "=" * 120
)