import os
# ============================================================
# Paper 1 — Physical characterization of frozen PB-MMAE
# 4-domain solution
#
# Purpose:
#   Characterize M0–M3 using the ORIGINAL 14 physical
#   geophysical channels.
#
# Important methodological rules:
#   - No geology
#   - No warm springs
#   - No GPI
#   - No latent coordinates used for interpretation
#   - Tied consensus cells are excluded
#   - Domain IDs are categorical, not ordinal
#
# Statistics:
#   - N
#   - mean / SD
#   - median / Q1 / Q3 / IQR
#   - global robust-standardized domain medians
#   - Kruskal-Wallis H
#   - epsilon-squared effect size
#   - pairwise Cliff's delta
#   - physics-family robust signatures
# ============================================================

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from scipy.stats import kruskal


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(os.environ["PBMMAE_PROJECT_ROOT"]).expanduser().resolve()

CONSENSUS_FILE = (
    PROJECT_ROOT /
    "03_outputs" /
    "models" /
    "PBMMAE" /
    "final_4domain_consensus" /
    "PBMMAE_4domain_cell_consensus.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "03_outputs" /
    "models" /
    "PBMMAE" /
    "final_4domain_consensus" /
    "physical_characterization"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOCKED 14 CHANNELS
# ============================================================

CHANNELS = [

    "RTE_TMI",

    "CBG",
    "CBG_RES",
    "HGM",

    "K",
    "eTh",
    "eU",

    "CPD",

    "MAG_LD",
    "GRAV_LD",
    "DEM_LD",
    "ID",

    "DEM",
    "SLOPE"
]


PHYSICS_FAMILIES = {

    "Magnetic": [
        "RTE_TMI"
    ],

    "Gravity": [
        "CBG",
        "CBG_RES",
        "HGM"
    ],

    "Radiometric": [
        "K",
        "eTh",
        "eU"
    ],

    "Thermal": [
        "CPD"
    ],

    "Structural": [
        "MAG_LD",
        "GRAV_LD",
        "DEM_LD",
        "ID"
    ],

    "Terrain": [
        "DEM",
        "SLOPE"
    ]
}


DOMAINS = [
    0,
    1,
    2,
    3
]


# ============================================================
# 3. FIND ORIGINAL 14-CHANNEL ANALYTICAL CSV
#
# We do NOT guess the filename.
#
# Search project CSVs for a file containing all 14 channels
# plus either Cell_ID or Row+Col.
# ============================================================

print("=" * 100)
print("PB-MMAE 4-DOMAIN PHYSICAL CHARACTERIZATION")
print("=" * 100)

print(
    "\nSearching for original 14-channel analytical table..."
)


EXCLUDE_PARTS = [

    "baselines",
    "models",
    "physical_characterization",
    "spatial_cv",
    "patches"
]


candidate_files = []


for path in PROJECT_ROOT.rglob(
    "*.csv"
):

    path_string = str(
        path
    ).lower()


    if any(
        part.lower() in path_string
        for part in EXCLUDE_PARTS
    ):

        continue


    try:

        header = pd.read_csv(
            path,
            nrows=0
        )

    except Exception:

        continue


    columns = set(
        header.columns
    )


    has_channels = set(
        CHANNELS
    ).issubset(
        columns
    )


    has_spatial_key = (
        "Cell_ID" in columns
        or
        (
            "Row" in columns
            and
            "Col" in columns
        )
    )


    if (
        has_channels
        and
        has_spatial_key
    ):

        candidate_files.append(
            path
        )


print(
    "\nCandidate analytical tables found:",
    len(
        candidate_files
    )
)


for i, path in enumerate(
    candidate_files,
    start=1
):

    print(
        f"  [{i}] {path}"
    )


if len(
    candidate_files
) == 0:

    raise RuntimeError(
        "\nNo CSV containing all 14 locked physical channels "
        "and a spatial key was found.\n"
        "Send the console output to ChatGPT so the correct "
        "analytical table can be located."
    )


# ------------------------------------------------------------
# If multiple candidates exist, identify the most plausible
# original analytical table by preferring:
#   - largest number of rows
#   - absence of TRAIN / VALIDATION / scaled markers
# ------------------------------------------------------------

candidate_info = []


for path in candidate_files:

    try:

        temp = pd.read_csv(
            path
        )

    except Exception:

        continue


    name_lower = path.name.lower()


    penalty = 0


    for marker in [
        "train",
        "validation",
        "scaled",
        "buffer",
        "fold"
    ]:

        if marker in name_lower:

            penalty += 1


    candidate_info.append({

        "path":
            path,

        "n_rows":
            len(
                temp
            ),

        "penalty":
            penalty
    })


candidate_info = sorted(

    candidate_info,

    key=lambda x: (
        x[
            "penalty"
        ],
        -
        x[
            "n_rows"
        ]
    )
)


ANALYTICAL_FILE = (
    candidate_info[
        0
    ][
        "path"
    ]
)


print(
    "\nSelected analytical table:"
)

print(
    ANALYTICAL_FILE
)


print(
    "Rows:",
    candidate_info[
        0
    ][
        "n_rows"
    ]
)


# ============================================================
# 4. LOAD ORIGINAL ANALYTICAL DATA
# ============================================================

physical = pd.read_csv(
    ANALYTICAL_FILE
)


# ------------------------------------------------------------
# Construct Cell_ID if necessary
# ------------------------------------------------------------

if "Cell_ID" not in physical.columns:

    if not {
        "Row",
        "Col"
    }.issubset(
        physical.columns
    ):

        raise RuntimeError(
            "Analytical table has neither Cell_ID nor Row/Col."
        )


    physical[
        "Cell_ID"
    ] = [

        f"R{int(r):02d}_C{int(c):02d}"

        for r, c in zip(
            physical[
                "Row"
            ],
            physical[
                "Col"
            ]
        )
    ]


if physical[
    "Cell_ID"
].duplicated().any():

    raise RuntimeError(
        "Analytical table contains duplicate Cell_ID values."
    )


# Ensure physical channels numeric

for channel in CHANNELS:

    physical[
        channel
    ] = pd.to_numeric(
        physical[
            channel
        ],
        errors="coerce"
    )


print(
    "\nPhysical analytical cells:",
    len(
        physical
    )
)


# ============================================================
# 5. LOAD FROZEN 4-DOMAIN CONSENSUS
# ============================================================

if not CONSENSUS_FILE.exists():

    raise FileNotFoundError(
        CONSENSUS_FILE
    )


consensus = pd.read_csv(
    CONSENSUS_FILE
)


required_consensus = {

    "Cell_ID",
    "Consensus_domain_4",
    "Consensus_tie",
    "Agreement_fraction",
    "N_models"
}


if not required_consensus.issubset(
    consensus.columns
):

    missing = (
        required_consensus
        -
        set(
            consensus.columns
        )
    )

    raise RuntimeError(
        f"Missing consensus columns: {missing}"
    )


print(
    "\nConsensus cells:",
    len(
        consensus
    )
)


# ============================================================
# 6. EXCLUDE TIED / AMBIGUOUS CELLS
# ============================================================

consensus_unique = consensus[
    ~consensus[
        "Consensus_tie"
    ].astype(
        bool
    )
].copy()


consensus_unique = consensus_unique[
    consensus_unique[
        "Consensus_domain_4"
    ].isin(
        DOMAINS
    )
].copy()


print(
    "Unique-consensus cells retained:",
    len(
        consensus_unique
    )
)


# ============================================================
# 7. MERGE DOMAIN LABELS WITH PHYSICAL VALUES
# ============================================================

merge_columns = [

    "Cell_ID",
    "Consensus_domain_4",
    "Agreement_fraction",
    "N_models"
]


for optional_column in [
    "Row",
    "Col",
    "X",
    "Y",
    "Spatial_Fold"
]:

    if optional_column in consensus_unique.columns:

        merge_columns.append(
            optional_column
        )


merged = consensus_unique[
    merge_columns
].merge(

    physical[
        [
            "Cell_ID"
        ]
        +
        CHANNELS
    ],

    on="Cell_ID",

    how="left",

    validate="one_to_one"
)


print(
    "\nMerged unique-consensus cells:",
    len(
        merged
    )
)


missing_all = (
    merged[
        CHANNELS
    ]
    .isna()
    .all(
        axis=1
    )
    .sum()
)


print(
    "Cells missing all 14 physical variables:",
    int(
        missing_all
    )
)


if missing_all > 0:

    raise RuntimeError(
        "Some consensus Cell_ID values could not be matched "
        "to the original analytical table."
    )


# ============================================================
# 8. DATA COMPLETENESS AUDIT
# ============================================================

missing_records = []


for channel in CHANNELS:

    missing_records.append({

        "Channel":
            channel,

        "N_total":
            len(
                merged
            ),

        "N_valid":
            int(
                merged[
                    channel
                ].notna().sum()
            ),

        "N_missing":
            int(
                merged[
                    channel
                ].isna().sum()
            )
    })


missing_df = pd.DataFrame(
    missing_records
)


missing_file = (
    OUTPUT_DIR /
    "PBMMAE_4domain_physical_data_completeness.csv"
)


missing_df.to_csv(
    missing_file,
    index=False
)


# ============================================================
# 9. RAW DESCRIPTIVE STATISTICS BY DOMAIN
# ============================================================

descriptive_records = []


for domain in DOMAINS:

    subset = merged[
        merged[
            "Consensus_domain_4"
        ] == domain
    ]


    for channel in CHANNELS:

        values = (
            subset[
                channel
            ]
            .dropna()
            .to_numpy(
                dtype=float
            )
        )


        if len(
            values
        ) == 0:

            continue


        q1 = float(
            np.percentile(
                values,
                25
            )
        )


        median = float(
            np.median(
                values
            )
        )


        q3 = float(
            np.percentile(
                values,
                75
            )
        )


        descriptive_records.append({

            "Meta_domain":
                domain,

            "Channel":
                channel,

            "N":
                len(
                    values
                ),

            "Mean":
                float(
                    np.mean(
                        values
                    )
                ),

            "SD":
                float(
                    np.std(
                        values,
                        ddof=1
                    )
                )
                if len(
                    values
                ) > 1
                else np.nan,

            "Q1":
                q1,

            "Median":
                median,

            "Q3":
                q3,

            "IQR":
                q3 - q1,

            "Minimum":
                float(
                    np.min(
                        values
                    )
                ),

            "Maximum":
                float(
                    np.max(
                        values
                    )
                )
        })


descriptive_df = pd.DataFrame(
    descriptive_records
)


descriptive_file = (
    OUTPUT_DIR /
    "PBMMAE_4domain_channel_descriptive_statistics.csv"
)


descriptive_df.to_csv(
    descriptive_file,
    index=False
)


# ============================================================
# 10. GLOBAL ROBUST REFERENCE VALUES
#
# Robust z:
#
#     (domain median - global median) / global IQR
#
# This gives a dimensionless physical signature.
# ============================================================

global_records = []

robust_reference = {}


for channel in CHANNELS:

    values = (
        merged[
            channel
        ]
        .dropna()
        .to_numpy(
            dtype=float
        )
    )


    q1 = float(
        np.percentile(
            values,
            25
        )
    )

    med = float(
        np.median(
            values
        )
    )

    q3 = float(
        np.percentile(
            values,
            75
        )
    )

    iqr = (
        q3
        -
        q1
    )


    robust_reference[
        channel
    ] = {
        "median":
            med,
        "iqr":
            iqr
    }


    global_records.append({

        "Channel":
            channel,

        "Global_Q1":
            q1,

        "Global_Median":
            med,

        "Global_Q3":
            q3,

        "Global_IQR":
            iqr
    })


global_reference_df = pd.DataFrame(
    global_records
)


global_reference_file = (
    OUTPUT_DIR /
    "PBMMAE_4domain_global_robust_reference.csv"
)


global_reference_df.to_csv(
    global_reference_file,
    index=False
)


# ============================================================
# 11. ROBUST STANDARDIZED DOMAIN MEDIANS
# ============================================================

signature_records = []


for domain in DOMAINS:

    subset = merged[
        merged[
            "Consensus_domain_4"
        ] == domain
    ]


    for channel in CHANNELS:

        domain_median = float(
            subset[
                channel
            ].median()
        )


        global_median = (
            robust_reference[
                channel
            ][
                "median"
            ]
        )


        global_iqr = (
            robust_reference[
                channel
            ][
                "iqr"
            ]
        )


        if (
            np.isfinite(
                global_iqr
            )
            and
            global_iqr > 0
        ):

            robust_z = (
                domain_median
                -
                global_median
            ) / global_iqr

        else:

            robust_z = np.nan


        signature_records.append({

            "Meta_domain":
                domain,

            "Channel":
                channel,

            "Domain_median":
                domain_median,

            "Global_median":
                global_median,

            "Global_IQR":
                global_iqr,

            "Robust_standardized_median":
                robust_z
        })


signature_df = pd.DataFrame(
    signature_records
)


signature_file = (
    OUTPUT_DIR /
    "PBMMAE_4domain_robust_standardized_signatures.csv"
)


signature_df.to_csv(
    signature_file,
    index=False
)


# ============================================================
# 12. WIDE SIGNATURE MATRIX
# ============================================================

signature_matrix = (
    signature_df
    .pivot(
        index="Meta_domain",
        columns="Channel",
        values="Robust_standardized_median"
    )
    .reindex(
        columns=CHANNELS
    )
)


signature_matrix_file = (
    OUTPUT_DIR /
    "PBMMAE_4domain_robust_signature_matrix.csv"
)


signature_matrix.to_csv(
    signature_matrix_file
)


# ============================================================
# 13. KRUSKAL-WALLIS + EPSILON-SQUARED
#
# epsilon^2 = (H - k + 1) / (N - k)
#
# Spatial autocorrelation means p-values must not be
# interpreted as strict iid confirmatory inference.
# ============================================================

kw_records = []


for channel in CHANNELS:

    groups = []


    for domain in DOMAINS:

        values = (
            merged.loc[
                merged[
                    "Consensus_domain_4"
                ] == domain,
                channel
            ]
            .dropna()
            .to_numpy(
                dtype=float
            )
        )


        groups.append(
            values
        )


    if any(
        len(
            group
        ) == 0
        for group in groups
    ):

        continue


    H, p = kruskal(
        *groups
    )


    N_total = sum(
        len(
            group
        )
        for group in groups
    )


    k_groups = len(
        groups
    )


    if N_total > k_groups:

        epsilon_squared = (
            H
            -
            k_groups
            +
            1
        ) / (
            N_total
            -
            k_groups
        )

        epsilon_squared = float(
            np.clip(
                epsilon_squared,
                0.0,
                1.0
            )
        )

    else:

        epsilon_squared = np.nan


    kw_records.append({

        "Channel":
            channel,

        "N_total":
            N_total,

        "Kruskal_Wallis_H":
            float(
                H
            ),

        "p_value_exploratory":
            float(
                p
            ),

        "Epsilon_squared":
            epsilon_squared
    })


kw_df = pd.DataFrame(
    kw_records
)


kw_df = kw_df.sort_values(

    "Epsilon_squared",

    ascending=False
).reset_index(
    drop=True
)


kw_file = (
    OUTPUT_DIR /
    "PBMMAE_4domain_channel_separation_kruskal_effectsize.csv"
)


kw_df.to_csv(
    kw_file,
    index=False
)


# ============================================================
# 14. CLIFF'S DELTA
#
# Efficient exact calculation using sorted values.
#
# delta:
#   +1 = all A > B
#   -1 = all A < B
#    0 = strong overlap
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


    x = x[
        np.isfinite(
            x
        )
    ]

    y = y[
        np.isfinite(
            y
        )
    ]


    if (
        len(
            x
        ) == 0
        or
        len(
            y
        ) == 0
    ):

        return np.nan


    y_sorted = np.sort(
        y
    )


    greater = 0
    less = 0


    for value in x:

        less_than_value = np.searchsorted(
            y_sorted,
            value,
            side="left"
        )


        less_or_equal = np.searchsorted(
            y_sorted,
            value,
            side="right"
        )


        greater += less_than_value

        less += (
            len(
                y_sorted
            )
            -
            less_or_equal
        )


    delta = (
        greater
        -
        less
    ) / (
        len(
            x
        )
        *
        len(
            y
        )
    )


    return float(
        delta
    )


def cliff_magnitude(
    delta
):

    absolute = abs(
        delta
    )


    if absolute < 0.147:

        return "negligible"

    elif absolute < 0.330:

        return "small"

    elif absolute < 0.474:

        return "medium"

    else:

        return "large"


pairwise_records = []


for channel in CHANNELS:

    for domain_a, domain_b in combinations(
        DOMAINS,
        2
    ):

        a = merged.loc[
            merged[
                "Consensus_domain_4"
            ] == domain_a,
            channel
        ].dropna().to_numpy(
            dtype=float
        )


        b = merged.loc[
            merged[
                "Consensus_domain_4"
            ] == domain_b,
            channel
        ].dropna().to_numpy(
            dtype=float
        )


        delta = cliffs_delta(
            a,
            b
        )


        pairwise_records.append({

            "Channel":
                channel,

            "Domain_A":
                domain_a,

            "Domain_B":
                domain_b,

            "N_A":
                len(
                    a
                ),

            "N_B":
                len(
                    b
                ),

            "Cliffs_delta_A_vs_B":
                delta,

            "Absolute_delta":
                abs(
                    delta
                ),

            "Magnitude":
                cliff_magnitude(
                    delta
                )
        })


pairwise_df = pd.DataFrame(
    pairwise_records
)


pairwise_df = pairwise_df.sort_values(

    [
        "Channel",
        "Absolute_delta"
    ],

    ascending=[
        True,
        False
    ]
)


pairwise_file = (
    OUTPUT_DIR /
    "PBMMAE_4domain_pairwise_cliffs_delta.csv"
)


pairwise_df.to_csv(
    pairwise_file,
    index=False
)


# ============================================================
# 15. PHYSICS-FAMILY SIGNATURES
#
# Equal channel contribution within each family using the
# robust standardized domain medians.
# ============================================================

family_records = []


for domain in DOMAINS:

    for family, family_channels in (
        PHYSICS_FAMILIES.items()
    ):

        values = []


        for channel in family_channels:

            row = signature_df[
                (
                    signature_df[
                        "Meta_domain"
                    ] == domain
                )
                &
                (
                    signature_df[
                        "Channel"
                    ] == channel
                )
            ]


            if len(
                row
            ) == 1:

                values.append(
                    float(
                        row.iloc[
                            0
                        ][
                            "Robust_standardized_median"
                        ]
                    )
                )


        values = np.asarray(
            values,
            dtype=float
        )


        family_records.append({

            "Meta_domain":
                domain,

            "Physics_family":
                family,

            "N_channels":
                len(
                    family_channels
                ),

            "Mean_robust_signature":
                float(
                    np.nanmean(
                        values
                    )
                ),

            "Mean_absolute_robust_signature":
                float(
                    np.nanmean(
                        np.abs(
                            values
                        )
                    )
                )
        })


family_df = pd.DataFrame(
    family_records
)


family_file = (
    OUTPUT_DIR /
    "PBMMAE_4domain_physics_family_signatures.csv"
)


family_df.to_csv(
    family_file,
    index=False
)


# ============================================================
# 16. SAVE MERGED CELL-LEVEL PHYSICAL TABLE
# ============================================================

merged_file = (
    OUTPUT_DIR /
    "PBMMAE_4domain_cells_with_14_physical_channels.csv"
)


merged.to_csv(
    merged_file,
    index=False
)


# ============================================================
# 17. DOMAIN SAMPLE COUNTS
# ============================================================

domain_counts = (

    merged[
        "Consensus_domain_4"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# 18. PRINT MAIN RESULTS
# ============================================================

print(
    "\n" +
    "=" * 100
)

print(
    "DOMAIN SAMPLE COUNTS"
)

print(
    "=" * 100
)


for domain in DOMAINS:

    print(
        f"M{domain}:",
        int(
            domain_counts.get(
                domain,
                0
            )
        )
    )


print(
    "\n" +
    "=" * 110
)

print(
    "CHANNEL SEPARATION — RANKED BY EPSILON-SQUARED"
)

print(
    "=" * 110
)


print(

    kw_df[
        [
            "Channel",
            "Kruskal_Wallis_H",
            "p_value_exploratory",
            "Epsilon_squared"
        ]
    ]
    .round(
        5
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
    "ROBUST STANDARDIZED DOMAIN SIGNATURE MATRIX"
)

print(
    "=" * 120
)


print(
    signature_matrix
    .round(
        3
    )
    .to_string()
)


print(
    "\n" +
    "=" * 100
)

print(
    "PHYSICS-FAMILY SIGNATURES"
)

print(
    "=" * 100
)


print(

    family_df
    .pivot(
        index="Meta_domain",
        columns="Physics_family",
        values="Mean_robust_signature"
    )
    .round(
        3
    )
    .to_string()
)


print(
    "\n" +
    "=" * 100
)

print(
    "FILES SAVED"
)

print(
    "=" * 100
)


for path in [

    descriptive_file,
    signature_file,
    signature_matrix_file,
    kw_file,
    pairwise_file,
    family_file,
    merged_file,
    missing_file,
    global_reference_file
]:

    print(
        path
    )


print(
    "\nIMPORTANT:"
)

print(
    "Kruskal-Wallis p-values are exploratory because neighboring "
    "4-km cells are spatially autocorrelated."
)

print(
    "Interpretation should emphasize robust domain signatures, "
    "effect sizes, and subsequent spatial/geological validation."
)


print(
    "\n" +
    "=" * 100
)

print(
    "PHYSICAL CHARACTERIZATION COMPLETED"
)

print(
    "=" * 100
)