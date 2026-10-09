import os
# ============================================================
# Paper 1 — Build final 4-domain PB-MMAE consensus maps
#
# Inputs:
#   1. k=9 aligned labels across outer folds
#   2. fine-to-meta mapping for N_meta_domains = 4
#   3. locked 4-km master raster
#
# Outputs:
#   - 4-domain consensus GeoTIFF
#   - model-support-count GeoTIFF
#   - agreement-fraction GeoTIFF
#   - ambiguity/tie GeoTIFF
#   - cell-level CSV
#
# Important:
#   Domain numbering is categorical only.
#   M0, M1, M2, M3 have no ordinal meaning.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(os.environ["PBMMAE_PROJECT_ROOT"]).expanduser().resolve()

# Corrected master-grid location
MASTER_GRID = (
    PROJECT_ROOT /
    "02_data" /
    "masks" /
    "Paper1_MASTER_4km.tif"
)

ALIGNMENT_DIR = (
    PROJECT_ROOT /
    "03_outputs" /
    "models" /
    "PBMMAE" /
    "OuterFolds_13x13_LinearFusion64" /
    "crossfold_alignment_k9" /
    "hungarian_alignment"
)

METADOMAIN_DIR = (
    ALIGNMENT_DIR /
    "metadomain_analysis"
)

ALIGNED_LONG_FILE = (
    ALIGNMENT_DIR /
    "PBMMAE_k9_aligned_labels_long.csv"
)

MAPPING_FILE = (
    METADOMAIN_DIR /
    "PBMMAE_k9_fine_to_metadomain_mappings.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    "03_outputs" /
    "models" /
    "PBMMAE" /
    "final_4domain_consensus"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

N_FINE = 9
N_META = 4

DOMAIN_NODATA = -9999
COUNT_NODATA = 0
AGREEMENT_NODATA = -9999.0
AMBIGUITY_NODATA = 255

print("=" * 96)
print("PB-MMAE FINAL 4-DOMAIN CONSENSUS MAP GENERATION")
print("=" * 96)


# ============================================================
# 3. CHECK INPUT FILES
# ============================================================

for path in [
    MASTER_GRID,
    ALIGNED_LONG_FILE,
    MAPPING_FILE
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
# 4. LOAD ALIGNED k=9 LABELS
# ============================================================

aligned = pd.read_csv(
    ALIGNED_LONG_FILE
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
    "Aligned_cluster_k9"
}


if not required_columns.issubset(
    aligned.columns
):

    missing = (
        required_columns
        -
        set(
            aligned.columns
        )
    )

    raise RuntimeError(
        f"Missing aligned-label columns: {missing}"
    )


assert aligned[
    "Aligned_cluster_k9"
].between(
    0,
    N_FINE - 1
).all()


print(
    "\nAligned label rows:",
    len(
        aligned
    )
)

print(
    "Unique Cell_ID:",
    aligned[
        "Cell_ID"
    ].nunique()
)


# ============================================================
# 5. LOAD 4-DOMAIN MAPPING
# ============================================================

mapping_df = pd.read_csv(
    MAPPING_FILE
)


mapping4 = (
    mapping_df[
        mapping_df[
            "N_meta_domains"
        ] == N_META
    ]
    .copy()
)


if len(
    mapping4
) != N_FINE:

    raise RuntimeError(
        "Expected exactly 9 fine-cluster mappings "
        "for the 4-meta-domain solution."
    )


fine_to_meta = {

    int(
        row.Fine_cluster_k9
    ):
        int(
            row.Meta_domain
        )

    for row in mapping4.itertuples(
        index=False
    )
}


print(
    "\nFine k=9 -> 4-domain mapping:"
)

print(
    fine_to_meta
)


assert set(
    fine_to_meta.keys()
) == set(
    range(
        N_FINE
    )
)


# ============================================================
# 6. APPLY META-DOMAIN MAPPING
# ============================================================

aligned[
    "Meta_domain_4"
] = (

    aligned[
        "Aligned_cluster_k9"
    ]
    .map(
        fine_to_meta
    )
    .astype(
        int
    )
)


assert aligned[
    "Meta_domain_4"
].between(
    0,
    N_META - 1
).all()


# ============================================================
# 7. COMPUTE CELL-LEVEL CONSENSUS
# ============================================================

cell_records = []


for cell_id, group in aligned.groupby(
    "Cell_ID"
):

    labels = (
        group[
            "Meta_domain_4"
        ]
        .astype(
            int
        )
        .to_numpy()
    )

    n_models = len(
        labels
    )

    counts = np.bincount(
        labels,
        minlength=N_META
    )

    max_count = int(
        counts.max()
    )

    winners = np.where(
        counts == max_count
    )[0]

    tied = (
        len(
            winners
        ) > 1
    )

    if tied:
        consensus_domain = -1
    else:
        consensus_domain = int(
            winners[0]
        )

    agreement_fraction = float(
        max_count /
        n_models
    )

    first = group.iloc[0]

    record = {

        "Cell_ID":
            cell_id,

        "Row":
            int(
                first[
                    "Row"
                ]
            ),

        "Col":
            int(
                first[
                    "Col"
                ]
            ),

        "X":
            float(
                first[
                    "X"
                ]
            ),

        "Y":
            float(
                first[
                    "Y"
                ]
            ),

        "Spatial_Fold":
            int(
                first[
                    "Spatial_Fold"
                ]
            ),

        "N_models":
            int(
                n_models
            ),

        "Consensus_domain_4":
            int(
                consensus_domain
            ),

        "Consensus_tie":
            bool(
                tied
            ),

        "Agreement_count":
            int(
                max_count
            ),

        "Agreement_fraction":
            agreement_fraction
    }

    for domain_id in range(
        N_META
    ):

        record[
            f"M{domain_id}_votes"
        ] = int(
            counts[
                domain_id
            ]
        )

    cell_records.append(
        record
    )


cell_df = pd.DataFrame(
    cell_records
)


print(
    "\nConsensus cells:",
    len(
        cell_df
    )
)

print(
    "Unique-domain cells:",
    int(
        (
            ~cell_df[
                "Consensus_tie"
            ]
        ).sum()
    )
)

print(
    "Tied cells:",
    int(
        cell_df[
            "Consensus_tie"
        ].sum()
    )
)

print(
    "Mean agreement fraction:",
    round(
        cell_df[
            "Agreement_fraction"
        ].mean(),
        4
    )
)

print(
    "Median agreement fraction:",
    round(
        cell_df[
            "Agreement_fraction"
        ].median(),
        4
    )
)


# ============================================================
# 8. READ MASTER RASTER GEOMETRY
# ============================================================

with rasterio.open(
    MASTER_GRID
) as src:

    master_profile = (
        src.profile.copy()
    )

    master_transform = (
        src.transform
    )

    master_crs = (
        src.crs
    )

    master_height = (
        src.height
    )

    master_width = (
        src.width
    )


print(
    "\nMaster raster:",
    master_height,
    "rows x",
    master_width,
    "cols"
)

print(
    "CRS:",
    master_crs
)

print(
    "Transform:",
    master_transform
)


# ============================================================
# 9. CHECK ROW / COL BOUNDS
# ============================================================

if (
    cell_df[
        "Row"
    ].min() < 0
    or
    cell_df[
        "Row"
    ].max() >= master_height
):

    raise RuntimeError(
        "Row index exceeds master raster bounds."
    )


if (
    cell_df[
        "Col"
    ].min() < 0
    or
    cell_df[
        "Col"
    ].max() >= master_width
):

    raise RuntimeError(
        "Column index exceeds master raster bounds."
    )


# ============================================================
# 10. VERIFY STORED X/Y AGAINST MASTER GRID CENTERS
# ============================================================

computed_x = []
computed_y = []


for row, col in zip(
    cell_df[
        "Row"
    ],
    cell_df[
        "Col"
    ]
):

    x, y = rasterio.transform.xy(
        master_transform,
        int(
            row
        ),
        int(
            col
        ),
        offset="center"
    )

    computed_x.append(
        x
    )

    computed_y.append(
        y
    )


computed_x = np.asarray(
    computed_x
)

computed_y = np.asarray(
    computed_y
)


x_match = np.allclose(
    computed_x,
    cell_df[
        "X"
    ].to_numpy(),
    atol=1e-3
)

y_match = np.allclose(
    computed_y,
    cell_df[
        "Y"
    ].to_numpy(),
    atol=1e-3
)


print(
    "\nMaster-grid coordinate check:"
)

print(
    "X match:",
    x_match
)

print(
    "Y match:",
    y_match
)


if not (
    x_match
    and
    y_match
):

    raise RuntimeError(
        "Patch metadata coordinates do not match "
        "the locked master-grid cell centers."
    )


# ============================================================
# 11. INITIALIZE OUTPUT RASTERS
# ============================================================

domain_raster = np.full(
    (
        master_height,
        master_width
    ),
    DOMAIN_NODATA,
    dtype=np.int16
)

support_raster = np.full(
    (
        master_height,
        master_width
    ),
    COUNT_NODATA,
    dtype=np.uint8
)

agreement_raster = np.full(
    (
        master_height,
        master_width
    ),
    AGREEMENT_NODATA,
    dtype=np.float32
)

ambiguity_raster = np.full(
    (
        master_height,
        master_width
    ),
    AMBIGUITY_NODATA,
    dtype=np.uint8
)


# ============================================================
# 12. POPULATE RASTERS
# ============================================================

for row in cell_df.itertuples(
    index=False
):

    r = int(
        row.Row
    )

    c = int(
        row.Col
    )

    domain_raster[
        r,
        c
    ] = int(
        row.Consensus_domain_4
    )

    support_raster[
        r,
        c
    ] = int(
        row.N_models
    )

    agreement_raster[
        r,
        c
    ] = float(
        row.Agreement_fraction
    )

    ambiguity_raster[
        r,
        c
    ] = (
        1
        if row.Consensus_tie
        else 0
    )


# ============================================================
# 13. SAVE CELL-LEVEL CONSENSUS CSV
# ============================================================

cell_csv = (
    OUTPUT_DIR /
    "PBMMAE_4domain_cell_consensus.csv"
)

cell_df.to_csv(
    cell_csv,
    index=False
)


# ============================================================
# 14. SAVE DOMAIN RASTER
# ============================================================

domain_profile = (
    master_profile.copy()
)

domain_profile.update(
    dtype="int16",
    count=1,
    nodata=DOMAIN_NODATA,
    compress="lzw"
)

domain_file = (
    OUTPUT_DIR /
    "PBMMAE_4domain_consensus_4km.tif"
)

with rasterio.open(
    domain_file,
    "w",
    **domain_profile
) as dst:

    dst.write(
        domain_raster,
        1
    )


# ============================================================
# 15. SAVE MODEL-SUPPORT RASTER
# ============================================================

support_profile = (
    master_profile.copy()
)

support_profile.update(
    dtype="uint8",
    count=1,
    nodata=COUNT_NODATA,
    compress="lzw"
)

support_file = (
    OUTPUT_DIR /
    "PBMMAE_4domain_model_support_count_4km.tif"
)

with rasterio.open(
    support_file,
    "w",
    **support_profile
) as dst:

    dst.write(
        support_raster,
        1
    )


# ============================================================
# 16. SAVE AGREEMENT FRACTION RASTER
# ============================================================

agreement_profile = (
    master_profile.copy()
)

agreement_profile.update(
    dtype="float32",
    count=1,
    nodata=AGREEMENT_NODATA,
    compress="lzw"
)

agreement_file = (
    OUTPUT_DIR /
    "PBMMAE_4domain_agreement_fraction_4km.tif"
)

with rasterio.open(
    agreement_file,
    "w",
    **agreement_profile
) as dst:

    dst.write(
        agreement_raster,
        1
    )


# ============================================================
# 17. SAVE AMBIGUITY RASTER
# ============================================================

ambiguity_profile = (
    master_profile.copy()
)

ambiguity_profile.update(
    dtype="uint8",
    count=1,
    nodata=AMBIGUITY_NODATA,
    compress="lzw"
)

ambiguity_file = (
    OUTPUT_DIR /
    "PBMMAE_4domain_ambiguity_4km.tif"
)

with rasterio.open(
    ambiguity_file,
    "w",
    **ambiguity_profile
) as dst:

    dst.write(
        ambiguity_raster,
        1
    )


# ============================================================
# 18. DOMAIN OCCUPANCY SUMMARY
# ============================================================

unique_cells = cell_df[
    ~cell_df[
        "Consensus_tie"
    ]
].copy()


domain_summary_records = []


for domain_id in range(
    N_META
):

    subset = unique_cells[
        unique_cells[
            "Consensus_domain_4"
        ] == domain_id
    ]

    domain_summary_records.append({

        "Meta_domain":
            domain_id,

        "N_cells":
            len(
                subset
            ),

        "Fraction_of_unique_consensus_cells":
            (
                len(
                    subset
                )
                /
                len(
                    unique_cells
                )
                if len(
                    unique_cells
                ) > 0
                else np.nan
            ),

        "Mean_agreement_fraction":
            (
                subset[
                    "Agreement_fraction"
                ].mean()
                if len(
                    subset
                ) > 0
                else np.nan
            ),

        "Median_agreement_fraction":
            (
                subset[
                    "Agreement_fraction"
                ].median()
                if len(
                    subset
                ) > 0
                else np.nan
            ),

        "Mean_model_support":
            (
                subset[
                    "N_models"
                ].mean()
                if len(
                    subset
                ) > 0
                else np.nan
            )
    })


domain_summary_df = pd.DataFrame(
    domain_summary_records
)

domain_summary_file = (
    OUTPUT_DIR /
    "PBMMAE_4domain_occupancy_summary.csv"
)

domain_summary_df.to_csv(
    domain_summary_file,
    index=False
)


# ============================================================
# 19. OVERALL MAP SUMMARY
# ============================================================

overall_summary = pd.DataFrame(
    [{
        "Total_consensus_cells":
            len(
                cell_df
            ),

        "Unique_consensus_cells":
            int(
                (
                    ~cell_df[
                        "Consensus_tie"
                    ]
                ).sum()
            ),

        "Tied_cells":
            int(
                cell_df[
                    "Consensus_tie"
                ].sum()
            ),

        "Tie_fraction":
            float(
                cell_df[
                    "Consensus_tie"
                ].mean()
            ),

        "Mean_agreement_fraction":
            float(
                cell_df[
                    "Agreement_fraction"
                ].mean()
            ),

        "Median_agreement_fraction":
            float(
                cell_df[
                    "Agreement_fraction"
                ].median()
            ),

        "Min_model_support":
            int(
                cell_df[
                    "N_models"
                ].min()
            ),

        "Max_model_support":
            int(
                cell_df[
                    "N_models"
                ].max()
            ),

        "Mean_model_support":
            float(
                cell_df[
                    "N_models"
                ].mean()
            )
    }]
)

overall_summary_file = (
    OUTPUT_DIR /
    "PBMMAE_4domain_map_summary.csv"
)

overall_summary.to_csv(
    overall_summary_file,
    index=False
)


# ============================================================
# 20. PRINT RESULTS
# ============================================================

print(
    "\n" +
    "=" * 100
)

print(
    "FINAL 4-DOMAIN OCCUPANCY"
)

print(
    "=" * 100
)

print(
    domain_summary_df
    .round(
        4
    )
    .to_string(
        index=False
    )
)

print(
    "\n" +
    "=" * 100
)

print(
    "MAP-WIDE AGREEMENT SUMMARY"
)

print(
    "=" * 100
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
    "\nDomain raster:"
)

print(
    domain_file
)

print(
    "\nAgreement raster:"
)

print(
    agreement_file
)

print(
    "\nModel-support raster:"
)

print(
    support_file
)

print(
    "\nAmbiguity raster:"
)

print(
    ambiguity_file
)

print(
    "\nCell-level CSV:"
)

print(
    cell_csv
)

print(
    "\nDomain occupancy summary:"
)

print(
    domain_summary_file
)

print(
    "\nOverall map summary:"
)

print(
    overall_summary_file
)

print(
    "=" * 100
)

print(
    "4-DOMAIN CONSENSUS MAP GENERATION COMPLETED"
)

print(
    "=" * 100
)