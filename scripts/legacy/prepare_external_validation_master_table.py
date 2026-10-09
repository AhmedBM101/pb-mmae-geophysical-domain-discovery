import os
# ============================================================
# Paper 1 — External Validation QC + Master Table
#
# WITHHELD validation datasets:
#   1. Categorical geology polygons
#   2. Warm-spring point locations
#   3. Continuous AHP-GPI raster
#
# Frozen model product:
#   PB-MMAE 4-domain consensus map
#
# IMPORTANT CONSENSUS CODING
# --------------------------
#   -1 = tied / unresolved cell
#    0 = M0
#    1 = M1
#    2 = M2
#    3 = M3
#
# Tied cells remain excluded from categorical validation.
# ============================================================

import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import xy
from shapely.geometry import Point


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(os.environ["PBMMAE_PROJECT_ROOT"]).expanduser().resolve()

MASTER_GRID = (
    PROJECT_ROOT /
    "02_data" /
    "masks" /
    "Paper1_MASTER_4km.tif"
)

DOMAIN_RASTER = (
    PROJECT_ROOT /
    "03_outputs" /
    "models" /
    "PBMMAE" /
    "final_4domain_consensus" /
    "PBMMAE_4domain_consensus_4km.tif"
)

AGREEMENT_RASTER = (
    PROJECT_ROOT /
    "03_outputs" /
    "models" /
    "PBMMAE" /
    "final_4domain_consensus" /
    "PBMMAE_4domain_agreement_fraction_4km.tif"
)

SUPPORT_RASTER = (
    PROJECT_ROOT /
    "03_outputs" /
    "models" /
    "PBMMAE" /
    "final_4domain_consensus" /
    "PBMMAE_4domain_model_support_count_4km.tif"
)

AMBIGUITY_RASTER = (
    PROJECT_ROOT /
    "03_outputs" /
    "models" /
    "PBMMAE" /
    "final_4domain_consensus" /
    "PBMMAE_4domain_ambiguity_4km.tif"
)


# ------------------------------------------------------------
# WITHHELD EXTERNAL VALIDATION DATA
# ------------------------------------------------------------

GEOLOGY_FILE = (
    PROJECT_ROOT /
    "02_data" /
    "external_validation" /
    "geology" /
    "MBT_UBT_Geology_Categorical.shp"
)

WARM_SPRING_FILE = (
    PROJECT_ROOT /
    "02_data" /
    "external_validation" /
    "warm_springs" /
    "Warm_spring_XY.shp"
)

GPI_FILE = (
    PROJECT_ROOT /
    "02_data" /
    "external_validation" /
    "gpi" /
    "GPI_4Km_F.tif"
)


OUTPUT_ROOT = (
    PROJECT_ROOT /
    "03_outputs" /
    "external_validation"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. REQUIRED FILE CHECK
# ============================================================

required_files = {

    "Master grid":
        MASTER_GRID,

    "4-domain raster":
        DOMAIN_RASTER,

    "Agreement raster":
        AGREEMENT_RASTER,

    "Support raster":
        SUPPORT_RASTER,

    "Ambiguity raster":
        AMBIGUITY_RASTER,

    "Geology":
        GEOLOGY_FILE,

    "Warm springs":
        WARM_SPRING_FILE,

    "GPI":
        GPI_FILE
}


print(
    "=" * 110
)

print(
    "PAPER 1 — EXTERNAL VALIDATION PREPARATION"
)

print(
    "=" * 110
)


for label, path in required_files.items():

    if not path.exists():

        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )

    print(
        f"{label:<22} OK"
    )


# ============================================================
# 3. RASTER METADATA
# ============================================================

def raster_metadata(path):

    with rasterio.open(path) as src:

        return {

            "File":
                path.name,

            "CRS":
                str(src.crs),

            "Width":
                src.width,

            "Height":
                src.height,

            "Cell_X":
                src.transform.a,

            "Cell_Y":
                abs(src.transform.e),

            "Left":
                src.bounds.left,

            "Right":
                src.bounds.right,

            "Bottom":
                src.bounds.bottom,

            "Top":
                src.bounds.top,

            "Bands":
                src.count,

            "Dtype":
                src.dtypes[0],

            "NoData":
                src.nodata
        }


# ============================================================
# 4. RASTER GRID QC
# ============================================================

raster_paths = [

    MASTER_GRID,
    DOMAIN_RASTER,
    AGREEMENT_RASTER,
    SUPPORT_RASTER,
    AMBIGUITY_RASTER,
    GPI_FILE
]


raster_qc = pd.DataFrame(
    [
        raster_metadata(path)
        for path in raster_paths
    ]
)


raster_qc_file = (
    OUTPUT_ROOT /
    "Paper1_external_validation_raster_grid_QC.csv"
)


raster_qc.to_csv(
    raster_qc_file,
    index=False
)


print(
    "\n" +
    "=" * 110
)

print(
    "RASTER GRID QC"
)

print(
    "=" * 110
)

print(
    raster_qc.to_string(
        index=False
    )
)


# ============================================================
# 5. STRICT GRID MATCH
# ============================================================

with rasterio.open(
    MASTER_GRID
) as master:

    MASTER_CRS = master.crs
    MASTER_TRANSFORM = master.transform
    MASTER_WIDTH = master.width
    MASTER_HEIGHT = master.height
    MASTER_BOUNDS = master.bounds


def verify_same_grid(
    path,
    label
):

    with rasterio.open(path) as src:

        checks = {

            "CRS_match":
                src.crs == MASTER_CRS,

            "Width_match":
                src.width == MASTER_WIDTH,

            "Height_match":
                src.height == MASTER_HEIGHT,

            "Transform_match":
                src.transform.almost_equals(
                    MASTER_TRANSFORM
                ),

            "Bounds_match":
                np.allclose(
                    [
                        src.bounds.left,
                        src.bounds.bottom,
                        src.bounds.right,
                        src.bounds.top
                    ],
                    [
                        MASTER_BOUNDS.left,
                        MASTER_BOUNDS.bottom,
                        MASTER_BOUNDS.right,
                        MASTER_BOUNDS.top
                    ],
                    atol=1e-6,
                    rtol=0
                )
        }

        return {

            "Dataset":
                label,

            **checks,

            "ALL_GRID_CHECKS":
                all(checks.values())
        }


grid_check_records = []


for label, path in [

    (
        "PBMMAE_4domain",
        DOMAIN_RASTER
    ),

    (
        "PBMMAE_agreement",
        AGREEMENT_RASTER
    ),

    (
        "PBMMAE_support",
        SUPPORT_RASTER
    ),

    (
        "PBMMAE_ambiguity",
        AMBIGUITY_RASTER
    ),

    (
        "Withheld_GPI",
        GPI_FILE
    )
]:

    grid_check_records.append(
        verify_same_grid(
            path,
            label
        )
    )


grid_check_df = pd.DataFrame(
    grid_check_records
)


grid_check_file = (
    OUTPUT_ROOT /
    "Paper1_external_validation_grid_match_checks.csv"
)


grid_check_df.to_csv(
    grid_check_file,
    index=False
)


print(
    "\nGrid-match audit:"
)

print(
    grid_check_df.to_string(
        index=False
    )
)


if not grid_check_df[
    "ALL_GRID_CHECKS"
].all():

    raise RuntimeError(
        "At least one raster does not match "
        "the immutable Paper 1 master grid."
    )


print(
    "\nALL raster grids match the master grid."
)


# ============================================================
# 6. VECTOR QC
# ============================================================

geology = gpd.read_file(
    GEOLOGY_FILE
)

springs = gpd.read_file(
    WARM_SPRING_FILE
)


print(
    "\n" +
    "=" * 110
)

print(
    "VECTOR QC"
)

print(
    "=" * 110
)


print(
    "\nGeology:"
)

print(
    "  Features:",
    len(geology)
)

print(
    "  CRS:",
    geology.crs
)

print(
    "  Geometry types:",
    geology.geometry.geom_type.value_counts().to_dict()
)


print(
    "\nWarm springs:"
)

print(
    "  Features:",
    len(springs)
)

print(
    "  CRS:",
    springs.crs
)

print(
    "  Geometry types:",
    springs.geometry.geom_type.value_counts().to_dict()
)


if geology.crs is None:

    raise RuntimeError(
        "Geology layer has no CRS."
    )


if springs.crs is None:

    raise RuntimeError(
        "Warm-spring layer has no CRS."
    )


if geology.crs != MASTER_CRS:

    print(
        "\nReprojecting geology in memory to:",
        MASTER_CRS
    )

    geology = geology.to_crs(
        MASTER_CRS
    )


if springs.crs != MASTER_CRS:

    print(
        "\nReprojecting warm springs in memory to:",
        MASTER_CRS
    )

    springs = springs.to_crs(
        MASTER_CRS
    )


# ============================================================
# 7. GEOLOGY ATTRIBUTE AUDIT
# ============================================================

required_geology_fields = [

    "class_id",
    "unit_code",
    "unit_name",
    "age",
    "lithology"
]


missing_geology_fields = [

    field

    for field
    in required_geology_fields

    if field not in geology.columns
]


if missing_geology_fields:

    raise RuntimeError(
        "Required geology fields absent: "
        +
        ", ".join(
            missing_geology_fields
        )
    )


geology_attribute_summary = (

    geology[
        required_geology_fields
    ]

    .drop_duplicates()

    .sort_values(
        [
            "unit_name",
            "unit_code"
        ]
    )

    .reset_index(
        drop=True
    )
)


geology_attribute_file = (
    OUTPUT_ROOT /
    "Paper1_geology_unique_attribute_combinations.csv"
)


geology_attribute_summary.to_csv(
    geology_attribute_file,
    index=False
)


print(
    "\nUnique geology units:",
    geology[
        "unit_name"
    ].nunique(
        dropna=True
    )
)


print(
    "Unique geology codes:",
    geology[
        "unit_code"
    ].nunique(
        dropna=True
    )
)


# ============================================================
# 8. READ RASTERS
# ============================================================

def read_single_band(path):

    with rasterio.open(path) as src:

        array = src.read(1)

        nodata = src.nodata

        if nodata is not None:

            if np.issubdtype(
                array.dtype,
                np.floating
            ):

                valid = (
                    np.isfinite(array)
                    &
                    ~np.isclose(
                        array,
                        nodata
                    )
                )

            else:

                valid = (
                    array != nodata
                )

        else:

            if np.issubdtype(
                array.dtype,
                np.floating
            ):

                valid = np.isfinite(
                    array
                )

            else:

                valid = np.ones(
                    array.shape,
                    dtype=bool
                )

        return (
            array,
            valid,
            src.transform
        )


domain_array, domain_valid, transform = (
    read_single_band(
        DOMAIN_RASTER
    )
)

agreement_array, agreement_valid, _ = (
    read_single_band(
        AGREEMENT_RASTER
    )
)

support_array, support_valid, _ = (
    read_single_band(
        SUPPORT_RASTER
    )
)

ambiguity_array, ambiguity_valid, _ = (
    read_single_band(
        AMBIGUITY_RASTER
    )
)


# ============================================================
# 9. CONSENSUS CODING AUDIT
#
# IMPORTANT:
#
#   -1 = tied / unresolved
#    0 = M0
#    1 = M1
#    2 = M2
#    3 = M3
#
# Only 0,1,2,3 enter categorical external validation.
# ============================================================

resolved_domain_values = [
    0,
    1,
    2,
    3
]


mapped_mask = (

    domain_valid
    &
    np.isin(
        domain_array,
        resolved_domain_values
    )
)


tied_mask = (

    domain_valid
    &
    (
        domain_array == -1
    )
)


n_resolved_cells = int(
    mapped_mask.sum()
)

n_tied_cells = int(
    tied_mask.sum()
)

n_consensus_centers = (
    n_resolved_cells
    +
    n_tied_cells
)


print(
    "\n" +
    "=" * 110
)

print(
    "CONSENSUS RASTER AUDIT"
)

print(
    "=" * 110
)


print(
    "Resolved domain cells:",
    n_resolved_cells
)

print(
    "Tied / unresolved cells (-1):",
    n_tied_cells
)

print(
    "Total consensus patch centers:",
    n_consensus_centers
)


# Expected values from frozen consensus solution.

if n_consensus_centers != 1477:

    print(
        "\nWARNING:"
    )

    print(
        "Expected 1477 consensus patch centers, "
        f"but found {n_consensus_centers}."
    )


if n_resolved_cells != 1382:

    print(
        "\nWARNING:"
    )

    print(
        "Expected 1382 unique-consensus cells, "
        f"but found {n_resolved_cells}."
    )


if n_tied_cells != 95:

    print(
        "\nWARNING:"
    )

    print(
        "Expected 95 tied cells, "
        f"but found {n_tied_cells}."
    )


# ============================================================
# 10. CREATE RESOLVED DOMAIN-CELL TABLE
# ============================================================

rows, cols = np.where(
    mapped_mask
)


x_coords, y_coords = xy(
    transform,
    rows,
    cols,
    offset="center"
)


domain_values = domain_array[
    rows,
    cols
].astype(
    int
)


agreement_values = agreement_array[
    rows,
    cols
]


support_values = support_array[
    rows,
    cols
]


ambiguity_values = ambiguity_array[
    rows,
    cols
]


domain_cells_df = pd.DataFrame({

    "Row":
        rows.astype(
            int
        ),

    "Col":
        cols.astype(
            int
        ),

    "X":
        np.asarray(
            x_coords,
            dtype=float
        ),

    "Y":
        np.asarray(
            y_coords,
            dtype=float
        ),

    "PBMMAE_domain_raw":
        domain_values,

    "PBMMAE_agreement":
        agreement_values,

    "PBMMAE_model_support":
        support_values,

    "PBMMAE_ambiguity":
        ambiguity_values
})


# ============================================================
# 11. DOMAIN LABEL AUDIT
# ============================================================

unique_domains = np.unique(
    domain_cells_df[
        "PBMMAE_domain_raw"
    ]
)


print(
    "\n" +
    "=" * 110
)

print(
    "FROZEN RESOLVED DOMAIN MAP"
)

print(
    "=" * 110
)


print(
    "Resolved mapped cells:",
    len(
        domain_cells_df
    )
)


print(
    "Raw unique domain values:",
    unique_domains
)


if not np.array_equal(
    np.sort(unique_domains),
    np.array(
        [0, 1, 2, 3]
    )
):

    raise RuntimeError(
        "Expected resolved domain values [0,1,2,3]; "
        f"found {unique_domains}"
    )


# Explicit mapping. Never infer or reorder these IDs.

raw_to_domain_name = {

    0:
        "M0",

    1:
        "M1",

    2:
        "M2",

    3:
        "M3"
}


domain_cells_df[
    "PBMMAE_domain"
] = domain_cells_df[
    "PBMMAE_domain_raw"
].map(
    raw_to_domain_name
)


print(
    "\nDomain mapping:"
)

for raw_value, domain_name in raw_to_domain_name.items():

    print(
        f"  {raw_value} -> {domain_name}"
    )


print(
    "\nResolved domain occupancy:"
)

print(
    domain_cells_df[
        "PBMMAE_domain"
    ].value_counts(
        sort=False
    )
)


# ============================================================
# 12. DOMAIN POINT GEODATAFRAME
# ============================================================

domain_geometry = [

    Point(
        x,
        y
    )

    for x, y
    in zip(
        domain_cells_df[
            "X"
        ],
        domain_cells_df[
            "Y"
        ]
    )
]


domain_gdf = gpd.GeoDataFrame(

    domain_cells_df.copy(),

    geometry=domain_geometry,

    crs=MASTER_CRS
)


domain_gdf[
    "Domain_Cell_ID"
] = np.arange(
    1,
    len(domain_gdf) + 1
)


# ============================================================
# 13. GEOLOGY ASSIGNMENT FUNCTION
#
# If a point lies exactly on two polygons, geological
# assignment is flagged as boundary-ambiguous.
# ============================================================

def assign_geology_to_points(
    point_gdf,
    geology_gdf,
    id_column
):

    geology_subset = geology_gdf[
        [
            "class_id",
            "unit_code",
            "unit_name",
            "age",
            "lithology",
            "geometry"
        ]
    ].copy()


    joined = gpd.sjoin(

        point_gdf,

        geology_subset,

        how="left",

        predicate="intersects"
    )


    # Count actual polygon matches.
    match_counts = (

        joined.groupby(
            id_column
        )[
            "index_right"
        ]
        .count()
    )


    ambiguous_ids = set(
        match_counts[
            match_counts > 1
        ].index
    )


    joined_unique = (

        joined

        .drop_duplicates(
            subset=[
                id_column
            ],
            keep="first"
        )

        .copy()
    )


    joined_unique[
        "Geology_match_count"
    ] = joined_unique[
        id_column
    ].map(
        match_counts
    ).fillna(
        0
    ).astype(
        int
    )


    joined_unique[
        "Geology_boundary_ambiguous"
    ] = joined_unique[
        id_column
    ].isin(
        ambiguous_ids
    )


    geology_fields = [

        "class_id",
        "unit_code",
        "unit_name",
        "age",
        "lithology"
    ]


    if ambiguous_ids:

        joined_unique.loc[
            joined_unique[
                "Geology_boundary_ambiguous"
            ],
            geology_fields
        ] = np.nan


    return joined_unique


# ============================================================
# 14. ASSIGN GEOLOGY TO DOMAIN CELLS
# ============================================================

domain_with_geology = assign_geology_to_points(

    point_gdf=domain_gdf,

    geology_gdf=geology,

    id_column="Domain_Cell_ID"
)


print(
    "\n" +
    "=" * 110
)

print(
    "DOMAIN-CELL GEOLOGY ASSIGNMENT"
)

print(
    "=" * 110
)


print(
    "Resolved domain cells:",
    len(
        domain_with_geology
    )
)


print(
    "Geology assigned:",
    domain_with_geology[
        "unit_name"
    ].notna().sum()
)


print(
    "No geological match:",
    (
        domain_with_geology[
            "Geology_match_count"
        ] == 0
    ).sum()
)


print(
    "Boundary-ambiguous geology:",
    domain_with_geology[
        "Geology_boundary_ambiguous"
    ].sum()
)


# ============================================================
# 15. SAMPLE GPI AT DOMAIN CELL CENTERS
# ============================================================

with rasterio.open(
    GPI_FILE
) as gpi_src:

    gpi_nodata = gpi_src.nodata

    coordinates = list(
        zip(
            domain_with_geology.geometry.x,
            domain_with_geology.geometry.y
        )
    )

    sampled = np.asarray(
        [
            value[0]
            for value in gpi_src.sample(
                coordinates
            )
        ],
        dtype=float
    )


if gpi_nodata is not None:

    sampled[
        np.isclose(
            sampled,
            gpi_nodata
        )
    ] = np.nan


domain_with_geology[
    "GPI_continuous"
] = sampled


domain_with_geology[
    "GPI_nearest_class"
] = np.where(

    np.isfinite(
        sampled
    ),

    np.rint(
        sampled
    ),

    np.nan
)


print(
    "\nGPI sampled at resolved domain cells:"
)

print(
    "Valid GPI:",
    np.isfinite(
        sampled
    ).sum(),
    "/",
    len(sampled)
)


if np.isfinite(sampled).any():

    print(
        "GPI range:",
        float(
            np.nanmin(sampled)
        ),
        "to",
        float(
            np.nanmax(sampled)
        )
    )


# ============================================================
# 16. SAVE DOMAIN MASTER TABLE
# ============================================================

domain_output_columns = [

    "Domain_Cell_ID",

    "Row",
    "Col",
    "X",
    "Y",

    "PBMMAE_domain_raw",
    "PBMMAE_domain",

    "PBMMAE_agreement",
    "PBMMAE_model_support",
    "PBMMAE_ambiguity",

    "Geology_match_count",
    "Geology_boundary_ambiguous",

    "class_id",
    "unit_code",
    "unit_name",
    "age",
    "lithology",

    "GPI_continuous",
    "GPI_nearest_class"
]


domain_master_df = pd.DataFrame(
    domain_with_geology[
        domain_output_columns
    ]
)


domain_master_file = (
    OUTPUT_ROOT /
    "Paper1_PBMMAE_domain_external_validation_MASTER.csv"
)


domain_master_df.to_csv(
    domain_master_file,
    index=False
)


# ============================================================
# 17. SAVE DOMAIN POINTS AS GPKG
# ============================================================

domain_gpkg_file = (
    OUTPUT_ROOT /
    "Paper1_PBMMAE_domain_external_validation.gpkg"
)


if domain_gpkg_file.exists():

    domain_gpkg_file.unlink()


domain_with_geology.to_file(

    domain_gpkg_file,

    layer="domain_cells",

    driver="GPKG"
)


# ============================================================
# 18. WARM-SPRING NAME FIELD
# ============================================================

candidate_name_fields = [

    "Name",
    "NAME",
    "name",

    "Spring",
    "SPRING",
    "spring",

    "Site",
    "SITE",
    "site",

    "Location",
    "LOCATION",
    "location"
]


SPRING_NAME_FIELD = None


for field in candidate_name_fields:

    if field in springs.columns:

        SPRING_NAME_FIELD = field

        break


if SPRING_NAME_FIELD is None:

    object_fields = [

        column

        for column
        in springs.columns

        if (
            column != "geometry"
            and
            springs[
                column
            ].dtype == "object"
        )
    ]


    # Prefer the object field whose values resemble known names.

    expected_names = {
        "awe",
        "keana",
        "azara",
        "kanje",
        "ribi",
        "akiri"
    }


    for field in object_fields:

        field_values = set(
            springs[
                field
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            .tolist()
        )

        if len(
            field_values
            &
            expected_names
        ) >= 4:

            SPRING_NAME_FIELD = field
            break


if SPRING_NAME_FIELD is not None:

    springs[
        "Spring_name"
    ] = (
        springs[
            SPRING_NAME_FIELD
        ]
        .astype(str)
        .str.strip()
    )

else:

    springs[
        "Spring_name"
    ] = [
        f"Spring_{i+1}"
        for i in range(
            len(springs)
        )
    ]


print(
    "\n" +
    "=" * 110
)

print(
    "WARM-SPRING ATTRIBUTE AUDIT"
)

print(
    "=" * 110
)


print(
    "Detected name field:",
    SPRING_NAME_FIELD
)


print(
    "Spring names:",
    springs[
        "Spring_name"
    ].tolist()
)


# ============================================================
# 19. GENERIC RASTER POINT SAMPLING
# ============================================================

spring_coordinates = list(
    zip(
        springs.geometry.x,
        springs.geometry.y
    )
)


def sample_raster_at_points(
    path,
    coordinates
):

    with rasterio.open(path) as src:

        nodata = src.nodata

        values = np.asarray(
            [
                value[0]
                for value in src.sample(
                    coordinates
                )
            ],
            dtype=float
        )


        if nodata is not None:

            values[
                np.isclose(
                    values,
                    nodata
                )
            ] = np.nan


        return values


# ============================================================
# 20. SAMPLE PB-MMAE + GPI AT SPRINGS
# ============================================================

spring_domain_raw = sample_raster_at_points(
    DOMAIN_RASTER,
    spring_coordinates
)

spring_agreement = sample_raster_at_points(
    AGREEMENT_RASTER,
    spring_coordinates
)

spring_support = sample_raster_at_points(
    SUPPORT_RASTER,
    spring_coordinates
)

spring_ambiguity = sample_raster_at_points(
    AMBIGUITY_RASTER,
    spring_coordinates
)

spring_gpi = sample_raster_at_points(
    GPI_FILE,
    spring_coordinates
)


springs[
    "PBMMAE_domain_raw"
] = spring_domain_raw


# Explicit treatment of tied/unresolved spring locations.

springs[
    "PBMMAE_domain"
] = [

        raw_to_domain_name.get(
            int(value),
            np.nan
        )

        if (
            np.isfinite(value)
            and
            int(value) in raw_to_domain_name
        )

        else np.nan

        for value in spring_domain_raw
    ]


springs[
    "PBMMAE_domain_tied"
] = [

    bool(
        np.isfinite(value)
        and
        int(value) == -1
    )

    for value in spring_domain_raw
]


springs[
    "PBMMAE_agreement"
] = spring_agreement


springs[
    "PBMMAE_model_support"
] = spring_support


springs[
    "PBMMAE_ambiguity"
] = spring_ambiguity


springs[
    "GPI_continuous"
] = spring_gpi


springs[
    "GPI_nearest_class"
] = np.where(

    np.isfinite(
        spring_gpi
    ),

    np.rint(
        spring_gpi
    ),

    np.nan
)


# ============================================================
# 21. GEOLOGY AT SPRINGS
# ============================================================

springs[
    "Spring_ID"
] = np.arange(
    1,
    len(springs) + 1
)


springs_with_geology = assign_geology_to_points(

    point_gdf=springs,

    geology_gdf=geology,

    id_column="Spring_ID"
)


# ============================================================
# 22. SAVE SPRING MASTER TABLE
# ============================================================

spring_output_columns = [

    "Spring_ID",
    "Spring_name",

    "PBMMAE_domain_raw",
    "PBMMAE_domain",
    "PBMMAE_domain_tied",

    "PBMMAE_agreement",
    "PBMMAE_model_support",
    "PBMMAE_ambiguity",

    "Geology_match_count",
    "Geology_boundary_ambiguous",

    "class_id",
    "unit_code",
    "unit_name",
    "age",
    "lithology",

    "GPI_continuous",
    "GPI_nearest_class"
]


spring_master_df = pd.DataFrame(
    springs_with_geology[
        spring_output_columns
    ]
)


spring_master_file = (
    OUTPUT_ROOT /
    "Paper1_warm_spring_external_validation_MASTER.csv"
)


spring_master_df.to_csv(
    spring_master_file,
    index=False
)


# ============================================================
# 23. SAVE SPRINGS GPKG
# ============================================================

spring_gpkg_file = (
    OUTPUT_ROOT /
    "Paper1_warm_springs_external_validation.gpkg"
)


if spring_gpkg_file.exists():

    spring_gpkg_file.unlink()


springs_with_geology.to_file(

    spring_gpkg_file,

    layer="warm_springs",

    driver="GPKG"
)


# ============================================================
# 24. PRELIMINARY DOMAIN × GEOLOGY COUNTS
# ============================================================

geology_valid = domain_master_df[

    (
        domain_master_df[
            "unit_name"
        ].notna()
    )

    &

    (
        ~domain_master_df[
            "Geology_boundary_ambiguous"
        ]
    )

].copy()


domain_geology_counts = pd.crosstab(

    geology_valid[
        "PBMMAE_domain"
    ],

    geology_valid[
        "unit_name"
    ]
)


domain_geology_counts_file = (
    OUTPUT_ROOT /
    "Paper1_preliminary_domain_by_geology_counts.csv"
)


domain_geology_counts.to_csv(
    domain_geology_counts_file
)


# ============================================================
# 25. PRELIMINARY DOMAIN × GPI SUMMARY
# ============================================================

domain_gpi_summary = (

    domain_master_df

    .groupby(
        "PBMMAE_domain"
    )[
        "GPI_continuous"
    ]

    .agg(
        [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max"
        ]
    )

    .reset_index()
)


domain_gpi_summary_file = (
    OUTPUT_ROOT /
    "Paper1_preliminary_domain_GPI_descriptive_summary.csv"
)


domain_gpi_summary.to_csv(
    domain_gpi_summary_file,
    index=False
)


# ============================================================
# 26. QC SUMMARY
# ============================================================

qc_summary = pd.DataFrame(
    [
        {
            "Metric":
                "Consensus patch centers",

            "Value":
                n_consensus_centers
        },

        {
            "Metric":
                "PBMMAE resolved domain cells",

            "Value":
                n_resolved_cells
        },

        {
            "Metric":
                "PBMMAE tied unresolved cells",

            "Value":
                n_tied_cells
        },

        {
            "Metric":
                "PBMMAE unique resolved domains",

            "Value":
                domain_master_df[
                    "PBMMAE_domain"
                ].nunique()
        },

        {
            "Metric":
                "Resolved cells with geology",

            "Value":
                domain_master_df[
                    "unit_name"
                ].notna().sum()
        },

        {
            "Metric":
                "Resolved cells geology boundary ambiguous",

            "Value":
                domain_master_df[
                    "Geology_boundary_ambiguous"
                ].sum()
        },

        {
            "Metric":
                "Unique geology units at resolved cells",

            "Value":
                geology_valid[
                    "unit_name"
                ].nunique()
        },

        {
            "Metric":
                "Resolved cells with valid GPI",

            "Value":
                domain_master_df[
                    "GPI_continuous"
                ].notna().sum()
        },

        {
            "Metric":
                "Warm springs",

            "Value":
                len(
                    spring_master_df
                )
        },

        {
            "Metric":
                "Warm springs assigned resolved PBMMAE domain",

            "Value":
                spring_master_df[
                    "PBMMAE_domain"
                ].notna().sum()
        },

        {
            "Metric":
                "Warm springs falling on tied PBMMAE cells",

            "Value":
                spring_master_df[
                    "PBMMAE_domain_tied"
                ].sum()
        },

        {
            "Metric":
                "Warm springs assigned geology",

            "Value":
                spring_master_df[
                    "unit_name"
                ].notna().sum()
        },

        {
            "Metric":
                "Warm springs with valid GPI",

            "Value":
                spring_master_df[
                    "GPI_continuous"
                ].notna().sum()
        }
    ]
)


qc_summary_file = (
    OUTPUT_ROOT /
    "Paper1_external_validation_master_QC_summary.csv"
)


qc_summary.to_csv(
    qc_summary_file,
    index=False
)


# ============================================================
# 27. PRINT FINAL RESULTS
# ============================================================

print(
    "\n\n" +
    "=" * 110
)

print(
    "EXTERNAL VALIDATION MASTER QC SUMMARY"
)

print(
    "=" * 110
)


print(
    qc_summary.to_string(
        index=False
    )
)


print(
    "\n" +
    "=" * 110
)

print(
    "WARM-SPRING POST HOC ASSIGNMENTS"
)

print(
    "=" * 110
)


print(
    spring_master_df[
        [
            "Spring_name",
            "PBMMAE_domain",
            "PBMMAE_domain_tied",
            "PBMMAE_agreement",
            "unit_name",
            "GPI_continuous"
        ]
    ]
    .round(
        4
    )
    .to_string(
        index=False
    )
)


print(
    "\n" +
    "=" * 110
)

print(
    "PRELIMINARY GPI BY DOMAIN"
)

print(
    "=" * 110
)


print(
    domain_gpi_summary
    .round(
        4
    )
    .to_string(
        index=False
    )
)


print(
    "\n" +
    "=" * 110
)

print(
    "FILES CREATED"
)

print(
    "=" * 110
)


for path in [

    raster_qc_file,
    grid_check_file,
    geology_attribute_file,
    domain_master_file,
    spring_master_file,
    domain_geology_counts_file,
    domain_gpi_summary_file,
    qc_summary_file,
    domain_gpkg_file,
    spring_gpkg_file
]:

    print(
        path
    )


print(
    "\nIMPORTANT:"
)

print(
    "The -1 consensus code was treated as tied/unresolved "
    "model uncertainty, not as a fifth domain."
)

print(
    "Only resolved M0-M3 cells enter categorical external "
    "validation."
)

print(
    "No geology, warm-spring or GPI information was used "
    "to alter the frozen PB-MMAE solution."
)

print(
    "=" * 110
)