import os
# ============================================================
# Paper 1 — Inspect available patch / spatial metadata
#
# Purpose:
# Determine which saved files can identify the geographic
# center of every 13x13 TRAIN and VALIDATION patch.
#
# No data are modified.
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ["PBMMAE_PROJECT_ROOT"]).expanduser().resolve()

PATCH_ROOT = (
    PROJECT_ROOT /
    "03_outputs" /
    "patches"
)

CONSENSUS_ROOT = (
    PROJECT_ROOT /
    "03_outputs" /
    "models" /
    "PBMMAE" /
    "OuterFolds_13x13_LinearFusion64" /
    "outer_fold_consensus_k8_k9"
)

FOLDS = [1, 2, 4, 5]


print("=" * 90)
print("PATCH / SPATIAL METADATA INVENTORY")
print("=" * 90)


# ============================================================
# 1. INSPECT PATCH DIRECTORIES
# ============================================================

for fold in FOLDS:

    fold_dir = (
        PATCH_ROOT /
        "13x13" /
        f"Fold_{fold}"
    )

    print("\n" + "#" * 90)
    print(f"FOLD {fold} PATCH DIRECTORY")
    print("#" * 90)

    print("Directory:")
    print(fold_dir)

    if not fold_dir.exists():

        print("DIRECTORY DOES NOT EXIST")
        continue


    files = sorted(
        [
            p
            for p in fold_dir.iterdir()
            if p.is_file()
        ]
    )


    for path in files:

        suffix = path.suffix.lower()

        print("\nFILE:")
        print(path.name)


        # ----------------------------------------------------
        # NPY
        # ----------------------------------------------------

        if suffix == ".npy":

            try:

                arr = np.load(
                    path,
                    mmap_mode="r",
                    allow_pickle=True
                )

                print(
                    "  type : NPY"
                )

                print(
                    "  shape:",
                    arr.shape
                )

                print(
                    "  dtype:",
                    arr.dtype
                )

            except Exception as e:

                print(
                    "  Could not inspect:",
                    repr(e)
                )


        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        elif suffix == ".csv":

            try:

                df = pd.read_csv(
                    path,
                    nrows=5
                )

                print(
                    "  type   : CSV"
                )

                print(
                    "  columns:"
                )

                print(
                    "   ",
                    list(df.columns)
                )

                print(
                    "  preview:"
                )

                print(
                    df.to_string(
                        index=False
                    )
                )

            except Exception as e:

                print(
                    "  Could not inspect:",
                    repr(e)
                )


        else:

            print(
                "  type:",
                suffix
            )


# ============================================================
# 2. INSPECT k=9 CONSENSUS OUTPUTS
# ============================================================

print("\n\n" + "=" * 90)
print("k=9 CONSENSUS LABEL INVENTORY")
print("=" * 90)


for fold in FOLDS:

    k9_dir = (
        CONSENSUS_ROOT /
        f"Fold{fold}" /
        "k9"
    )

    print("\n" + "#" * 90)
    print(f"FOLD {fold} — k=9")
    print("#" * 90)

    print(
        "Directory:",
        k9_dir
    )


    if not k9_dir.exists():

        print(
            "DIRECTORY DOES NOT EXIST"
        )

        continue


    files = sorted(
        [
            p
            for p in k9_dir.iterdir()
            if p.is_file()
        ]
    )


    for path in files:

        print("\nFILE:")
        print(path.name)


        if path.suffix.lower() == ".npy":

            try:

                arr = np.load(
                    path,
                    mmap_mode="r",
                    allow_pickle=True
                )

                print(
                    "  shape:",
                    arr.shape
                )

                print(
                    "  dtype:",
                    arr.dtype
                )

            except Exception as e:

                print(
                    "  Could not inspect:",
                    repr(e)
                )


# ============================================================
# 3. RECURSIVE SEARCH FOR LIKELY SPATIAL-METADATA FILES
# ============================================================

print("\n\n" + "=" * 90)
print("SEARCH FOR LIKELY CENTER / COORDINATE METADATA")
print("=" * 90)


keywords = [

    "center",
    "centre",
    "coord",
    "xy",
    "row",
    "col",
    "index",
    "indices",
    "meta",
    "position",
    "location",
    "patch_id",
    "patchid"
]


candidate_files = []


for path in PATCH_ROOT.rglob("*"):

    if not path.is_file():

        continue

    name_lower = (
        path.name.lower()
    )


    if any(
        keyword in name_lower
        for keyword in keywords
    ):

        candidate_files.append(
            path
        )


if len(candidate_files) == 0:

    print(
        "No filenames containing likely spatial-metadata "
        "keywords were found."
    )

else:

    for path in sorted(
        candidate_files
    ):

        print(
            path
        )


# ============================================================
# 4. ALSO SEARCH SPATIAL-CV OUTPUTS
# ============================================================

SPATIAL_CV_ROOT = (
    PROJECT_ROOT /
    "03_outputs" /
    "spatial_cv"
)


print("\n\n" + "=" * 90)
print("SPATIAL-CV DIRECTORY INVENTORY")
print("=" * 90)


if SPATIAL_CV_ROOT.exists():

    for path in sorted(
        SPATIAL_CV_ROOT.rglob("*")
    ):

        if path.is_file():

            print(
                path
            )

else:

    print(
        "Spatial-CV directory does not exist:"
    )

    print(
        SPATIAL_CV_ROOT
    )


print("\n" + "=" * 90)
print("INSPECTION COMPLETED")
print("=" * 90)