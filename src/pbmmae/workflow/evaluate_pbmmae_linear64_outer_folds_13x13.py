import os
# ============================================================
# Paper 1 — Frozen PB-MMAE LinearFusion64 outer-fold
# representation extraction and transfer diagnostics
#
# Evaluation folds:
#   1, 2, 4, 5
#
# Fold 3 is the development fold and is excluded here.
#
# For each outer fold:
#   - load frozen trained model
#   - load TRAIN and held-out VALIDATION patches
#   - extract UNMASKED fused 64-D representations
#   - extract raw and LayerNorm family embeddings
#   - compute representation-transfer diagnostics
#   - save fused embeddings and per-fold diagnostics
#
# No model training occurs in this script.
# No clustering occurs in this script.
# ============================================================

import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

from pathlib import Path

# PyTorch first
import torch
import torch.nn as nn
import torch.nn.functional as F

# Remaining libraries
import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(os.environ["PBMMAE_PROJECT_ROOT"]).expanduser().resolve()

PATCH_DIR = (
    PROJECT_ROOT /
    "03_outputs" /
    "patches"
)

MODEL_ROOT = (
    PROJECT_ROOT /
    "03_outputs" /
    "models" /
    "PBMMAE" /
    "OuterFolds_13x13_LinearFusion64"
)

OUTPUT_DIR = (
    MODEL_ROOT /
    "representation_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. FROZEN CONFIGURATION
# ============================================================

OUTER_FOLDS = [
    1,
    2,
    4,
    5
]

PATCH_SIZE = 13

N_CHANNELS = 14
N_FAMILIES = 6

BRANCH_EMBED_DIM = 16

FUSION_INPUT_DIM = 96
FUSED_LATENT_DIM = 64

DEVICE = torch.device(
    "cpu"
)

torch.set_num_threads(
    2
)


print("=" * 90)
print("PB-MMAE LINEAR-FUSION-64 OUTER-FOLD REPRESENTATION EVALUATION")
print("=" * 90)

print("Outer folds :", OUTER_FOLDS)
print("Development fold excluded : 3")
print("Device      :", DEVICE)


# ============================================================
# 3. CHANNEL / PHYSICS-FAMILY DEFINITIONS
# ============================================================

CHANNEL_NAMES = [

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

    "magnetic": [
        "RTE_TMI"
    ],

    "gravity": [
        "CBG",
        "CBG_RES",
        "HGM"
    ],

    "radiometric": [
        "K",
        "eTh",
        "eU"
    ],

    "thermal": [
        "CPD"
    ],

    "structural": [
        "MAG_LD",
        "GRAV_LD",
        "DEM_LD",
        "ID"
    ],

    "terrain": [
        "DEM",
        "SLOPE"
    ]
}


PHYSICS_FAMILY_INDICES = {

    family: [
        CHANNEL_NAMES.index(
            channel
        )
        for channel
        in channels
    ]

    for family, channels
    in PHYSICS_FAMILIES.items()
}


FAMILY_NAMES = list(
    PHYSICS_FAMILIES.keys()
)


# ============================================================
# 4. INTEGRITY CHECKS
# ============================================================

assert len(
    CHANNEL_NAMES
) == N_CHANNELS

assert len(
    FAMILY_NAMES
) == N_FAMILIES

assert (
    N_FAMILIES *
    BRANCH_EMBED_DIM
    ==
    FUSION_INPUT_DIM
)


# ============================================================
# 5. MODEL DEFINITION
# Must exactly match the frozen outer-fold architecture
# ============================================================

class FamilyEncoder(
    nn.Module
):

    def __init__(
        self,
        in_channels,
        embed_dim=16
    ):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.AdaptiveAvgPool2d(
                (3, 3)
            )
        )

        self.fc = nn.Linear(
            32 * 3 * 3,
            embed_dim
        )


    def forward(
        self,
        x
    ):

        x = self.features(
            x
        )

        x = torch.flatten(
            x,
            start_dim=1
        )

        return self.fc(
            x
        )


class FamilyDecoder(
    nn.Module
):

    def __init__(
        self,
        latent_dim,
        out_channels
    ):

        super().__init__()

        self.fc = nn.Linear(
            latent_dim,
            32 * 3 * 3
        )

        self.conv = nn.Sequential(

            nn.Conv2d(
                32,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                16,
                out_channels,
                kernel_size=3,
                padding=1
            )
        )


    def forward(
        self,
        z,
        output_size
    ):

        x = self.fc(
            z
        )

        x = x.view(
            -1,
            32,
            3,
            3
        )

        x = F.interpolate(

            x,

            size=
                output_size,

            mode=
                "bilinear",

            align_corners=
                False
        )

        return self.conv(
            x
        )


class PBMMAE_Linear64(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()


        # ----------------------------------------------------
        # Six physics-family encoders
        # ----------------------------------------------------

        self.encoders = nn.ModuleDict()

        for family, channels in (
            PHYSICS_FAMILIES.items()
        ):

            self.encoders[
                family
            ] = FamilyEncoder(

                in_channels=
                    len(
                        channels
                    ),

                embed_dim=
                    BRANCH_EMBED_DIM
            )


        # ----------------------------------------------------
        # Per-family LayerNorm
        # ----------------------------------------------------

        self.branch_norms = nn.ModuleDict({

            family:
                nn.LayerNorm(
                    BRANCH_EMBED_DIM
                )

            for family
            in FAMILY_NAMES
        })


        # ----------------------------------------------------
        # Frozen single-linear fusion
        #
        # 96 -> 64
        # ----------------------------------------------------

        self.fusion = nn.Linear(

            FUSION_INPUT_DIM,

            FUSED_LATENT_DIM
        )


        # ----------------------------------------------------
        # Decoders retained for state-dict compatibility
        # ----------------------------------------------------

        self.decoders = nn.ModuleDict()

        for family, channels in (
            PHYSICS_FAMILIES.items()
        ):

            self.decoders[
                family
            ] = FamilyDecoder(

                latent_dim=
                    FUSED_LATENT_DIM,

                out_channels=
                    len(
                        channels
                    )
            )


    def encode(
        self,
        family_inputs
    ):

        raw_embeddings = []

        normalized_embeddings = []


        for family in FAMILY_NAMES:

            z_family = (
                self.encoders[
                    family
                ](
                    family_inputs[
                        family
                    ]
                )
            )


            raw_embeddings.append(
                z_family
            )


            z_norm = (
                self.branch_norms[
                    family
                ](
                    z_family
                )
            )


            normalized_embeddings.append(
                z_norm
            )


        z_concat = torch.cat(

            normalized_embeddings,

            dim=1
        )


        z_fused = self.fusion(
            z_concat
        )


        return (
            z_fused,
            raw_embeddings,
            normalized_embeddings
        )


# ============================================================
# 6. FAMILY SPLITTER
# ============================================================

def split_into_families(
    x
):

    outputs = {}

    for family, indices in (
        PHYSICS_FAMILY_INDICES.items()
    ):

        outputs[
            family
        ] = x[
            :,
            indices,
            :,
            :
        ]

    return outputs


# ============================================================
# 7. UNMASKED REPRESENTATION EXTRACTION
# ============================================================

def encode_unmasked(
    model,
    tensor,
    batch_size=64
):

    fused_batches = []

    raw_branch_batches = {

        family: []

        for family
        in FAMILY_NAMES
    }

    norm_branch_batches = {

        family: []

        for family
        in FAMILY_NAMES
    }


    with torch.no_grad():

        for start in range(
            0,
            len(
                tensor
            ),
            batch_size
        ):

            batch = tensor[
                start:
                start + batch_size
            ].to(
                DEVICE
            )


            family_inputs = (
                split_into_families(
                    batch
                )
            )


            (
                z_fused,
                raw_embeddings,
                norm_embeddings
            ) = model.encode(
                family_inputs
            )


            fused_batches.append(
                z_fused
                .cpu()
                .numpy()
            )


            for i, family in enumerate(
                FAMILY_NAMES
            ):

                raw_branch_batches[
                    family
                ].append(

                    raw_embeddings[
                        i
                    ]
                    .cpu()
                    .numpy()
                )


                norm_branch_batches[
                    family
                ].append(

                    norm_embeddings[
                        i
                    ]
                    .cpu()
                    .numpy()
                )


    fused = np.concatenate(
        fused_batches,
        axis=0
    )


    raw_branches = {

        family:
            np.concatenate(
                raw_branch_batches[
                    family
                ],
                axis=0
            )

        for family
        in FAMILY_NAMES
    }


    norm_branches = {

        family:
            np.concatenate(
                norm_branch_batches[
                    family
                ],
                axis=0
            )

        for family
        in FAMILY_NAMES
    }


    return (
        fused,
        raw_branches,
        norm_branches
    )


# ============================================================
# 8. CORRELATION DIAGNOSTIC
#
# Use torch.corrcoef because np.corrcoef caused the prior
# Windows OpenMP DLL conflict.
# ============================================================

def mean_abs_offdiag_correlation(
    array
):

    x = torch.from_numpy(
        array.astype(
            np.float32
        )
    ).T


    corr = torch.corrcoef(
        x
    )


    n = corr.shape[0]


    offdiag_mask = ~torch.eye(
        n,
        dtype=torch.bool
    )


    values = torch.abs(
        corr[
            offdiag_mask
        ]
    )


    values = values[
        torch.isfinite(
            values
        )
    ]


    return float(
        values.mean().item()
    )


# ============================================================
# 9. PROCESS ONE OUTER FOLD
# ============================================================

def evaluate_fold(
    fold
):

    print(
        "\n\n" +
        "#" * 90
    )

    print(
        f"OUTER FOLD {fold}"
    )

    print(
        "#" * 90
    )


    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    fold_model_dir = (

        MODEL_ROOT /
        f"Fold{fold}"
    )


    model_file = (

        fold_model_dir /
        f"PBMMAE_LINEAR64_"
        f"Fold{fold}_13x13_FINAL.pt"
    )


    train_file = (

        PATCH_DIR /
        "13x13" /
        f"Fold_{fold}" /
        f"Fold{fold}_TRAIN_13x13_14ch.npy"
    )


    val_file = (

        PATCH_DIR /
        "13x13" /
        f"Fold_{fold}" /
        f"Fold{fold}_VALIDATION_13x13_14ch.npy"
    )


    fold_output_dir = (

        OUTPUT_DIR /
        f"Fold{fold}"
    )


    fold_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for path in [
        model_file,
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
    # Load data
    # --------------------------------------------------------

    X_train = np.load(
        train_file
    ).astype(
        np.float32
    )


    X_val = np.load(
        val_file
    ).astype(
        np.float32
    )


    print(
        "\nOriginal TRAIN:",
        X_train.shape
    )

    print(
        "Original VALIDATION:",
        X_val.shape
    )


    assert X_train.shape[1:] == (
        PATCH_SIZE,
        PATCH_SIZE,
        N_CHANNELS
    )


    assert X_val.shape[1:] == (
        PATCH_SIZE,
        PATCH_SIZE,
        N_CHANNELS
    )


    assert np.isfinite(
        X_train
    ).all()


    assert np.isfinite(
        X_val
    ).all()


    # NHWC -> NCHW

    X_train = np.transpose(

        X_train,

        (
            0,
            3,
            1,
            2
        )
    )


    X_val = np.transpose(

        X_val,

        (
            0,
            3,
            1,
            2
        )
    )


    X_train_t = torch.from_numpy(
        X_train
    )


    X_val_t = torch.from_numpy(
        X_val
    )


    # --------------------------------------------------------
    # Load frozen model
    # --------------------------------------------------------

    checkpoint = torch.load(

        model_file,

        map_location=
            DEVICE,

        weights_only=
            False
    )


    assert checkpoint.get(
        "architecture_frozen"
    ) is True


    assert checkpoint.get(
        "development_fold"
    ) == 3


    assert checkpoint.get(
        "branch_layernorm"
    ) is True


    assert checkpoint.get(
        "fusion_type"
    ) == "single_linear_projection"


    assert checkpoint.get(
        "fused_latent_dim"
    ) == FUSED_LATENT_DIM


    model = (
        PBMMAE_Linear64()
        .to(
            DEVICE
        )
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    model.eval()


    print(
        "\nFrozen model loaded."
    )

    print(
        "Training seed:",
        checkpoint.get(
            "seed"
        )
    )

    print(
        "Final training PB loss:",
        checkpoint.get(
            "final_total_pb_loss"
        )
    )


    # --------------------------------------------------------
    # Extract unmasked representations
    # --------------------------------------------------------

    (
        Z_train,
        B_train_raw,
        B_train_norm
    ) = encode_unmasked(

        model,

        X_train_t
    )


    (
        Z_val,
        B_val_raw,
        B_val_norm
    ) = encode_unmasked(

        model,

        X_val_t
    )


    print(
        "\nTRAIN fused:",
        Z_train.shape
    )

    print(
        "VALIDATION fused:",
        Z_val.shape
    )


    assert Z_train.shape == (
        len(
            X_train_t
        ),
        FUSED_LATENT_DIM
    )


    assert Z_val.shape == (
        len(
            X_val_t
        ),
        FUSED_LATENT_DIM
    )


    assert np.isfinite(
        Z_train
    ).all()


    assert np.isfinite(
        Z_val
    ).all()


    # --------------------------------------------------------
    # Fused latent diagnostics
    # --------------------------------------------------------

    train_sd_dims = Z_train.std(
        axis=0,
        ddof=1
    )


    val_sd_dims = Z_val.std(
        axis=0,
        ddof=1
    )


    train_mean_sd = float(
        np.mean(
            train_sd_dims
        )
    )


    val_mean_sd = float(
        np.mean(
            val_sd_dims
        )
    )


    sd_ratio = float(
        val_mean_sd /
        train_mean_sd
    )


    train_near_zero = int(
        np.sum(
            train_sd_dims <
            1e-8
        )
    )


    val_near_zero = int(
        np.sum(
            val_sd_dims <
            1e-8
        )
    )


    train_corr = (
        mean_abs_offdiag_correlation(
            Z_train
        )
    )


    val_corr = (
        mean_abs_offdiag_correlation(
            Z_val
        )
    )


    print(
        "\nFUSED REPRESENTATION DIAGNOSTICS"
    )


    print(
        "TRAIN mean latent SD      :",
        round(
            train_mean_sd,
            6
        )
    )


    print(
        "VALIDATION mean latent SD :",
        round(
            val_mean_sd,
            6
        )
    )


    print(
        "VAL/TRAIN SD ratio        :",
        round(
            sd_ratio,
            6
        )
    )


    print(
        "TRAIN near-zero dims      :",
        train_near_zero
    )


    print(
        "VALIDATION near-zero dims :",
        val_near_zero
    )


    print(
        "TRAIN mean abs corr       :",
        round(
            train_corr,
            6
        )
    )


    print(
        "VALIDATION mean abs corr  :",
        round(
            val_corr,
            6
        )
    )


    # --------------------------------------------------------
    # Branch-level diagnostics
    # --------------------------------------------------------

    branch_records = []


    for family in FAMILY_NAMES:

        train_raw = (
            B_train_raw[
                family
            ]
        )

        val_raw = (
            B_val_raw[
                family
            ]
        )

        train_norm = (
            B_train_norm[
                family
            ]
        )

        val_norm = (
            B_val_norm[
                family
            ]
        )


        train_raw_sd = (
            train_raw.std(
                axis=0,
                ddof=1
            )
        )

        val_raw_sd = (
            val_raw.std(
                axis=0,
                ddof=1
            )
        )


        train_norm_sd = (
            train_norm.std(
                axis=0,
                ddof=1
            )
        )

        val_norm_sd = (
            val_norm.std(
                axis=0,
                ddof=1
            )
        )


        train_raw_mean_sd = float(
            np.mean(
                train_raw_sd
            )
        )


        val_raw_mean_sd = float(
            np.mean(
                val_raw_sd
            )
        )


        train_norm_mean_sd = float(
            np.mean(
                train_norm_sd
            )
        )


        val_norm_mean_sd = float(
            np.mean(
                val_norm_sd
            )
        )


        branch_records.append({

            "Fold":
                fold,

            "Physics_family":
                family,

            "Embedding_dim":
                BRANCH_EMBED_DIM,

            "TRAIN_raw_mean_dimension_SD":
                train_raw_mean_sd,

            "VALIDATION_raw_mean_dimension_SD":
                val_raw_mean_sd,

            "RAW_VAL_to_TRAIN_SD_ratio":
                float(
                    val_raw_mean_sd /
                    train_raw_mean_sd
                ),

            "TRAIN_norm_mean_dimension_SD":
                train_norm_mean_sd,

            "VALIDATION_norm_mean_dimension_SD":
                val_norm_mean_sd,

            "NORM_VAL_to_TRAIN_SD_ratio":
                float(
                    val_norm_mean_sd /
                    train_norm_mean_sd
                ),

            "TRAIN_raw_near_zero_dims":
                int(
                    np.sum(
                        train_raw_sd <
                        1e-8
                    )
                ),

            "VALIDATION_raw_near_zero_dims":
                int(
                    np.sum(
                        val_raw_sd <
                        1e-8
                    )
                ),

            "TRAIN_norm_near_zero_dims":
                int(
                    np.sum(
                        train_norm_sd <
                        1e-8
                    )
                ),

            "VALIDATION_norm_near_zero_dims":
                int(
                    np.sum(
                        val_norm_sd <
                        1e-8
                    )
                )
        })


    branch_df = pd.DataFrame(
        branch_records
    )


    print(
        "\nNORMALIZED FAMILY TRANSFER RATIOS"
    )


    print(

        branch_df[
            [
                "Physics_family",
                "NORM_VAL_to_TRAIN_SD_ratio"
            ]
        ]
        .round(
            4
        )
        .to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # Save fused representations
    # --------------------------------------------------------

    train_latent_file = (

        fold_output_dir /
        f"PBMMAE_LINEAR64_"
        f"Fold{fold}_13x13_"
        f"TRAIN_latent64.npy"
    )


    val_latent_file = (

        fold_output_dir /
        f"PBMMAE_LINEAR64_"
        f"Fold{fold}_13x13_"
        f"VALIDATION_latent64.npy"
    )


    np.save(
        train_latent_file,
        Z_train
    )


    np.save(
        val_latent_file,
        Z_val
    )


    # --------------------------------------------------------
    # Save CSV versions of fused representations
    # --------------------------------------------------------

    latent_columns = [

        f"Z{i+1:02d}"

        for i in range(
            FUSED_LATENT_DIM
        )
    ]


    pd.DataFrame(

        Z_train,

        columns=
            latent_columns

    ).to_csv(

        fold_output_dir /
        f"PBMMAE_LINEAR64_"
        f"Fold{fold}_13x13_"
        f"TRAIN_latent64.csv",

        index=False
    )


    pd.DataFrame(

        Z_val,

        columns=
            latent_columns

    ).to_csv(

        fold_output_dir /
        f"PBMMAE_LINEAR64_"
        f"Fold{fold}_13x13_"
        f"VALIDATION_latent64.csv",

        index=False
    )


    # --------------------------------------------------------
    # Save normalized family embeddings
    # --------------------------------------------------------

    for family in FAMILY_NAMES:

        np.save(

            fold_output_dir /
            f"PBMMAE_LINEAR64_"
            f"Fold{fold}_13x13_"
            f"TRAIN_{family}_NORM_latent16.npy",

            B_train_norm[
                family
            ]
        )


        np.save(

            fold_output_dir /
            f"PBMMAE_LINEAR64_"
            f"Fold{fold}_13x13_"
            f"VALIDATION_{family}_NORM_latent16.npy",

            B_val_norm[
                family
            ]
        )


    # --------------------------------------------------------
    # Save per-fold fused diagnostic
    # --------------------------------------------------------

    fused_record = {

        "Fold":
            fold,

        "Development_fold":
            3,

        "Architecture_frozen":
            True,

        "TRAIN_patches":
            Z_train.shape[0],

        "VALIDATION_patches":
            Z_val.shape[0],

        "Fused_latent_dim":
            FUSED_LATENT_DIM,

        "TRAIN_mean_latent_SD":
            train_mean_sd,

        "VALIDATION_mean_latent_SD":
            val_mean_sd,

        "VAL_to_TRAIN_latent_SD_ratio":
            sd_ratio,

        "TRAIN_near_zero_dims":
            train_near_zero,

        "VALIDATION_near_zero_dims":
            val_near_zero,

        "TRAIN_mean_abs_offdiag_correlation":
            train_corr,

        "VALIDATION_mean_abs_offdiag_correlation":
            val_corr
    }


    fused_df = pd.DataFrame(
        [
            fused_record
        ]
    )


    fused_file = (

        fold_output_dir /
        f"PBMMAE_LINEAR64_"
        f"Fold{fold}_13x13_"
        f"representation_diagnostics.csv"
    )


    fused_df.to_csv(
        fused_file,
        index=False
    )


    # --------------------------------------------------------
    # Save branch diagnostic
    # --------------------------------------------------------

    branch_file = (

        fold_output_dir /
        f"PBMMAE_LINEAR64_"
        f"Fold{fold}_13x13_"
        f"branch_embedding_diagnostics.csv"
    )


    branch_df.to_csv(
        branch_file,
        index=False
    )


    print(
        "\nSaved fused diagnostics:"
    )

    print(
        fused_file
    )


    print(
        "\nSaved branch diagnostics:"
    )

    print(
        branch_file
    )


    return (
        fused_record,
        branch_records
    )


# ============================================================
# 10. RUN ALL OUTER FOLDS
# ============================================================

all_fused_records = []

all_branch_records = []


for fold in OUTER_FOLDS:

    (
        fused_record,
        branch_records
    ) = evaluate_fold(
        fold
    )


    all_fused_records.append(
        fused_record
    )


    all_branch_records.extend(
        branch_records
    )


# ============================================================
# 11. COMBINED OUTER-FOLD FUSED DIAGNOSTICS
# ============================================================

outer_fused_df = pd.DataFrame(
    all_fused_records
)


combined_fused_file = (

    OUTPUT_DIR /
    "PBMMAE_LINEAR64_outer_folds_representation_diagnostics.csv"
)


outer_fused_df.to_csv(
    combined_fused_file,
    index=False
)


# ============================================================
# 12. COMBINED BRANCH DIAGNOSTICS
# ============================================================

outer_branch_df = pd.DataFrame(
    all_branch_records
)


combined_branch_file = (

    OUTPUT_DIR /
    "PBMMAE_LINEAR64_outer_folds_branch_embedding_diagnostics.csv"
)


outer_branch_df.to_csv(
    combined_branch_file,
    index=False
)


# ============================================================
# 13. OUTER-FOLD SUMMARY STATISTICS
# ============================================================

ratio_values = (
    outer_fused_df[
        "VAL_to_TRAIN_latent_SD_ratio"
    ]
)


summary_df = pd.DataFrame(
    [{
        "N_outer_folds":
            len(
                OUTER_FOLDS
            ),

        "Mean_VAL_to_TRAIN_latent_SD_ratio":
            float(
                ratio_values.mean()
            ),

        "SD_VAL_to_TRAIN_latent_SD_ratio":
            float(
                ratio_values.std(
                    ddof=1
                )
            ),

        "Min_VAL_to_TRAIN_latent_SD_ratio":
            float(
                ratio_values.min()
            ),

        "Max_VAL_to_TRAIN_latent_SD_ratio":
            float(
                ratio_values.max()
            ),

        "Mean_TRAIN_abs_offdiag_corr":
            float(
                outer_fused_df[
                    "TRAIN_mean_abs_offdiag_correlation"
                ].mean()
            ),

        "Mean_VALIDATION_abs_offdiag_corr":
            float(
                outer_fused_df[
                    "VALIDATION_mean_abs_offdiag_correlation"
                ].mean()
            ),

        "Total_TRAIN_near_zero_dims":
            int(
                outer_fused_df[
                    "TRAIN_near_zero_dims"
                ].sum()
            ),

        "Total_VALIDATION_near_zero_dims":
            int(
                outer_fused_df[
                    "VALIDATION_near_zero_dims"
                ].sum()
            )
    }]
)


summary_file = (

    OUTPUT_DIR /
    "PBMMAE_LINEAR64_outer_folds_representation_summary.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# 14. PRINT COMBINED RESULTS
# ============================================================

print(
    "\n\n" +
    "=" * 110
)

print(
    "OUTER-FOLD FUSED REPRESENTATION SUMMARY"
)

print(
    "=" * 110
)


print(

    outer_fused_df[
        [
            "Fold",
            "TRAIN_patches",
            "VALIDATION_patches",
            "TRAIN_mean_latent_SD",
            "VALIDATION_mean_latent_SD",
            "VAL_to_TRAIN_latent_SD_ratio",
            "TRAIN_near_zero_dims",
            "VALIDATION_near_zero_dims",
            "TRAIN_mean_abs_offdiag_correlation",
            "VALIDATION_mean_abs_offdiag_correlation"
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
    "OUTER-FOLD REPRESENTATION SUMMARY STATISTICS"
)

print(
    "=" * 110
)


print(
    summary_df
    .round(
        4
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 15. FINAL OUTPUT
# ============================================================

print(
    "\n" +
    "=" * 90
)

print(
    "OUTER-FOLD REPRESENTATION EVALUATION COMPLETED"
)

print(
    "=" * 90
)


print(
    "\nCombined fused diagnostics:"
)

print(
    combined_fused_file
)


print(
    "\nCombined branch diagnostics:"
)

print(
    combined_branch_file
)


print(
    "\nOverall summary:"
)

print(
    summary_file
)


print(
    "\nRepresentation directory:"
)

print(
    OUTPUT_DIR
)


print(
    "=" * 90
)