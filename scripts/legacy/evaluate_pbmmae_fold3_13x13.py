import os
# ============================================================
# Paper 1 — PB-MMAE representation extraction
# Fold 3, 13x13
#
# Uses FINAL trained model.
# No further optimization.
#
# Extracts unmasked fused 64-D representations for:
#   - TRAIN
#   - held-out VALIDATION
#
# Also computes representation diagnostics.
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
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(os.environ["PBMMAE_PROJECT_ROOT"]).expanduser().resolve()

PATCH_DIR = (
    PROJECT_ROOT /
    "03_outputs" /
    "patches"
)

RUN_DIR = (
    PROJECT_ROOT /
    "03_outputs" /
    "models" /
    "PBMMAE" /
    "Fold3_13x13"
)

MODEL_FILE = (
    RUN_DIR /
    "PBMMAE_Fold3_13x13_FINAL.pt"
)

TRAIN_FILE = (
    PATCH_DIR /
    "13x13" /
    "Fold_3" /
    "Fold3_TRAIN_13x13_14ch.npy"
)

VAL_FILE = (
    PATCH_DIR /
    "13x13" /
    "Fold_3" /
    "Fold3_VALIDATION_13x13_14ch.npy"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

N_CHANNELS = 14

BRANCH_EMBED_DIM = 16
FUSION_INPUT_DIM = 96
FUSED_LATENT_DIM = 64

DEVICE = torch.device("cpu")

torch.set_num_threads(2)

print("=" * 78)
print("PB-MMAE REPRESENTATION EXTRACTION")
print("=" * 78)

print("Device:", DEVICE)


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
        CHANNEL_NAMES.index(ch)
        for ch in channels
    ]

    for family, channels
    in PHYSICS_FAMILIES.items()
}


FAMILY_NAMES = list(
    PHYSICS_FAMILIES.keys()
)


# ============================================================
# 4. MODEL DEFINITION
# Must exactly match training script
# ============================================================

class FamilyEncoder(nn.Module):

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

            nn.ReLU(inplace=True),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

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


class FamilyDecoder(nn.Module):

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

            nn.ReLU(inplace=True),

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
            size=output_size,
            mode="bilinear",
            align_corners=False
        )

        return self.conv(
            x
        )


class PBMMAE(nn.Module):

    def __init__(
        self
    ):

        super().__init__()

        self.encoders = nn.ModuleDict()

        for family, channels in PHYSICS_FAMILIES.items():

            self.encoders[
                family
            ] = FamilyEncoder(

                in_channels=
                    len(channels),

                embed_dim=
                    BRANCH_EMBED_DIM
            )


        self.fusion = nn.Sequential(

            nn.Linear(
                FUSION_INPUT_DIM,
                128
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                128,
                FUSED_LATENT_DIM
            )
        )


        self.decoders = nn.ModuleDict()

        for family, channels in PHYSICS_FAMILIES.items():

            self.decoders[
                family
            ] = FamilyDecoder(

                latent_dim=
                    FUSED_LATENT_DIM,

                out_channels=
                    len(channels)
            )


    def encode(
        self,
        family_inputs
    ):

        branch_embeddings = []

        for family in FAMILY_NAMES:

            z_family = self.encoders[
                family
            ](
                family_inputs[
                    family
                ]
            )

            branch_embeddings.append(
                z_family
            )

        z_concat = torch.cat(
            branch_embeddings,
            dim=1
        )

        z_fused = self.fusion(
            z_concat
        )

        return (
            z_fused,
            branch_embeddings
        )


    def decode(
        self,
        z_fused,
        output_size
    ):

        reconstructions = {}

        for family in FAMILY_NAMES:

            reconstructions[
                family
            ] = self.decoders[
                family
            ](
                z_fused,
                output_size
            )

        return reconstructions


    def forward(
        self,
        family_inputs
    ):

        first_family = FAMILY_NAMES[0]

        output_size = (

            family_inputs[
                first_family
            ].shape[-2],

            family_inputs[
                first_family
            ].shape[-1]
        )

        (
            z_fused,
            branch_embeddings
        ) = self.encode(
            family_inputs
        )

        reconstructions = self.decode(
            z_fused,
            output_size
        )

        return (
            reconstructions,
            z_fused,
            branch_embeddings
        )


# ============================================================
# 5. FAMILY SPLITTER
# ============================================================

def split_into_families(
    x
):

    outputs = {}

    for family, indices in PHYSICS_FAMILY_INDICES.items():

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
# 6. CHECK FILES
# ============================================================

for file_path in [
    MODEL_FILE,
    TRAIN_FILE,
    VAL_FILE
]:

    print(
        "\nExists:",
        file_path.exists(),
        "|",
        file_path
    )

    if not file_path.exists():

        raise FileNotFoundError(
            file_path
        )


# ============================================================
# 7. LOAD DATA
# ============================================================

X_train = np.load(
    TRAIN_FILE
).astype(
    np.float32
)

X_val = np.load(
    VAL_FILE
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


# ------------------------------------------------------------
# NHWC -> NCHW
# ------------------------------------------------------------

X_train = np.transpose(
    X_train,
    (0, 3, 1, 2)
)

X_val = np.transpose(
    X_val,
    (0, 3, 1, 2)
)


X_train_t = torch.from_numpy(
    X_train
)

X_val_t = torch.from_numpy(
    X_val
)


assert torch.isfinite(
    X_train_t
).all()

assert torch.isfinite(
    X_val_t
).all()


print(
    "\nPyTorch TRAIN:",
    X_train_t.shape
)

print(
    "PyTorch VALIDATION:",
    X_val_t.shape
)


# ============================================================
# 8. LOAD FINAL MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False
)

model = PBMMAE().to(
    DEVICE
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()

print(
    "\nLoaded training epoch count:",
    checkpoint.get(
        "epochs",
        "not found"
    )
)

print(
    "Saved final PB loss:",
    checkpoint.get(
        "final_total_pb_loss",
        "not found"
    )
)


# ============================================================
# 9. EXTRACT UNMASKED REPRESENTATIONS
# ============================================================

def encode_unmasked(
    tensor,
    batch_size=64
):

    fused_batches = []

    branch_batches = {

        family: []

        for family
        in FAMILY_NAMES
    }

    with torch.no_grad():

        for start in range(
            0,
            len(tensor),
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
                branch_embeddings
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

                branch_batches[
                    family
                ].append(

                    branch_embeddings[
                        i
                    ]
                    .cpu()
                    .numpy()
                )


    fused = np.concatenate(
        fused_batches,
        axis=0
    )


    branches = {

        family:
            np.concatenate(
                branch_batches[
                    family
                ],
                axis=0
            )

        for family
        in FAMILY_NAMES
    }


    return (
        fused,
        branches
    )


Z_train, B_train = (
    encode_unmasked(
        X_train_t
    )
)

Z_val, B_val = (
    encode_unmasked(
        X_val_t
    )
)


print(
    "\nTRAIN fused representation:",
    Z_train.shape
)

print(
    "VALIDATION fused representation:",
    Z_val.shape
)

print(
    "TRAIN finite:",
    np.isfinite(
        Z_train
    ).all()
)

print(
    "VALIDATION finite:",
    np.isfinite(
        Z_val
    ).all()
)


assert Z_train.shape == (
    len(X_train_t),
    FUSED_LATENT_DIM
)

assert Z_val.shape == (
    len(X_val_t),
    FUSED_LATENT_DIM
)

assert np.isfinite(
    Z_train
).all()

assert np.isfinite(
    Z_val
).all()


# ============================================================
# 10. FUSED-LATENT DIAGNOSTICS
# ============================================================

train_std = Z_train.std(
    axis=0,
    ddof=1
)

val_std = Z_val.std(
    axis=0,
    ddof=1
)


train_near_zero = int(
    np.sum(
        train_std <
        1e-8
    )
)

val_near_zero = int(
    np.sum(
        val_std <
        1e-8
    )
)


print(
    "\nNear-zero TRAIN fused dimensions:",
    train_near_zero
)

print(
    "Near-zero VALIDATION fused dimensions:",
    val_near_zero
)


# ============================================================
# Mean absolute off-diagonal latent correlation
#
# Use PyTorch instead of np.corrcoef because the current
# Windows environment contains competing OpenMP runtimes.
# ============================================================

def torch_mean_abs_offdiag_correlation(
    array
):

    x = torch.from_numpy(
        array.astype(np.float32)
    )

    # Dimensions should be rows for torch.corrcoef:
    # (64 latent dimensions, N samples)
    x = x.T

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

    # Remove any possible non-finite correlations
    values = values[
        torch.isfinite(values)
    ]

    return float(
        values.mean().item()
    )


train_mean_abs_corr = (
    torch_mean_abs_offdiag_correlation(
        Z_train
    )
)

val_mean_abs_corr = (
    torch_mean_abs_offdiag_correlation(
        Z_val
    )
)


print(
    "\nMean absolute off-diagonal correlation"
)

print(
    "TRAIN:",
    round(
        train_mean_abs_corr,
        4
    )
)

print(
    "VALIDATION:",
    round(
        val_mean_abs_corr,
        4
    )
)


# ============================================================
# 11. BRANCH EMBEDDING DIAGNOSTICS
# ============================================================

branch_records = []

for family in FAMILY_NAMES:

    train_branch = (
        B_train[
            family
        ]
    )

    val_branch = (
        B_val[
            family
        ]
    )


    train_branch_std = (
        train_branch.std(
            axis=0,
            ddof=1
        )
    )

    val_branch_std = (
        val_branch.std(
            axis=0,
            ddof=1
        )
    )


    branch_records.append({

        "Physics_family":
            family,

        "Embedding_dim":
            train_branch.shape[1],

        "TRAIN_mean_abs":
            float(
                np.mean(
                    np.abs(
                        train_branch
                    )
                )
            ),

        "VALIDATION_mean_abs":
            float(
                np.mean(
                    np.abs(
                        val_branch
                    )
                )
            ),

        "TRAIN_mean_dimension_SD":
            float(
                np.mean(
                    train_branch_std
                )
            ),

        "VALIDATION_mean_dimension_SD":
            float(
                np.mean(
                    val_branch_std
                )
            ),

        "TRAIN_near_zero_dims":
            int(
                np.sum(
                    train_branch_std <
                    1e-8
                )
            ),

        "VALIDATION_near_zero_dims":
            int(
                np.sum(
                    val_branch_std <
                    1e-8
                )
            )
    })


branch_df = pd.DataFrame(
    branch_records
)


print(
    "\nBRANCH EMBEDDING DIAGNOSTICS"
)

print(
    branch_df.round(
        5
    ).to_string(
        index=False
    )
)


# ============================================================
# 12. SAVE FUSED LATENTS
# ============================================================

train_latent_file = (
    RUN_DIR /
    "PBMMAE_Fold3_13x13_TRAIN_latent64.npy"
)

val_latent_file = (
    RUN_DIR /
    "PBMMAE_Fold3_13x13_VALIDATION_latent64.npy"
)


np.save(
    train_latent_file,
    Z_train
)

np.save(
    val_latent_file,
    Z_val
)


# ============================================================
# 13. SAVE FUSED LATENTS AS CSV
# ============================================================

latent_columns = [

    f"Z{i+1:02d}"

    for i in range(
        FUSED_LATENT_DIM
    )
]


train_latent_csv = (
    RUN_DIR /
    "PBMMAE_Fold3_13x13_TRAIN_latent64.csv"
)

val_latent_csv = (
    RUN_DIR /
    "PBMMAE_Fold3_13x13_VALIDATION_latent64.csv"
)


pd.DataFrame(
    Z_train,
    columns=latent_columns
).to_csv(
    train_latent_csv,
    index=False
)


pd.DataFrame(
    Z_val,
    columns=latent_columns
).to_csv(
    val_latent_csv,
    index=False
)


# ============================================================
# 14. SAVE BRANCH EMBEDDINGS
# ============================================================

for family in FAMILY_NAMES:

    np.save(

        RUN_DIR /
        f"PBMMAE_Fold3_13x13_"
        f"TRAIN_{family}_latent16.npy",

        B_train[
            family
        ]
    )

    np.save(

        RUN_DIR /
        f"PBMMAE_Fold3_13x13_"
        f"VALIDATION_{family}_latent16.npy",

        B_val[
            family
        ]
    )


# ============================================================
# 15. SAVE REPRESENTATION DIAGNOSTICS
# ============================================================

diagnostics_df = pd.DataFrame(
    [{
        "Fold": 3,

        "Patch_size": 13,

        "TRAIN_patches":
            Z_train.shape[0],

        "VALIDATION_patches":
            Z_val.shape[0],

        "Fused_latent_dim":
            FUSED_LATENT_DIM,

        "TRAIN_near_zero_dims":
            train_near_zero,

        "VALIDATION_near_zero_dims":
            val_near_zero,

        "TRAIN_mean_abs_offdiag_correlation":
            train_mean_abs_corr,

        "VALIDATION_mean_abs_offdiag_correlation":
            val_mean_abs_corr,

        "TRAIN_mean_latent_SD":
            float(
                np.mean(
                    train_std
                )
            ),

        "VALIDATION_mean_latent_SD":
            float(
                np.mean(
                    val_std
                )
            )
    }]
)


diagnostics_file = (
    RUN_DIR /
    "PBMMAE_Fold3_13x13_representation_diagnostics.csv"
)

diagnostics_df.to_csv(
    diagnostics_file,
    index=False
)


branch_file = (
    RUN_DIR /
    "PBMMAE_Fold3_13x13_branch_embedding_diagnostics.csv"
)

branch_df.to_csv(
    branch_file,
    index=False
)


# ============================================================
# 16. FINISH
# ============================================================

print("\n" + "=" * 78)

print(
    "PB-MMAE REPRESENTATION EXTRACTION COMPLETED"
)

print("=" * 78)


print("\nTRAIN fused latent:")
print(train_latent_file)

print("\nVALIDATION fused latent:")
print(val_latent_file)

print("\nRepresentation diagnostics:")
print(diagnostics_file)

print("\nBranch diagnostics:")
print(branch_file)

print("=" * 78)