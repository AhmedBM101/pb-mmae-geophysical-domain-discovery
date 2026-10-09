import os
# ============================================================
# Paper 1 — PB-MMAE architecture unit test
# No training
# Tests 9x9 and 13x13 inputs
# ============================================================

import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

from pathlib import Path
import random

# PyTorch first
import torch
import torch.nn as nn
import torch.nn.functional as F

# Remaining libraries
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(os.environ["PBMMAE_PROJECT_ROOT"]).expanduser().resolve()

PATCH_DIR = (
    PROJECT_ROOT /
    "03_outputs" /
    "patches"
)

PBMMAE_DIR = (
    PROJECT_ROOT /
    "03_outputs" /
    "models" /
    "PBMMAE"
)

PBMMAE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TABLE_DIR = (
    PROJECT_ROOT /
    "03_outputs" /
    "tables"
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

PATCH_SIZES = [9, 13]

TEST_FOLD = 3

N_CHANNELS = 14

BRANCH_EMBED_DIM = 16
FUSION_INPUT_DIM = 96
FUSED_LATENT_DIM = 64

SPATIAL_MASK_RATIO = 0.40
MODALITY_MASK_PROB = 0.30

SMOOTH_L1_BETA = 1.0

SEED = 42

DEVICE = torch.device("cpu")

torch.set_num_threads(2)

print("=" * 78)
print("PB-MMAE ARCHITECTURE UNIT TEST")
print("=" * 78)

print("Device:", DEVICE)


# ============================================================
# CHANNEL / PHYSICS-FAMILY DEFINITION
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


N_FAMILIES = len(
    PHYSICS_FAMILIES
)

FAMILY_WEIGHT = (
    1.0 /
    N_FAMILIES
)

assert N_FAMILIES == 6
assert FUSION_INPUT_DIM == (
    N_FAMILIES *
    BRANCH_EMBED_DIM
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# FAMILY ENCODER
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

    def forward(self, x):

        x = self.features(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        return self.fc(x)


# ============================================================
# FAMILY DECODER
# ============================================================

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

        x = self.fc(z)

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

        return self.conv(x)


# ============================================================
# PB-MMAE
# ============================================================

class PBMMAE(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Six modality-specific encoders
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Multimodal fusion
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Six modality-specific decoders
        # ----------------------------------------------------

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

        for family in PHYSICS_FAMILIES:

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

        for family in PHYSICS_FAMILIES:

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

        first_family = next(
            iter(
                family_inputs
            )
        )

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
# MASKING FUNCTION
# ============================================================

def apply_batch_masks(
    x,
    spatial_mask_ratio=0.40,
    modality_mask_prob=0.30
):

    """
    x shape:
    (B, 14, H, W)

    Returns:
        masked_x
        spatial_masks
        whole_family_masks
    """

    B, C, H, W = x.shape

    masked_x = x.clone()

    n_positions = H * W

    n_spatial_mask = int(
        round(
            n_positions *
            spatial_mask_ratio
        )
    )

    spatial_masks = {}

    whole_family_masks = {}

    # --------------------------------------------------------
    # Initialize tracking masks
    # --------------------------------------------------------

    for family in PHYSICS_FAMILIES:

        spatial_masks[
            family
        ] = torch.zeros(

            B,
            H,
            W,

            dtype=torch.bool,
            device=x.device
        )

        whole_family_masks[
            family
        ] = torch.zeros(

            B,

            dtype=torch.bool,
            device=x.device
        )

    # --------------------------------------------------------
    # Per-sample masking
    # --------------------------------------------------------

    family_names = list(
        PHYSICS_FAMILIES.keys()
    )

    for b in range(B):

        # ----------------------------------------------------
        # Decide whether to completely mask one family
        # ----------------------------------------------------

        whole_family = None

        if torch.rand(
            1,
            device=x.device
        ).item() < modality_mask_prob:

            selected = torch.randint(

                low=0,
                high=len(family_names),
                size=(1,),
                device=x.device

            ).item()

            whole_family = family_names[
                selected
            ]

            whole_family_masks[
                whole_family
            ][b] = True

        # ----------------------------------------------------
        # Family-by-family spatial masks
        # ----------------------------------------------------

        for family in family_names:

            channel_indices = (
                PHYSICS_FAMILY_INDICES[
                    family
                ]
            )

            # ------------------------------------------------
            # Complete modality mask
            # ------------------------------------------------

            if family == whole_family:

                masked_x[
                    b,
                    channel_indices,
                    :,
                    :
                ] = 0.0

                spatial_masks[
                    family
                ][b] = True

                continue

            # ------------------------------------------------
            # Random spatial mask
            # ------------------------------------------------

            selected_positions = torch.randperm(

                n_positions,
                device=x.device

            )[
                :n_spatial_mask
            ]

            family_mask = torch.zeros(

                n_positions,

                dtype=torch.bool,
                device=x.device
            )

            family_mask[
                selected_positions
            ] = True

            family_mask = (
                family_mask.view(
                    H,
                    W
                )
            )

            spatial_masks[
                family
            ][b] = family_mask

            # Apply identical spatial mask to all channels
            # belonging to this physics family.

            for c in channel_indices:

                masked_x[
                    b,
                    c
                ][
                    family_mask
                ] = 0.0

    return (
        masked_x,
        spatial_masks,
        whole_family_masks
    )


# ============================================================
# SPLIT 14-CHANNEL TENSOR INTO SIX FAMILY INPUTS
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
# PHYSICS-BALANCED MASKED RECONSTRUCTION LOSS
# ============================================================

def physics_balanced_masked_loss(
    reconstructions,
    targets,
    spatial_masks
):

    family_losses = {}

    total_loss = torch.tensor(
        0.0,
        device=next(
            iter(
                reconstructions.values()
            )
        ).device
    )

    for family in PHYSICS_FAMILIES:

        pred = reconstructions[
            family
        ]

        target = targets[
            family
        ]

        mask = spatial_masks[
            family
        ]

        # ----------------------------------------------------
        # Expand B,H,W mask across channels
        # ----------------------------------------------------

        mask_expanded = (
            mask
            .unsqueeze(1)
            .expand_as(pred)
        )

        # ----------------------------------------------------
        # Reconstruction evaluated ONLY at deliberately
        # masked positions.
        # ----------------------------------------------------

        pred_masked = pred[
            mask_expanded
        ]

        target_masked = target[
            mask_expanded
        ]

        assert pred_masked.numel() > 0

        family_loss = F.smooth_l1_loss(

            pred_masked,
            target_masked,

            beta=
                SMOOTH_L1_BETA,

            reduction="mean"
        )

        family_losses[
            family
        ] = family_loss

        total_loss = (
            total_loss
            +
            FAMILY_WEIGHT
            * family_loss
        )

    return (
        total_loss,
        family_losses
    )


# ============================================================
# UNIT TESTS
# ============================================================

model = PBMMAE().to(
    DEVICE
)

total_parameters = sum(

    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(

    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    "\nTotal parameters:",
    f"{total_parameters:,}"
)

print(
    "Trainable parameters:",
    f"{trainable_parameters:,}"
)


test_records = []


for patch_size in PATCH_SIZES:

    print("\n" + "-" * 78)
    print(
        f"Testing {patch_size}x{patch_size}"
    )
    print("-" * 78)

    # --------------------------------------------------------
    # Load real Fold-3 training data
    # --------------------------------------------------------

    train_file = (

        PATCH_DIR /
        f"{patch_size}x{patch_size}" /
        f"Fold_{TEST_FOLD}" /
        f"Fold{TEST_FOLD}_TRAIN_"
        f"{patch_size}x{patch_size}_14ch.npy"
    )

    assert train_file.exists()

    X = np.load(
        train_file
    ).astype(np.float32)

    # Use only small unit-test batch
    X = X[:8]

    # NHWC -> NCHW
    X = np.transpose(
        X,
        (0, 3, 1, 2)
    )

    x = torch.from_numpy(
        X
    ).to(
        DEVICE
    )

    print(
        "Input:",
        tuple(
            x.shape
        )
    )

    assert torch.isfinite(
        x
    ).all()

    # --------------------------------------------------------
    # Apply masks
    # --------------------------------------------------------

    (
        masked_x,
        spatial_masks,
        whole_family_masks
    ) = apply_batch_masks(

        x,

        spatial_mask_ratio=
            SPATIAL_MASK_RATIO,

        modality_mask_prob=
            MODALITY_MASK_PROB
    )

    # --------------------------------------------------------
    # Family tensors
    # --------------------------------------------------------

    masked_family_inputs = (
        split_into_families(
            masked_x
        )
    )

    target_family_inputs = (
        split_into_families(
            x
        )
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        (
            reconstructions,
            z_fused,
            branch_embeddings
        ) = model(
            masked_family_inputs
        )

        (
            total_loss,
            family_losses
        ) = (
            physics_balanced_masked_loss(

                reconstructions,
                target_family_inputs,
                spatial_masks
            )
        )

    # --------------------------------------------------------
    # Assertions: fused latent
    # --------------------------------------------------------

    assert z_fused.shape == (
        len(x),
        FUSED_LATENT_DIM
    )

    assert torch.isfinite(
        z_fused
    ).all()

    # --------------------------------------------------------
    # Assertions: branch embeddings
    # --------------------------------------------------------

    assert len(
        branch_embeddings
    ) == N_FAMILIES

    for embedding in branch_embeddings:

        assert embedding.shape == (
            len(x),
            BRANCH_EMBED_DIM
        )

        assert torch.isfinite(
            embedding
        ).all()

    # --------------------------------------------------------
    # Assertions: reconstructions
    # --------------------------------------------------------

    for family, indices in PHYSICS_FAMILY_INDICES.items():

        expected_shape = (

            len(x),
            len(indices),
            patch_size,
            patch_size
        )

        assert (
            reconstructions[
                family
            ].shape
            ==
            expected_shape
        )

        assert torch.isfinite(
            reconstructions[
                family
            ]
        ).all()

    # --------------------------------------------------------
    # Assertions: loss
    # --------------------------------------------------------

    assert torch.isfinite(
        total_loss
    )

    assert total_loss.item() > 0

    assert len(
        family_losses
    ) == 6

    # --------------------------------------------------------
    # Count complete modality masks
    # --------------------------------------------------------

    n_complete_masks = sum(

        int(
            mask.sum().item()
        )

        for mask
        in whole_family_masks.values()
    )

    print(
        "Fused latent:",
        tuple(
            z_fused.shape
        )
    )

    print(
        "Total PB masked loss:",
        round(
            total_loss.item(),
            6
        )
    )

    print(
        "Whole-family masks in batch:",
        n_complete_masks
    )

    print("\nFamily reconstruction losses:")

    for family, loss in family_losses.items():

        print(
            f"  {family:12s}: "
            f"{loss.item():.6f}"
        )

    # --------------------------------------------------------
    # Record test
    # --------------------------------------------------------

    row = {

        "Patch_size":
            patch_size,

        "Batch_size":
            len(x),

        "Input_channels":
            N_CHANNELS,

        "Branch_embed_dim":
            BRANCH_EMBED_DIM,

        "Fusion_input_dim":
            FUSION_INPUT_DIM,

        "Fused_latent_dim":
            FUSED_LATENT_DIM,

        "Total_PB_masked_loss":
            total_loss.item(),

        "Whole_family_masks_in_batch":
            n_complete_masks
    }

    for family, loss in family_losses.items():

        row[
            f"{family}_loss"
        ] = loss.item()

    test_records.append(
        row
    )

    print(
        "\nAll assertions PASSED for "
        f"{patch_size}x{patch_size}."
    )


# ============================================================
# SAVE UNIT-TEST RESULTS
# ============================================================

test_df = pd.DataFrame(
    test_records
)

output_file = (
    TABLE_DIR /
    "Paper1_PBMMAE_architecture_unit_test.csv"
)

test_df.to_csv(
    output_file,
    index=False
)


print("\n" + "=" * 78)
print("PB-MMAE ARCHITECTURE UNIT TEST PASSED")
print("=" * 78)

print("\nSaved:")
print(output_file)

print("=" * 78)