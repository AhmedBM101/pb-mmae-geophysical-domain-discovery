import os
# ============================================================
# Paper 1 — PB-MMAE Ablation C
# STRUCTURAL REDUNDANCY TEST: REMOVE MAG_LD
#
# Purpose:
#   Test whether direct magnetic lineament density (MAG_LD)
#   contributes useful representation information beyond the
#   strongly correlated intersection-density channel (ID).
#
# Frozen primary settings:
#   - 13x13 patches
#   - six physics families
#   - family LayerNorm
#   - Linear fusion -> 64-D latent
#   - spatial masking = 0.40
#   - whole-modality masking = 0.30
#   - maximum one whole family masked per sample
#   - physics-balanced SmoothL1 loss
#   - AdamW lr=1e-3, wd=1e-4
#   - batch size=32
#   - epochs=300
#
# ONLY DATA ABLATION:
#   MAG_LD (original channel index 8) is removed.
#
# Validation patches are NOT loaded during training.
# ============================================================

import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

from pathlib import Path
import random
import time

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(os.environ["PBMMAE_PROJECT_ROOT"]).expanduser().resolve()

PATCH_ROOT = (
    PROJECT_ROOT /
    "03_outputs" /
    "patches" /
    "13x13"
)

OUTPUT_ROOT = (
    PROJECT_ROOT /
    "03_outputs" /
    "models" /
    "PBMMAE" /
    "ablations" /
    "C_Remove_MAG_LD_13x13"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. FROZEN CONFIGURATION
# ============================================================

FOLDS = [1, 2, 4, 5]

PATCH_SIZE = 13

BRANCH_EMBED_DIM = 16
FUSED_LATENT_DIM = 64

SPATIAL_MASK_RATIO = 0.40
WHOLE_MODALITY_PROB = 0.30

SMOOTHL1_BETA = 1.0

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

BATCH_SIZE = 32
EPOCHS = 300

BASE_SEED = 42

DEVICE = torch.device("cpu")

torch.set_num_threads(2)


# ============================================================
# 3. CHANNEL CONFIGURATION
#
# Original:
#  0 RTE_TMI
#  1 CBG
#  2 CBG_RES
#  3 HGM
#  4 K
#  5 eTh
#  6 eU
#  7 CPD
#  8 MAG_LD   <-- REMOVE
#  9 GRAV_LD
# 10 DEM_LD
# 11 ID
# 12 DEM
# 13 SLOPE
#
# Ablated 13-channel tensor:
#  0 RTE_TMI
#  1 CBG
#  2 CBG_RES
#  3 HGM
#  4 K
#  5 eTh
#  6 eU
#  7 CPD
#  8 GRAV_LD
#  9 DEM_LD
# 10 ID
# 11 DEM
# 12 SLOPE
# ============================================================

REMOVE_ORIGINAL_CHANNEL_INDEX = 8

CHANNELS = [
    "RTE_TMI",
    "CBG",
    "CBG_RES",
    "HGM",
    "K",
    "eTh",
    "eU",
    "CPD",
    "GRAV_LD",
    "DEM_LD",
    "ID",
    "DEM",
    "SLOPE"
]


PHYSICS_FAMILIES = {

    "magnetic": [
        0
    ],

    "gravity": [
        1, 2, 3
    ],

    "radiometric": [
        4, 5, 6
    ],

    "thermal": [
        7
    ],

    "structural": [
        8, 9, 10
    ],

    "terrain": [
        11, 12
    ]
}

FAMILY_NAMES = list(
    PHYSICS_FAMILIES.keys()
)


# ============================================================
# 4. REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 5. DATASET
# ============================================================

class PatchDataset(Dataset):

    def __init__(self, array_14ch):

        if array_14ch.ndim != 4:
            raise ValueError(
                f"Expected 4-D array; got {array_14ch.shape}"
            )

        if array_14ch.shape[-1] != 14:
            raise ValueError(
                "Expected original 14-channel patch array; "
                f"got {array_14ch.shape[-1]} channels"
            )

        # Remove original MAG_LD channel index 8.
        array_13ch = np.delete(
            array_14ch,
            REMOVE_ORIGINAL_CHANNEL_INDEX,
            axis=-1
        )

        if array_13ch.shape[-1] != 13:
            raise RuntimeError(
                f"Ablated array has unexpected shape: "
                f"{array_13ch.shape}"
            )

        self.data = torch.from_numpy(
            np.asarray(
                array_13ch,
                dtype=np.float32
            )
        ).permute(
            0, 3, 1, 2
        ).contiguous()


    def __len__(self):
        return self.data.shape[0]


    def __getitem__(self, idx):
        return self.data[idx]


# ============================================================
# 6. FAMILY ENCODER
# ============================================================

class FamilyEncoder(nn.Module):

    def __init__(
        self,
        in_channels,
        embed_dim=16
    ):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (3, 3)
            )
        )

        self.fc = nn.Linear(
            32 * 3 * 3,
            embed_dim
        )

        self.norm = nn.LayerNorm(
            embed_dim
        )


    def forward(self, x):

        z = self.conv(x)
        z = z.flatten(1)
        z = self.fc(z)
        z = self.norm(z)

        return z


# ============================================================
# 7. FAMILY DECODER
# ============================================================

class FamilyDecoder(nn.Module):

    def __init__(
        self,
        latent_dim,
        out_channels,
        output_size
    ):

        super().__init__()

        self.output_size = (
            output_size,
            output_size
        )

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

            nn.ReLU(),

            nn.Conv2d(
                16,
                out_channels,
                kernel_size=3,
                padding=1
            )
        )


    def forward(self, z):

        x = self.fc(z)

        x = x.view(
            z.shape[0],
            32,
            3,
            3
        )

        x = nn.functional.interpolate(
            x,
            size=self.output_size,
            mode="bilinear",
            align_corners=False
        )

        return self.conv(x)


# ============================================================
# 8. PB-MMAE
# ============================================================

class PBMMAE(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()

        for family_name in FAMILY_NAMES:

            n_channels = len(
                PHYSICS_FAMILIES[
                    family_name
                ]
            )

            self.encoders[
                family_name
            ] = FamilyEncoder(
                in_channels=n_channels,
                embed_dim=BRANCH_EMBED_DIM
            )

            self.decoders[
                family_name
            ] = FamilyDecoder(
                latent_dim=FUSED_LATENT_DIM,
                out_channels=n_channels,
                output_size=PATCH_SIZE
            )

        fusion_input_dim = (
            len(FAMILY_NAMES)
            *
            BRANCH_EMBED_DIM
        )

        self.fusion = nn.Linear(
            fusion_input_dim,
            FUSED_LATENT_DIM
        )


    def encode(self, x):

        family_embeddings = []

        for family_name in FAMILY_NAMES:

            indices = PHYSICS_FAMILIES[
                family_name
            ]

            family_x = x[
                :,
                indices,
                :,
                :
            ]

            family_z = self.encoders[
                family_name
            ](
                family_x
            )

            family_embeddings.append(
                family_z
            )

        concatenated = torch.cat(
            family_embeddings,
            dim=1
        )

        fused = self.fusion(
            concatenated
        )

        return (
            fused,
            family_embeddings
        )


    def forward(self, x):

        fused, family_embeddings = self.encode(
            x
        )

        reconstructions = {}

        for family_name in FAMILY_NAMES:

            reconstructions[
                family_name
            ] = self.decoders[
                family_name
            ](
                fused
            )

        return (
            fused,
            family_embeddings,
            reconstructions
        )


# ============================================================
# 9. FROZEN MASKING PROTOCOL
#
# Each sample:
#   with p=0.30:
#       exactly one randomly selected physics family is
#       completely masked;
#
#   remaining families:
#       40% spatial masking.
#
#   if no whole family is selected:
#       all six families receive 40% spatial masking.
#
# Zero remains the mask token.
# ============================================================

def create_masked_batch(x):

    batch_size = x.shape[0]
    height = x.shape[2]
    width = x.shape[3]

    masked_x = x.clone()

    family_masks = {
        family_name: torch.zeros(
            (
                batch_size,
                1,
                height,
                width
            ),
            dtype=torch.bool,
            device=x.device
        )
        for family_name in FAMILY_NAMES
    }

    n_spatial_mask = int(
        round(
            height *
            width *
            SPATIAL_MASK_RATIO
        )
    )

    whole_family_samples = 0


    for sample_idx in range(
        batch_size
    ):

        whole_family = None

        if torch.rand(
            1,
            device=x.device
        ).item() < WHOLE_MODALITY_PROB:

            family_position = int(
                torch.randint(
                    low=0,
                    high=len(FAMILY_NAMES),
                    size=(1,),
                    device=x.device
                ).item()
            )

            whole_family = FAMILY_NAMES[
                family_position
            ]

            whole_family_samples += 1


        for family_name in FAMILY_NAMES:

            indices = PHYSICS_FAMILIES[
                family_name
            ]


            # ------------------------------------------------
            # Entire family masked
            # ------------------------------------------------

            if family_name == whole_family:

                family_masks[
                    family_name
                ][
                    sample_idx,
                    0,
                    :,
                    :
                ] = True

                for channel_idx in indices:

                    masked_x[
                        sample_idx,
                        channel_idx,
                        :,
                        :
                    ] = 0.0

                continue


            # ------------------------------------------------
            # Normal spatial masking
            # ------------------------------------------------

            flat_indices = torch.randperm(
                height * width,
                device=x.device
            )[
                :
                n_spatial_mask
            ]

            mask_flat = torch.zeros(
                height * width,
                dtype=torch.bool,
                device=x.device
            )

            mask_flat[
                flat_indices
            ] = True

            spatial_mask = mask_flat.view(
                height,
                width
            )

            family_masks[
                family_name
            ][
                sample_idx,
                0,
                :,
                :
            ] = spatial_mask

            for channel_idx in indices:

                masked_x[
                    sample_idx,
                    channel_idx,
                    spatial_mask
                ] = 0.0


    return (
        masked_x,
        family_masks,
        whole_family_samples
    )


# ============================================================
# 10. PHYSICS-BALANCED LOSS
# ============================================================

loss_function = nn.SmoothL1Loss(
    beta=SMOOTHL1_BETA,
    reduction="none"
)


def physics_balanced_loss(
    original_x,
    reconstructions,
    family_masks
):

    family_losses = {}


    for family_name in FAMILY_NAMES:

        indices = PHYSICS_FAMILIES[
            family_name
        ]

        target = original_x[
            :,
            indices,
            :,
            :
        ]

        reconstruction = reconstructions[
            family_name
        ]

        loss_map = loss_function(
            reconstruction,
            target
        )

        mask = family_masks[
            family_name
        ].expand_as(
            loss_map
        )

        masked_values = loss_map[
            mask
        ]

        if masked_values.numel() == 0:

            raise RuntimeError(
                f"No masked elements for family "
                f"{family_name}"
            )

        family_losses[
            family_name
        ] = masked_values.mean()


    # Equal 1/6 weighting across physics families.
    total_loss = torch.stack(
        [
            family_losses[
                family_name
            ]
            for family_name
            in FAMILY_NAMES
        ]
    ).mean()


    return (
        total_loss,
        family_losses
    )


# ============================================================
# 11. TRAIN ONE OUTER FOLD
# ============================================================

def train_fold(fold):

    seed = (
        BASE_SEED +
        fold
    )

    set_seed(
        seed
    )

    fold_output = (
        OUTPUT_ROOT /
        f"Fold{fold}"
    )

    fold_output.mkdir(
        parents=True,
        exist_ok=True
    )

    train_file = (
        PATCH_ROOT /
        f"Fold_{fold}" /
        f"Fold{fold}_TRAIN_13x13_14ch.npy"
    )

    if not train_file.exists():

        raise FileNotFoundError(
            train_file
        )


    print(
        "\n" +
        "=" * 100
    )

    print(
        f"ABLATION C — FOLD {fold}"
    )

    print(
        "=" * 100
    )

    print(
        "Removed channel: MAG_LD"
    )

    print(
        "Remaining channels:",
        len(CHANNELS)
    )

    print(
        "Structural family:",
        [
            CHANNELS[i]
            for i in PHYSICS_FAMILIES[
                "structural"
            ]
        ]
    )

    print(
        "Seed:",
        seed
    )


    # --------------------------------------------------------
    # TRAIN ONLY
    # --------------------------------------------------------

    train_array = np.load(
        train_file
    )

    print(
        "Original TRAIN patches:",
        train_array.shape
    )

    train_dataset = PatchDataset(
        train_array
    )

    print(
        "Ablated TRAIN tensor:",
        tuple(
            train_dataset.data.shape
        )
    )


    generator = torch.Generator()

    generator.manual_seed(
        seed
    )


    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        generator=generator
    )


    model = PBMMAE().to(
        DEVICE
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )


    print(
        "Trainable parameters:",
        parameter_count
    )

    print(
        "Spatial mask:",
        SPATIAL_MASK_RATIO
    )

    print(
        "Whole-modality probability:",
        WHOLE_MODALITY_PROB
    )


    history = []

    total_whole_samples = 0
    total_samples_seen = 0

    start_time = time.time()


    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        epoch_total = 0.0
        epoch_samples = 0

        family_totals = {
            family_name: 0.0
            for family_name in FAMILY_NAMES
        }


        for batch in loader:

            batch = batch.to(
                DEVICE
            )

            batch_n = batch.shape[0]


            masked_batch, masks, whole_count = (
                create_masked_batch(
                    batch
                )
            )


            total_whole_samples += whole_count
            total_samples_seen += batch_n


            optimizer.zero_grad(
                set_to_none=True
            )


            _, _, reconstructions = model(
                masked_batch
            )


            total_loss, family_losses = (
                physics_balanced_loss(
                    original_x=batch,
                    reconstructions=reconstructions,
                    family_masks=masks
                )
            )


            total_loss.backward()

            optimizer.step()


            epoch_total += (
                total_loss.item() *
                batch_n
            )


            for family_name in FAMILY_NAMES:

                family_totals[
                    family_name
                ] += (
                    family_losses[
                        family_name
                    ].item()
                    *
                    batch_n
                )


            epoch_samples += batch_n


        mean_total = (
            epoch_total /
            epoch_samples
        )


        row = {

            "Fold":
                fold,

            "Epoch":
                epoch,

            "Ablation":
                "Remove_MAG_LD",

            "Total_loss":
                mean_total
        }


        for family_name in FAMILY_NAMES:

            row[
                f"{family_name}_loss"
            ] = (
                family_totals[
                    family_name
                ] /
                epoch_samples
            )


        history.append(
            row
        )


        if (
            epoch == 1
            or epoch % 25 == 0
            or epoch == EPOCHS
        ):

            print(
                f"Epoch {epoch:3d}/{EPOCHS}"
                f" | PB loss = {mean_total:.8f}"
            )


    elapsed_seconds = (
        time.time() -
        start_time
    )


    # ========================================================
    # 12. HISTORY
    # ========================================================

    history_df = pd.DataFrame(
        history
    )

    history_file = (
        fold_output /
        f"Fold{fold}_AblationC_Remove_MAG_LD_history.csv"
    )

    history_df.to_csv(
        history_file,
        index=False
    )


    # ========================================================
    # 13. MODEL
    # ========================================================

    model_file = (
        fold_output /
        f"Fold{fold}_AblationC_Remove_MAG_LD_FINAL.pt"
    )


    torch.save(
        {

            "fold":
                fold,

            "seed":
                seed,

            "ablation":
                "Remove_MAG_LD",

            "removed_original_channel_index":
                REMOVE_ORIGINAL_CHANNEL_INDEX,

            "removed_channel":
                "MAG_LD",

            "channels":
                CHANNELS,

            "physics_families":
                PHYSICS_FAMILIES,

            "patch_size":
                PATCH_SIZE,

            "branch_embed_dim":
                BRANCH_EMBED_DIM,

            "fused_latent_dim":
                FUSED_LATENT_DIM,

            "spatial_mask_ratio":
                SPATIAL_MASK_RATIO,

            "whole_modality_probability":
                WHOLE_MODALITY_PROB,

            "smoothl1_beta":
                SMOOTHL1_BETA,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "batch_size":
                BATCH_SIZE,

            "epochs":
                EPOCHS,

            "model_state_dict":
                model.state_dict()

        },

        model_file
    )


    # ========================================================
    # 14. SUMMARY
    # ========================================================

    empirical_whole_probability = (
        total_whole_samples /
        total_samples_seen
    )


    minimum_idx = history_df[
        "Total_loss"
    ].idxmin()


    summary = {

        "Fold":
            fold,

        "Ablation":
            "Remove_MAG_LD",

        "Removed_channel":
            "MAG_LD",

        "Remaining_channels":
            13,

        "Patch_size":
            PATCH_SIZE,

        "Train_patches":
            len(
                train_dataset
            ),

        "Seed":
            seed,

        "Spatial_mask_ratio":
            SPATIAL_MASK_RATIO,

        "Whole_modality_probability_target":
            WHOLE_MODALITY_PROB,

        "Whole_modality_probability_empirical":
            float(
                empirical_whole_probability
            ),

        "Initial_loss":
            float(
                history_df.iloc[
                    0
                ][
                    "Total_loss"
                ]
            ),

        "Final_loss":
            float(
                history_df.iloc[
                    -1
                ][
                    "Total_loss"
                ]
            ),

        "Minimum_loss":
            float(
                history_df.loc[
                    minimum_idx,
                    "Total_loss"
                ]
            ),

        "Minimum_loss_epoch":
            int(
                history_df.loc[
                    minimum_idx,
                    "Epoch"
                ]
            ),

        "Final25_mean_loss":
            float(
                history_df[
                    "Total_loss"
                ].tail(
                    25
                ).mean()
            ),

        "Training_seconds":
            float(
                elapsed_seconds
            ),

        "Trainable_parameters":
            int(
                parameter_count
            )
    }


    for family_name in FAMILY_NAMES:

        summary[
            f"Final_{family_name}_loss"
        ] = float(
            history_df.iloc[
                -1
            ][
                f"{family_name}_loss"
            ]
        )


    summary_df = pd.DataFrame(
        [summary]
    )


    summary_file = (
        fold_output /
        f"Fold{fold}_AblationC_Remove_MAG_LD_summary.csv"
    )


    summary_df.to_csv(
        summary_file,
        index=False
    )


    print(
        "\nCompleted Fold",
        fold
    )

    print(
        "Final PB loss:",
        round(
            summary[
                "Final_loss"
            ],
            8
        )
    )

    print(
        "Minimum PB loss:",
        round(
            summary[
                "Minimum_loss"
            ],
            8
        ),
        "at epoch",
        summary[
            "Minimum_loss_epoch"
        ]
    )

    print(
        "Whole-modality empirical probability:",
        round(
            empirical_whole_probability,
            4
        )
    )

    print(
        "Training seconds:",
        round(
            elapsed_seconds,
            2
        )
    )


    return summary


# ============================================================
# 15. RUN ALL OUTER FOLDS
# ============================================================

all_summaries = []


for fold in FOLDS:

    all_summaries.append(
        train_fold(
            fold
        )
    )


# ============================================================
# 16. COMBINED SUMMARY
# ============================================================

combined_df = pd.DataFrame(
    all_summaries
)


combined_file = (
    OUTPUT_ROOT /
    "PBMMAE_AblationC_Remove_MAG_LD_outer_folds_training_summary.csv"
)


combined_df.to_csv(
    combined_file,
    index=False
)


print(
    "\n\n" +
    "=" * 120
)

print(
    "ABLATION C — REMOVE MAG_LD — TRAINING COMPLETED"
)

print(
    "=" * 120
)


print(
    combined_df[
        [
            "Fold",
            "Train_patches",
            "Remaining_channels",
            "Initial_loss",
            "Final_loss",
            "Minimum_loss",
            "Minimum_loss_epoch",
            "Final25_mean_loss",
            "Whole_modality_probability_empirical",
            "Trainable_parameters",
            "Training_seconds"
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
    "\nCombined summary:"
)

print(
    combined_file
)


print(
    "\nCONTROLLED ABLATION:"
)

print(
    "MAG_LD was removed from the 14-channel input."
)

print(
    "ID, GRAV_LD, and DEM_LD remain in the structural family."
)

print(
    "Physics-balanced reconstruction and whole-modality "
    "masking remain active."
)

print(
    "Validation data were not loaded during training."
)

print(
    "=" * 120
)