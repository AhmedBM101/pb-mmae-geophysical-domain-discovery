import os
# ============================================================
# Paper 1 — PB-MMAE 9x9 patch-size sensitivity
#
# Frozen architecture from the primary 13x13 analysis:
#
#   Six physics-family encoders
#   -> per-family LayerNorm
#   -> concatenate 6 x 16 = 96
#   -> single Linear(96, 64) fusion
#   -> family-specific decoders
#
# Frozen training protocol:
#   SmoothL1 beta = 1.0
#   equal physics-family loss weights = 1/6
#   spatial masking = 0.40
#   whole-modality masking probability = 0.30
#   maximum one whole family masked per sample
#   AdamW lr = 1e-3
#   weight decay = 1e-4
#   batch size = 32
#   epochs = 300
#
# Confirmatory outer folds:
#   1, 2, 4, 5
#
# IMPORTANT:
#   Validation patches are NOT loaded during training.
# ============================================================

import os

# Thread limits set before numerical libraries
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

from pathlib import Path
import time
import random
import json

import numpy as np

# Import torch before other heavy numerical operations
import torch
import torch.nn as nn
from torch.utils.data import (
    Dataset,
    DataLoader
)

import pandas as pd


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(os.environ["PBMMAE_PROJECT_ROOT"]).expanduser().resolve()

PATCH_ROOT = (
    PROJECT_ROOT /
    "03_outputs" /
    "patches" /
    "9x9"
)

OUTPUT_ROOT = (
    PROJECT_ROOT /
    "03_outputs" /
    "models" /
    "PBMMAE" /
    "OuterFolds_9x9_LinearFusion64_Sensitivity"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


FOLDS = [
    1,
    2,
    4,
    5
]


# ============================================================
# 2. LOCKED CHANNEL ORDER
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


# ============================================================
# 3. LOCKED PHYSICS FAMILIES
# ============================================================

PHYSICS_FAMILIES = {

    "magnetic": [
        0
    ],

    "gravity": [
        1,
        2,
        3
    ],

    "radiometric": [
        4,
        5,
        6
    ],

    "thermal": [
        7
    ],

    "structural": [
        8,
        9,
        10,
        11
    ],

    "terrain": [
        12,
        13
    ]
}


FAMILY_NAMES = list(
    PHYSICS_FAMILIES.keys()
)


N_FAMILIES = len(
    FAMILY_NAMES
)


# ============================================================
# 4. FROZEN HYPERPARAMETERS
# ============================================================

PATCH_SIZE = 9

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

DEVICE = torch.device(
    "cpu"
)

torch.set_num_threads(
    2
)


# ============================================================
# 5. REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# 6. DATASET
# ============================================================

class PatchDataset(
    Dataset
):

    def __init__(
        self,
        array
    ):

        # Input saved as:
        # (N, H, W, C)

        if array.ndim != 4:

            raise ValueError(
                f"Expected 4-D patch array, got {array.shape}"
            )

        if array.shape[-1] != 14:

            raise ValueError(
                f"Expected 14 channels, got {array.shape[-1]}"
            )

        # Convert to:
        # (N, C, H, W)

        self.data = torch.from_numpy(

            np.asarray(
                array,
                dtype=np.float32
            )

        ).permute(
            0,
            3,
            1,
            2
        ).contiguous()


    def __len__(
        self
    ):

        return self.data.shape[
            0
        ]


    def __getitem__(
        self,
        idx
    ):

        return self.data[
            idx
        ]


# ============================================================
# 7. FAMILY ENCODER
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
                (
                    3,
                    3
                )
            )
        )


        self.fc = nn.Linear(
            32 * 3 * 3,
            embed_dim
        )


        # Frozen controlled architecture:
        # LayerNorm applied separately to every family embedding

        self.norm = nn.LayerNorm(
            embed_dim
        )


    def forward(
        self,
        x
    ):

        z = self.conv(
            x
        )

        z = z.flatten(
            1
        )

        z = self.fc(
            z
        )

        z = self.norm(
            z
        )

        return z


# ============================================================
# 8. FAMILY DECODER
# ============================================================

class FamilyDecoder(
    nn.Module
):

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


    def forward(
        self,
        z
    ):

        x = self.fc(
            z
        )

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


        return self.conv(
            x
        )


# ============================================================
# 9. PB-MMAE — FROZEN LINEAR FUSION
# ============================================================

class PBMMAE(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()


        self.encoders = nn.ModuleDict()

        self.decoders = nn.ModuleDict()


        for family_name in FAMILY_NAMES:

            indices = PHYSICS_FAMILIES[
                family_name
            ]

            n_channels = len(
                indices
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
            N_FAMILIES
            *
            BRANCH_EMBED_DIM
        )


        # Frozen architecture:
        # no nonlinear hidden fusion layer

        self.fusion = nn.Linear(

            fusion_input_dim,

            FUSED_LATENT_DIM
        )


    def encode(
        self,
        x
    ):

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


    def forward(
        self,
        x
    ):

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
# 10. MASK GENERATION
# ============================================================

def create_masked_batch(
    x
):

    batch_size = x.shape[
        0
    ]

    height = x.shape[
        2
    ]

    width = x.shape[
        3
    ]


    masked_x = x.clone()


    family_masks = {}


    # Exact number of spatial cells
    # 9x9:
    # round(81 * 0.40) = 32

    n_spatial_mask = int(
        round(
            height
            *
            width
            *
            SPATIAL_MASK_RATIO
        )
    )


    whole_family_count = 0


    for sample_idx in range(
        batch_size
    ):

        # ----------------------------------------------------
        # At most one entire family is masked.
        # ----------------------------------------------------

        whole_family = None


        if random.random() < WHOLE_MODALITY_PROB:

            whole_family = random.choice(
                FAMILY_NAMES
            )

            whole_family_count += 1


        for family_name in FAMILY_NAMES:

            indices = PHYSICS_FAMILIES[
                family_name
            ]


            if family_name not in family_masks:

                family_masks[
                    family_name
                ] = torch.zeros(

                    (
                        batch_size,
                        1,
                        height,
                        width
                    ),

                    dtype=torch.bool,

                    device=x.device
                )


            # ------------------------------------------------
            # Whole-family mask
            # ------------------------------------------------

            if (
                whole_family
                ==
                family_name
            ):

                family_masks[
                    family_name
                ][
                    sample_idx,
                    0,
                    :,
                    :
                ] = True


                masked_x[
                    sample_idx,
                    indices,
                    :,
                    :
                ] = 0.0


            # ------------------------------------------------
            # Spatial mask
            # ------------------------------------------------

            else:

                flat_indices = torch.randperm(

                    height
                    *
                    width,

                    device=x.device

                )[
                    :
                    n_spatial_mask
                ]


                mask_flat = torch.zeros(

                    height
                    *
                    width,

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


                # Same spatial mask for all channels
                # inside that physics family

                for channel_index in indices:

                    masked_x[
                        sample_idx,
                        channel_index,
                        spatial_mask
                    ] = 0.0


    return (
        masked_x,
        family_masks,
        whole_family_count
    )


# ============================================================
# 11. PHYSICS-BALANCED MASKED LOSS
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


        # Mask shape:
        # B x 1 x H x W
        #
        # Broadcast across family channels.

        mask = family_masks[
            family_name
        ].expand_as(
            loss_map
        )


        masked_values = loss_map[
            mask
        ]


        if masked_values.numel() == 0:

            family_loss = torch.tensor(

                0.0,

                device=original_x.device
            )

        else:

            # Equalize channel count within family:
            # mean reconstruction error of that family.

            family_loss = masked_values.mean()


        family_losses[
            family_name
        ] = family_loss


    # Equal weight for all six physics families

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
# 12. TRAIN ONE OUTER FOLD
# ============================================================

def train_fold(
    fold
):

    seed = (
        BASE_SEED
        +
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
        f"Fold{fold}_TRAIN_9x9_14ch.npy"
    )


    print(
        "\n" +
        "=" * 100
    )

    print(
        f"FOLD {fold} — 9x9 SENSITIVITY"
    )

    print(
        "=" * 100
    )


    print(
        "Seed:",
        seed
    )

    print(
        "TRAIN file:",
        train_file
    )


    if not train_file.exists():

        raise FileNotFoundError(
            train_file
        )


    # --------------------------------------------------------
    # TRAIN ONLY.
    #
    # Validation data are deliberately not loaded here.
    # --------------------------------------------------------

    train_array = np.load(
        train_file
    )


    print(
        "TRAIN patches:",
        train_array.shape
    )


    if train_array.shape[
        1:
        4
    ] != (
        9,
        9,
        14
    ):

        raise RuntimeError(
            f"Unexpected TRAIN patch shape: "
            f"{train_array.shape}"
        )


    train_dataset = PatchDataset(
        train_array
    )


    generator = torch.Generator()

    generator.manual_seed(
        seed
    )


    train_loader = DataLoader(

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
        "Device:",
        DEVICE
    )

    print(
        "Epochs:",
        EPOCHS
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Spatial mask ratio:",
        SPATIAL_MASK_RATIO
    )

    print(
        "9x9 masked cells per non-whole family:",
        round(
            PATCH_SIZE
            *
            PATCH_SIZE
            *
            SPATIAL_MASK_RATIO
        )
    )

    print(
        "Whole-modality probability:",
        WHOLE_MODALITY_PROB
    )


    history = []

    start_time = time.time()

    total_whole_family_events = 0

    total_training_samples = 0


    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()


        epoch_total_loss = 0.0

        epoch_family_loss = {

            family_name:
                0.0

            for family_name
            in FAMILY_NAMES
        }


        epoch_samples = 0


        for batch in train_loader:

            batch = batch.to(
                DEVICE
            )


            batch_n = batch.shape[
                0
            ]


            masked_batch, masks, whole_count = (
                create_masked_batch(
                    batch
                )
            )


            total_whole_family_events += (
                whole_count
            )

            total_training_samples += (
                batch_n
            )


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


            epoch_total_loss += (
                total_loss.item()
                *
                batch_n
            )


            for family_name in FAMILY_NAMES:

                epoch_family_loss[
                    family_name
                ] += (

                    family_losses[
                        family_name
                    ].item()

                    *
                    batch_n
                )


            epoch_samples += (
                batch_n
            )


        mean_total_loss = (
            epoch_total_loss
            /
            epoch_samples
        )


        row = {

            "Fold":
                fold,

            "Patch_size":
                PATCH_SIZE,

            "Epoch":
                epoch,

            "Total_loss":
                mean_total_loss
        }


        for family_name in FAMILY_NAMES:

            row[
                f"{family_name}_loss"
            ] = (

                epoch_family_loss[
                    family_name
                ]
                /
                epoch_samples
            )


        history.append(
            row
        )


        if (
            epoch == 1
            or
            epoch % 25 == 0
            or
            epoch == EPOCHS
        ):

            print(
                f"Epoch {epoch:3d}/{EPOCHS} "
                f"| loss = {mean_total_loss:.8f}"
            )


    elapsed_seconds = (
        time.time()
        -
        start_time
    )


    # ========================================================
    # 13. TRAINING HISTORY
    # ========================================================

    history_df = pd.DataFrame(
        history
    )


    history_file = (
        fold_output /
        f"Fold{fold}_9x9_PBMMAE_training_history.csv"
    )


    history_df.to_csv(
        history_file,
        index=False
    )


    # ========================================================
    # 14. MODEL CHECKPOINT
    # ========================================================

    model_file = (
        fold_output /
        f"Fold{fold}_9x9_PBMMAE_LinearFusion64_FINAL.pt"
    )


    torch.save(

        {
            "fold":
                fold,

            "seed":
                seed,

            "patch_size":
                PATCH_SIZE,

            "channels":
                CHANNELS,

            "physics_families":
                PHYSICS_FAMILIES,

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
    # 15. SUMMARY
    # ========================================================

    final_loss = float(
        history_df.iloc[
            -1
        ][
            "Total_loss"
        ]
    )


    minimum_loss = float(
        history_df[
            "Total_loss"
        ].min()
    )


    minimum_epoch = int(

        history_df.loc[

            history_df[
                "Total_loss"
            ].idxmin(),

            "Epoch"
        ]
    )


    final25_mean = float(

        history_df[
            "Total_loss"
        ]
        .tail(
            25
        )
        .mean()
    )


    empirical_whole_probability = float(

        total_whole_family_events
        /
        total_training_samples
    )


    summary = {

        "Fold":
            fold,

        "Analysis_role":
            "9x9_patch_size_sensitivity",

        "Architecture":
            "LayerNorm_LinearFusion64",

        "Seed":
            seed,

        "Patch_size":
            PATCH_SIZE,

        "Train_patches":
            len(
                train_dataset
            ),

        "Branch_embed_dim":
            BRANCH_EMBED_DIM,

        "Fused_latent_dim":
            FUSED_LATENT_DIM,

        "Spatial_mask_ratio":
            SPATIAL_MASK_RATIO,

        "Spatial_mask_cells":
            int(
                round(
                    PATCH_SIZE
                    *
                    PATCH_SIZE
                    *
                    SPATIAL_MASK_RATIO
                )
            ),

        "Whole_modality_probability_target":
            WHOLE_MODALITY_PROB,

        "Whole_modality_probability_empirical":
            empirical_whole_probability,

        "SmoothL1_beta":
            SMOOTHL1_BETA,

        "Learning_rate":
            LEARNING_RATE,

        "Weight_decay":
            WEIGHT_DECAY,

        "Batch_size":
            BATCH_SIZE,

        "Epochs":
            EPOCHS,

        "Initial_loss":
            float(
                history_df.iloc[
                    0
                ][
                    "Total_loss"
                ]
            ),

        "Final_loss":
            final_loss,

        "Minimum_loss":
            minimum_loss,

        "Minimum_loss_epoch":
            minimum_epoch,

        "Final25_mean_loss":
            final25_mean,

        "Training_seconds":
            elapsed_seconds,

        "Trainable_parameters":
            parameter_count
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
        [
            summary
        ]
    )


    summary_file = (
        fold_output /
        f"Fold{fold}_9x9_PBMMAE_training_summary.csv"
    )


    summary_df.to_csv(
        summary_file,
        index=False
    )


    print(
        "\nFold completed."
    )

    print(
        "Final loss:",
        round(
            final_loss,
            8
        )
    )

    print(
        "Minimum loss:",
        round(
            minimum_loss,
            8
        ),
        "at epoch",
        minimum_epoch
    )

    print(
        "Final-25 mean:",
        round(
            final25_mean,
            8
        )
    )

    print(
        "Empirical whole-modality probability:",
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

    print(
        "Model:",
        model_file
    )

    print(
        "Summary:",
        summary_file
    )


    return summary


# ============================================================
# 16. TRAIN ALL FROZEN OUTER FOLDS
# ============================================================

all_summaries = []


for fold in FOLDS:

    result = train_fold(
        fold
    )

    all_summaries.append(
        result
    )


# ============================================================
# 17. COMBINED OUTER-FOLD SUMMARY
# ============================================================

combined_df = pd.DataFrame(
    all_summaries
)


combined_file = (
    OUTPUT_ROOT /
    "PBMMAE_9x9_outer_folds_training_summary.csv"
)


combined_df.to_csv(
    combined_file,
    index=False
)


print(
    "\n\n" +
    "=" * 110
)

print(
    "9x9 PB-MMAE OUTER-FOLD SENSITIVITY TRAINING COMPLETED"
)

print(
    "=" * 110
)


print(

    combined_df[
        [
            "Fold",
            "Train_patches",
            "Initial_loss",
            "Final_loss",
            "Minimum_loss",
            "Minimum_loss_epoch",
            "Final25_mean_loss",
            "Whole_modality_probability_empirical",
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
    "\nIMPORTANT:"
)

print(
    "No validation patches were loaded during training."
)

print(
    "The architecture and hyperparameters were frozen from "
    "the primary 13x13 PB-MMAE analysis."
)


print(
    "=" * 110
)