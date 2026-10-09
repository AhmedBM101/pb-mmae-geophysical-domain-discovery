import os
# ============================================================
# Paper 1 — PB-MMAE Ablation B
# REMOVE WHOLE-MODALITY MASKING
#
# Frozen primary configuration:
#   patch size = 13x13
#   six physics-family encoders
#   per-family LayerNorm
#   Linear(96 -> 64) fusion
#   spatial masking = 0.40
#   physics-balanced SmoothL1 reconstruction loss
#   AdamW lr = 1e-3
#   weight decay = 1e-4
#   batch size = 32
#   epochs = 300
#
# ONLY ABLATION:
#   whole-modality masking probability:
#       0.30 -> 0.00
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
from torch.utils.data import (
    Dataset,
    DataLoader
)

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
    "B_NoWholeModalityMasking_13x13"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. FROZEN CONFIGURATION
# ============================================================

FOLDS = [
    1,
    2,
    4,
    5
]

PATCH_SIZE = 13

BRANCH_EMBED_DIM = 16
FUSED_LATENT_DIM = 64

SPATIAL_MASK_RATIO = 0.40

# ------------------------------------------------------------
# ABLATION B:
# whole-family masking completely disabled
# ------------------------------------------------------------

WHOLE_MODALITY_PROB = 0.00

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
# 3. CHANNELS / PHYSICS FAMILIES
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


# ============================================================
# 4. REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# 5. DATASET
# ============================================================

class PatchDataset(Dataset):

    def __init__(self, array):

        if array.ndim != 4:

            raise ValueError(
                f"Expected 4-D array; got {array.shape}"
            )

        if array.shape[-1] != 14:

            raise ValueError(
                f"Expected 14 channels; got {array.shape[-1]}"
            )

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
# 8. FROZEN PB-MMAE ARCHITECTURE
# ============================================================

class PBMMAE(nn.Module):

    def __init__(self):

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
# 9. SPATIAL MASKING ONLY
#
# ABLATION B:
#   no entire physics family is ever masked.
# ============================================================

def create_masked_batch(x):

    batch_size = x.shape[0]
    height = x.shape[2]
    width = x.shape[3]

    masked_x = x.clone()

    family_masks = {}

    n_spatial_mask = int(
        round(
            height
            *
            width
            *
            SPATIAL_MASK_RATIO
        )
    )

    # This should remain zero for every epoch/run.
    whole_family_count = 0


    for family_name in FAMILY_NAMES:

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


    for sample_idx in range(
        batch_size
    ):

        for family_name in FAMILY_NAMES:

            indices = PHYSICS_FAMILIES[
                family_name
            ]

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
# 10. PHYSICS-BALANCED LOSS — UNCHANGED
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

            family_loss = torch.tensor(
                0.0,
                device=original_x.device
            )

        else:

            family_loss = masked_values.mean()

        family_losses[
            family_name
        ] = family_loss


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
        f"ABLATION B — FOLD {fold}"
    )

    print(
        "=" * 100
    )

    print(
        "Ablation:"
    )

    print(
        "Whole-modality masking disabled"
    )

    print(
        "Spatial mask ratio:",
        SPATIAL_MASK_RATIO
    )

    print(
        "Whole-modality probability:",
        WHOLE_MODALITY_PROB
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
        "TRAIN patches:",
        train_array.shape
    )

    if train_array.shape[1:4] != (
        13,
        13,
        14
    ):

        raise RuntimeError(
            f"Unexpected patch shape: {train_array.shape}"
        )


    train_dataset = PatchDataset(
        train_array
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
        "Epochs:",
        EPOCHS
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )


    history = []

    total_whole_events = 0
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
            for family_name
            in FAMILY_NAMES
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

            total_whole_events += (
                whole_count
            )

            total_samples_seen += (
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


            epoch_total += (
                total_loss.item()
                *
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


            epoch_samples += (
                batch_n
            )


        mean_total = (
            epoch_total
            /
            epoch_samples
        )


        row = {
            "Fold":
                fold,

            "Epoch":
                epoch,

            "Ablation":
                "NoWholeModalityMasking",

            "Total_loss":
                mean_total
        }


        for family_name in FAMILY_NAMES:

            row[
                f"{family_name}_loss"
            ] = (
                family_totals[
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
                f"Epoch {epoch:3d}/{EPOCHS}"
                f" | PB loss = {mean_total:.8f}"
            )


    elapsed_seconds = (
        time.time()
        -
        start_time
    )


    # ========================================================
    # 12. SAVE HISTORY
    # ========================================================

    history_df = pd.DataFrame(
        history
    )

    history_file = (
        fold_output /
        f"Fold{fold}_AblationB_NoWholeModalityMasking_history.csv"
    )

    history_df.to_csv(
        history_file,
        index=False
    )


    # ========================================================
    # 13. SAVE MODEL
    # ========================================================

    model_file = (
        fold_output /
        f"Fold{fold}_AblationB_NoWholeModalityMasking_FINAL.pt"
    )


    torch.save(
        {
            "fold":
                fold,

            "seed":
                seed,

            "ablation":
                "NoWholeModalityMasking",

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
    # 14. SUMMARY
    # ========================================================

    empirical_whole_probability = (
        total_whole_events
        /
        total_samples_seen
        if total_samples_seen > 0
        else np.nan
    )


    minimum_idx = (
        history_df[
            "Total_loss"
        ].idxmin()
    )


    summary = {

        "Fold":
            fold,

        "Ablation":
            "NoWholeModalityMasking",

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
                ]
                .tail(
                    25
                )
                .mean()
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
        f"Fold{fold}_AblationB_NoWholeModalityMasking_summary.csv"
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


    return summary


# ============================================================
# 15. RUN OUTER FOLDS
# ============================================================

all_summaries = []


for fold in FOLDS:

    summary = train_fold(
        fold
    )

    all_summaries.append(
        summary
    )


# ============================================================
# 16. COMBINED SUMMARY
# ============================================================

combined_df = pd.DataFrame(
    all_summaries
)


combined_file = (
    OUTPUT_ROOT /
    "PBMMAE_AblationB_NoWholeModalityMasking_outer_folds_training_summary.csv"
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
    "ABLATION B — OUTER-FOLD TRAINING COMPLETED"
)

print(
    "=" * 120
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
    "The only ablated component was whole-modality masking."
)

print(
    "Physics-balanced loss remains active."
)

print(
    "Spatial masking remains fixed at 0.40."
)

print(
    "Architecture, patch size, latent size, optimizer, "
    "training duration, and outer folds are unchanged."
)

print(
    "Validation data were not loaded during training."
)


print(
    "=" * 120
)