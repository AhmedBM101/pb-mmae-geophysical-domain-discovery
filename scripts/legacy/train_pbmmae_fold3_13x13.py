import os
# ============================================================
# Paper 1 — PB-MMAE representative training
# Fold 3, 13x13, 14 channels
#
# Physics-Balanced Multimodal Masked Autoencoder
#
# TRAIN only:
#   - spatial masking = 40%
#   - whole-modality masking probability = 30%
#   - equal six-family reconstruction weighting
#   - masked-position Smooth L1 loss
#
# No outer validation data used during optimization.
# ============================================================


# ============================================================
# 0. ENVIRONMENT
# ============================================================

import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

from pathlib import Path
import random
import time

# PyTorch first
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

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

PBMMAE_DIR = (
    PROJECT_ROOT /
    "03_outputs" /
    "models" /
    "PBMMAE"
)

RUN_DIR = (
    PBMMAE_DIR /
    "Fold3_13x13"
)

RUN_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRAIN_FILE = (
    PATCH_DIR /
    "13x13" /
    "Fold_3" /
    "Fold3_TRAIN_13x13_14ch.npy"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

PATCH_SIZE = 13
FOLD = 3

N_CHANNELS = 14
N_FAMILIES = 6

BRANCH_EMBED_DIM = 16
FUSION_INPUT_DIM = 96
FUSED_LATENT_DIM = 64

SPATIAL_MASK_RATIO = 0.40
MODALITY_MASK_PROB = 0.30

SMOOTH_L1_BETA = 1.0

EPOCHS = 300
BATCH_SIZE = 32

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

SEED = 42

CHECKPOINT_EVERY = 25

# CPU is deliberate because the earlier Jupyter/CUDA
# environment showed runtime instability.
DEVICE = torch.device("cpu")

torch.set_num_threads(2)


print("=" * 78)
print("PB-MMAE TRAINING — FOLD 3, 13x13")
print("=" * 78)

print("Device                     :", DEVICE)
print("Patch size                 :", PATCH_SIZE)
print("Fold                       :", FOLD)
print("Input channels             :", N_CHANNELS)
print("Physics families           :", N_FAMILIES)
print("Branch embedding           :", BRANCH_EMBED_DIM)
print("Fusion input               :", FUSION_INPUT_DIM)
print("Fused latent               :", FUSED_LATENT_DIM)
print("Spatial mask ratio         :", SPATIAL_MASK_RATIO)
print("Whole-modality probability :", MODALITY_MASK_PROB)
print("Smooth L1 beta             :", SMOOTH_L1_BETA)
print("Epochs                     :", EPOCHS)
print("Batch size                 :", BATCH_SIZE)
print("Learning rate              :", LEARNING_RATE)
print("Weight decay               :", WEIGHT_DECAY)
print("Seed                       :", SEED)


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
        CHANNEL_NAMES.index(channel)
        for channel in channels
    ]

    for family, channels
    in PHYSICS_FAMILIES.items()
}


FAMILY_NAMES = list(
    PHYSICS_FAMILIES.keys()
)

FAMILY_WEIGHT = (
    1.0 /
    len(FAMILY_NAMES)
)


# ------------------------------------------------------------
# Integrity assertions
# ------------------------------------------------------------

assert len(CHANNEL_NAMES) == N_CHANNELS

assert len(FAMILY_NAMES) == N_FAMILIES

assert (
    BRANCH_EMBED_DIM *
    N_FAMILIES
    ==
    FUSION_INPUT_DIM
)

all_family_channels = [

    channel

    for channels
    in PHYSICS_FAMILIES.values()

    for channel in channels
]

assert set(all_family_channels) == set(
    CHANNEL_NAMES
)

assert len(all_family_channels) == N_CHANNELS


# ============================================================
# 4. REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)


# ============================================================
# 5. FAMILY ENCODER
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


# ============================================================
# 6. FAMILY DECODER
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


# ============================================================
# 7. PB-MMAE MODEL
# ============================================================

class PBMMAE(nn.Module):

    def __init__(
        self
    ):

        super().__init__()

        # ----------------------------------------------------
        # Six physics-family encoders
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
        # Six physics-family decoders
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

        for family in FAMILY_NAMES:

            family_embedding = (
                self.encoders[
                    family
                ](
                    family_inputs[
                        family
                    ]
                )
            )

            branch_embeddings.append(
                family_embedding
            )

        concatenated = torch.cat(
            branch_embeddings,
            dim=1
        )

        fused_latent = self.fusion(
            concatenated
        )

        return (
            fused_latent,
            branch_embeddings
        )


    def decode(
        self,
        fused_latent,
        output_size
    ):

        outputs = {}

        for family in FAMILY_NAMES:

            outputs[
                family
            ] = self.decoders[
                family
            ](
                fused_latent,
                output_size
            )

        return outputs


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
            fused_latent,
            branch_embeddings
        ) = self.encode(
            family_inputs
        )

        reconstructions = self.decode(
            fused_latent,
            output_size
        )

        return (
            reconstructions,
            fused_latent,
            branch_embeddings
        )


# ============================================================
# 8. SPLIT 14-CHANNEL INPUT INTO PHYSICS FAMILIES
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
# 9. PB-MMAE MASK GENERATOR
# ============================================================

def apply_batch_masks(
    x,
    spatial_mask_ratio,
    modality_mask_prob
):

    """
    Parameters
    ----------
    x : torch.Tensor
        Shape (B, 14, H, W).

    Returns
    -------
    masked_x :
        Corrupted input.

    spatial_masks :
        Dictionary of Boolean masks.
        True indicates positions contributing to
        masked reconstruction loss.

    whole_family_masks :
        Dictionary showing which samples received
        complete modality masking.
    """

    B, C, H, W = x.shape

    masked_x = x.clone()

    n_positions = (
        H * W
    )

    n_spatial_mask = int(
        round(
            n_positions *
            spatial_mask_ratio
        )
    )


    # --------------------------------------------------------
    # Tracking dictionaries
    # --------------------------------------------------------

    spatial_masks = {

        family:
            torch.zeros(

                B,
                H,
                W,

                dtype=torch.bool,
                device=x.device
            )

        for family
        in FAMILY_NAMES
    }


    whole_family_masks = {

        family:
            torch.zeros(

                B,

                dtype=torch.bool,
                device=x.device
            )

        for family
        in FAMILY_NAMES
    }


    # --------------------------------------------------------
    # Sample-by-sample masking
    # --------------------------------------------------------

    for b in range(B):

        whole_family = None


        # ----------------------------------------------------
        # Whole-modality masking
        # ----------------------------------------------------

        if torch.rand(
            1,
            device=x.device
        ).item() < modality_mask_prob:

            selected_family_index = (
                torch.randint(

                    low=0,

                    high=N_FAMILIES,

                    size=(1,),

                    device=x.device

                ).item()
            )

            whole_family = FAMILY_NAMES[
                selected_family_index
            ]

            whole_family_masks[
                whole_family
            ][b] = True


        # ----------------------------------------------------
        # Family-specific spatial masking
        # ----------------------------------------------------

        for family in FAMILY_NAMES:

            channel_indices = (
                PHYSICS_FAMILY_INDICES[
                    family
                ]
            )


            # ------------------------------------------------
            # Complete family mask
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

            selected_positions = (
                torch.randperm(
                    n_positions,
                    device=x.device
                )[
                    :n_spatial_mask
                ]
            )

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


            # Same spatial mask for every channel
            # belonging to the family.

            for channel_index in channel_indices:

                masked_x[
                    b,
                    channel_index
                ][
                    family_mask
                ] = 0.0


    return (
        masked_x,
        spatial_masks,
        whole_family_masks
    )


# ============================================================
# 10. PHYSICS-BALANCED MASKED LOSS
# ============================================================

def physics_balanced_masked_loss(
    reconstructions,
    target_families,
    spatial_masks
):

    family_losses = {}

    total_loss = torch.tensor(
        0.0,
        device=DEVICE
    )


    for family in FAMILY_NAMES:

        prediction = (
            reconstructions[
                family
            ]
        )

        target = (
            target_families[
                family
            ]
        )

        mask = (
            spatial_masks[
                family
            ]
        )


        # ----------------------------------------------------
        # Expand spatial mask across family channels
        # ----------------------------------------------------

        expanded_mask = (

            mask
            .unsqueeze(1)
            .expand_as(
                prediction
            )
        )


        prediction_masked = (
            prediction[
                expanded_mask
            ]
        )

        target_masked = (
            target[
                expanded_mask
            ]
        )

        assert (
            prediction_masked.numel()
            > 0
        )


        family_loss = (
            F.smooth_l1_loss(

                prediction_masked,

                target_masked,

                beta=
                    SMOOTH_L1_BETA,

                reduction="mean"
            )
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
# 11. LOAD TRAIN DATA
# ============================================================

print("\nTRAIN file:")
print(TRAIN_FILE)

if not TRAIN_FILE.exists():

    raise FileNotFoundError(
        TRAIN_FILE
    )


X_train = np.load(
    TRAIN_FILE
).astype(
    np.float32
)

print(
    "\nOriginal TRAIN tensor:",
    X_train.shape
)


assert X_train.shape[1:] == (
    PATCH_SIZE,
    PATCH_SIZE,
    N_CHANNELS
)

assert np.isfinite(
    X_train
).all()


# ------------------------------------------------------------
# NHWC -> NCHW
# ------------------------------------------------------------

X_train = np.transpose(
    X_train,
    (0, 3, 1, 2)
)

X_train_tensor = torch.from_numpy(
    X_train
)

print(
    "PyTorch TRAIN tensor:",
    X_train_tensor.shape
)

print(
    "TRAIN finite:",
    torch.isfinite(
        X_train_tensor
    ).all().item()
)


# ============================================================
# 12. DATALOADER
# ============================================================

train_dataset = TensorDataset(
    X_train_tensor
)

generator = torch.Generator()

generator.manual_seed(
    SEED
)


train_loader = DataLoader(

    train_dataset,

    batch_size=
        BATCH_SIZE,

    shuffle=True,

    generator=
        generator,

    num_workers=0,

    pin_memory=False
)


print(
    "\nTRAIN samples:",
    len(
        train_dataset
    )
)

print(
    "Batches per epoch:",
    len(
        train_loader
    )
)


# ============================================================
# 13. INITIALIZE MODEL
# ============================================================

model = PBMMAE().to(
    DEVICE
)


total_parameters = sum(

    p.numel()

    for p
    in model.parameters()
)


trainable_parameters = sum(

    p.numel()

    for p
    in model.parameters()

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


# ============================================================
# 14. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=
        LEARNING_RATE,

    weight_decay=
        WEIGHT_DECAY
)


# ============================================================
# 15. TRAINING
# ============================================================

history_records = []

start_time = time.time()


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()


    total_epoch_loss = 0.0

    family_epoch_loss = {

        family: 0.0

        for family
        in FAMILY_NAMES
    }

    total_samples = 0

    total_whole_masks = 0


    # --------------------------------------------------------
    # Batches
    # --------------------------------------------------------

    for (batch_x,) in train_loader:

        batch_x = batch_x.to(
            DEVICE
        )


        # ----------------------------------------------------
        # Generate fresh stochastic masks each epoch/batch
        # ----------------------------------------------------

        (
            masked_x,
            spatial_masks,
            whole_family_masks
        ) = apply_batch_masks(

            batch_x,

            spatial_mask_ratio=
                SPATIAL_MASK_RATIO,

            modality_mask_prob=
                MODALITY_MASK_PROB
        )


        masked_family_inputs = (
            split_into_families(
                masked_x
            )
        )

        target_family_inputs = (
            split_into_families(
                batch_x
            )
        )


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )


        (
            reconstructions,
            fused_latent,
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


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        total_loss.backward()

        optimizer.step()


        # ----------------------------------------------------
        # Accumulate statistics
        # ----------------------------------------------------

        batch_n = batch_x.size(0)

        total_epoch_loss += (
            total_loss.item()
            * batch_n
        )

        for family in FAMILY_NAMES:

            family_epoch_loss[
                family
            ] += (

                family_losses[
                    family
                ].item()

                * batch_n
            )


        total_samples += (
            batch_n
        )


        total_whole_masks += sum(

            int(
                mask.sum().item()
            )

            for mask
            in whole_family_masks.values()
        )


    # ========================================================
    # Epoch means
    # ========================================================

    mean_total_loss = (
        total_epoch_loss /
        total_samples
    )


    mean_family_losses = {

        family:
            family_epoch_loss[
                family
            ]
            /
            total_samples

        for family
        in FAMILY_NAMES
    }


    empirical_whole_mask_fraction = (

        total_whole_masks /
        total_samples
    )


    # --------------------------------------------------------
    # Save epoch record
    # --------------------------------------------------------

    epoch_record = {

        "Epoch":
            epoch,

        "Total_PB_Loss":
            mean_total_loss,

        "Whole_modality_fraction":
            empirical_whole_mask_fraction
    }


    for family in FAMILY_NAMES:

        epoch_record[
            f"{family}_loss"
        ] = (
            mean_family_losses[
                family
            ]
        )


    history_records.append(
        epoch_record
    )


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        epoch == 1
        or epoch % 25 == 0
        or epoch == EPOCHS
    ):

        print(
            f"\nEpoch "
            f"{epoch:3d}/{EPOCHS}"
        )

        print(
            f"  Total PB loss : "
            f"{mean_total_loss:.6f}"
        )

        print(
            f"  Whole mask p  : "
            f"{empirical_whole_mask_fraction:.3f}"
        )

        for family in FAMILY_NAMES:

            print(
                f"  {family:12s}: "
                f"{mean_family_losses[family]:.6f}"
            )


    # --------------------------------------------------------
    # Checkpoint every 25 epochs
    # --------------------------------------------------------

    if (
        epoch %
        CHECKPOINT_EVERY
        == 0
    ):

        checkpoint_file = (

            RUN_DIR /
            f"PBMMAE_Fold3_13x13_"
            f"epoch{epoch:03d}.pt"
        )


        torch.save(

            {

                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "history":
                    history_records,

                "patch_size":
                    PATCH_SIZE,

                "fold":
                    FOLD,

                "branch_embed_dim":
                    BRANCH_EMBED_DIM,

                "fused_latent_dim":
                    FUSED_LATENT_DIM,

                "spatial_mask_ratio":
                    SPATIAL_MASK_RATIO,

                "modality_mask_prob":
                    MODALITY_MASK_PROB,

                "smooth_l1_beta":
                    SMOOTH_L1_BETA,

                "seed":
                    SEED
            },

            checkpoint_file
        )


# ============================================================
# 16. TRAINING COMPLETE
# ============================================================

training_seconds = (
    time.time()
    -
    start_time
)


history_df = pd.DataFrame(
    history_records
)


# ============================================================
# 17. SAVE TRAINING HISTORY
# ============================================================

history_file = (

    RUN_DIR /
    "PBMMAE_Fold3_13x13_training_history.csv"
)


history_df.to_csv(
    history_file,
    index=False
)


# ============================================================
# 18. SAVE FINAL MODEL
# ============================================================

final_model_file = (

    RUN_DIR /
    "PBMMAE_Fold3_13x13_FINAL.pt"
)


torch.save(

    {

        "model_state_dict":
            model.state_dict(),

        "patch_size":
            PATCH_SIZE,

        "fold":
            FOLD,

        "n_channels":
            N_CHANNELS,

        "branch_embed_dim":
            BRANCH_EMBED_DIM,

        "fusion_input_dim":
            FUSION_INPUT_DIM,

        "fused_latent_dim":
            FUSED_LATENT_DIM,

        "spatial_mask_ratio":
            SPATIAL_MASK_RATIO,

        "modality_mask_prob":
            MODALITY_MASK_PROB,

        "smooth_l1_beta":
            SMOOTH_L1_BETA,

        "epochs":
            EPOCHS,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "seed":
            SEED,

        "final_total_pb_loss":
            history_df[
                "Total_PB_Loss"
            ].iloc[-1],

        "training_seconds":
            training_seconds
    },

    final_model_file
)


# ============================================================
# 19. SAVE SUMMARY
# ============================================================

summary_file = (

    RUN_DIR /
    "PBMMAE_Fold3_13x13_training_summary.txt"
)


with open(
    summary_file,
    "w"
) as f:

    f.write(
        "Paper 1 PB-MMAE representative training\n"
    )

    f.write(
        "Fold: 3\n"
    )

    f.write(
        "Patch size: 13x13\n"
    )

    f.write(
        "Channels: 14\n"
    )

    f.write(
        "Physics families: 6\n"
    )

    f.write(
        "Branch embedding dimension: 16\n"
    )

    f.write(
        "Fused latent dimension: 64\n"
    )

    f.write(
        "Spatial masking ratio: 0.40\n"
    )

    f.write(
        "Whole-modality probability: 0.30\n"
    )

    f.write(
        "Smooth L1 beta: 1.0\n"
    )

    f.write(
        f"Epochs: {EPOCHS}\n"
    )

    f.write(
        f"Final PB loss: "
        f"{history_df['Total_PB_Loss'].iloc[-1]:.8f}\n"
    )

    f.write(
        f"Training seconds: "
        f"{training_seconds:.2f}\n"
    )


# ============================================================
# 20. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 78)

print(
    "PB-MMAE TRAINING COMPLETED"
)

print("=" * 78)

print(
    "\nTraining time:",
    round(
        training_seconds,
        2
    ),
    "seconds"
)

print(
    "\nInitial total PB loss:",
    round(
        history_df[
            "Total_PB_Loss"
        ].iloc[0],
        6
    )
)

print(
    "Final total PB loss:",
    round(
        history_df[
            "Total_PB_Loss"
        ].iloc[-1],
        6
    )
)


print(
    "\nFinal family losses:"
)

for family in FAMILY_NAMES:

    print(
        f"  {family:12s}: "
        f"{history_df[f'{family}_loss'].iloc[-1]:.6f}"
    )


print(
    "\nMean empirical whole-modality "
    "mask probability:"
)

print(
    round(
        history_df[
            "Whole_modality_fraction"
        ].mean(),
        4
    )
)


print("\nFinal model:")
print(final_model_file)

print("\nTraining history:")
print(history_file)

print("\nSummary:")
print(summary_file)

print("=" * 78)