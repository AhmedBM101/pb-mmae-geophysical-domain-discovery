import os
# ============================================================
# Paper 1 — Frozen PB-MMAE LinearFusion64 outer-fold training
#
# Evaluation folds:
#   1, 2, 4, 5
#
# Fold 3 is EXCLUDED because it was used as the development
# fold for architecture selection.
#
# Frozen architecture:
#   - 6 physics-family branches
#   - branch embedding = 16
#   - per-family LayerNorm
#   - single linear fusion: 96 -> 64
#   - fused latent = 64
#
# Frozen SSL settings:
#   - spatial masking = 0.40
#   - whole-modality masking = 0.30
#   - Smooth L1 beta = 1.0
#   - AdamW
#   - lr = 1e-3
#   - weight decay = 1e-4
#   - batch size = 32
#   - epochs = 300
#
# Each fold is trained independently from scratch using
# TRAIN patches only.
#
# Validation patches are NEVER loaded in this script.
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

MODEL_ROOT.mkdir(
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

FUSION_INPUT_DIM = (
    N_FAMILIES *
    BRANCH_EMBED_DIM
)

FUSED_LATENT_DIM = 64

USE_BRANCH_LAYERNORM = True

FUSION_TYPE = (
    "single_linear_projection"
)

SPATIAL_MASK_RATIO = 0.40

MODALITY_MASK_PROB = 0.30

SMOOTH_L1_BETA = 1.0

EPOCHS = 300

BATCH_SIZE = 32

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

BASE_SEED = 42

CHECKPOINT_EVERY = 25

DEVICE = torch.device(
    "cpu"
)

torch.set_num_threads(
    2
)


# ============================================================
# 3. CHANNEL DEFINITIONS
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

FAMILY_WEIGHT = (
    1.0 /
    len(
        FAMILY_NAMES
    )
)


# ============================================================
# 4. CONFIGURATION CHECKS
# ============================================================

assert len(
    CHANNEL_NAMES
) == N_CHANNELS

assert len(
    FAMILY_NAMES
) == N_FAMILIES

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

    for channel
    in channels
]


assert len(
    all_family_channels
) == N_CHANNELS

assert set(
    all_family_channels
) == set(
    CHANNEL_NAMES
)


# ============================================================
# 5. REPRODUCIBILITY
# ============================================================

def set_seed(
    seed
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )


# ============================================================
# 6. FAMILY ENCODER
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


# ============================================================
# 7. FAMILY DECODER
# ============================================================

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


# ============================================================
# 8. FROZEN PB-MMAE ARCHITECTURE
# ============================================================

class PBMMAE_Linear64(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()


        # ----------------------------------------------------
        # Physics-family encoders
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
        # Frozen linear fusion
        #
        # 96 -> 64
        # ----------------------------------------------------

        self.fusion = nn.Linear(

            FUSION_INPUT_DIM,

            FUSED_LATENT_DIM
        )


        # ----------------------------------------------------
        # Family decoders
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


    def decode(
        self,
        z_fused,
        output_size
    ):

        outputs = {}


        for family in FAMILY_NAMES:

            outputs[
                family
            ] = self.decoders[
                family
            ](
                z_fused,
                output_size
            )


        return outputs


    def forward(
        self,
        family_inputs
    ):

        first_family = (
            FAMILY_NAMES[0]
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
            raw_embeddings,
            normalized_embeddings
        ) = self.encode(
            family_inputs
        )


        reconstructions = (
            self.decode(
                z_fused,
                output_size
            )
        )


        return (
            reconstructions,
            z_fused,
            raw_embeddings,
            normalized_embeddings
        )


# ============================================================
# 9. FAMILY SPLITTER
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
# 10. MASK GENERATOR
# ============================================================

def apply_batch_masks(
    x,
    spatial_mask_ratio,
    modality_mask_prob
):

    B, C, H, W = (
        x.shape
    )


    masked_x = (
        x.clone()
    )


    n_positions = (
        H * W
    )


    n_spatial_mask = int(

        round(
            n_positions *
            spatial_mask_ratio
        )
    )


    spatial_masks = {

        family:
            torch.zeros(

                B,
                H,
                W,

                dtype=
                    torch.bool,

                device=
                    x.device
            )

        for family
        in FAMILY_NAMES
    }


    whole_family_masks = {

        family:
            torch.zeros(

                B,

                dtype=
                    torch.bool,

                device=
                    x.device
            )

        for family
        in FAMILY_NAMES
    }


    for b in range(
        B
    ):

        whole_family = None


        # ----------------------------------------------------
        # Optional whole-family mask
        # ----------------------------------------------------

        if torch.rand(
            1,
            device=x.device
        ).item() < modality_mask_prob:


            selected_family_index = (
                torch.randint(

                    low=0,

                    high=
                        N_FAMILIES,

                    size=(1,),

                    device=
                        x.device

                ).item()
            )


            whole_family = (
                FAMILY_NAMES[
                    selected_family_index
                ]
            )


            whole_family_masks[
                whole_family
            ][b] = True


        # ----------------------------------------------------
        # Spatial masks
        # ----------------------------------------------------

        for family in FAMILY_NAMES:

            channel_indices = (
                PHYSICS_FAMILY_INDICES[
                    family
                ]
            )


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


            selected_positions = (
                torch.randperm(

                    n_positions,

                    device=
                        x.device

                )[
                    :n_spatial_mask
                ]
            )


            family_mask = torch.zeros(

                n_positions,

                dtype=
                    torch.bool,

                device=
                    x.device
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
            ][b] = (
                family_mask
            )


            for channel_index in (
                channel_indices
            ):

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
# 11. PHYSICS-BALANCED MASKED LOSS
# ============================================================

def physics_balanced_masked_loss(
    reconstructions,
    target_families,
    spatial_masks
):

    family_losses = {}


    total_loss = torch.tensor(

        0.0,

        device=
            DEVICE
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
            >
            0
        )


        family_loss = (
            F.smooth_l1_loss(

                prediction_masked,

                target_masked,

                beta=
                    SMOOTH_L1_BETA,

                reduction=
                    "mean"
            )
        )


        family_losses[
            family
        ] = (
            family_loss
        )


        total_loss = (
            total_loss
            +
            FAMILY_WEIGHT
            *
            family_loss
        )


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

    print(
        "\n\n" +
        "#" * 90
    )

    print(
        f"OUTER EVALUATION FOLD {fold}"
    )

    print(
        "#" * 90
    )


    # --------------------------------------------------------
    # Independent deterministic seed per fold
    #
    # This avoids accidentally sharing an identical RNG stream
    # across independent outer-fold models while remaining
    # exactly reproducible.
    # --------------------------------------------------------

    fold_seed = (
        BASE_SEED +
        fold
    )


    set_seed(
        fold_seed
    )


    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    train_file = (

        PATCH_DIR /
        "13x13" /
        f"Fold_{fold}" /
        f"Fold{fold}_TRAIN_13x13_14ch.npy"
    )


    fold_dir = (

        MODEL_ROOT /
        f"Fold{fold}"
    )


    fold_dir.mkdir(

        parents=True,

        exist_ok=True
    )


    print(
        "\nTRAIN file:"
    )

    print(
        train_file
    )


    if not train_file.exists():

        raise FileNotFoundError(
            train_file
        )


    # --------------------------------------------------------
    # Load TRAIN only
    # --------------------------------------------------------

    X_train = np.load(
        train_file
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


    X_train_tensor = (
        torch.from_numpy(
            X_train
        )
    )


    print(
        "PyTorch TRAIN tensor:",
        X_train_tensor.shape
    )


    # --------------------------------------------------------
    # Dataset / loader
    # --------------------------------------------------------

    train_dataset = (
        TensorDataset(
            X_train_tensor
        )
    )


    generator = (
        torch.Generator()
    )


    generator.manual_seed(
        fold_seed
    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=
            BATCH_SIZE,

        shuffle=
            True,

        generator=
            generator,

        num_workers=
            0,

        pin_memory=
            False
    )


    print(
        "TRAIN samples:",
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


    # --------------------------------------------------------
    # Fresh model
    # --------------------------------------------------------

    model = (
        PBMMAE_Linear64()
        .to(
            DEVICE
        )
    )


    n_parameters = sum(

        p.numel()

        for p
        in model.parameters()
    )


    print(
        "Model parameters:",
        f"{n_parameters:,}"
    )


    optimizer = (
        torch.optim.AdamW(

            model.parameters(),

            lr=
                LEARNING_RATE,

            weight_decay=
                WEIGHT_DECAY
        )
    )


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history_records = []


    start_time = (
        time.time()
    )


    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()


        total_epoch_loss = (
            0.0
        )


        family_epoch_loss = {

            family:
                0.0

            for family
            in FAMILY_NAMES
        }


        total_samples = (
            0
        )


        total_whole_masks = (
            0
        )


        for (
            batch_x,
        ) in train_loader:


            batch_x = (
                batch_x.to(
                    DEVICE
                )
            )


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


            optimizer.zero_grad(
                set_to_none=True
            )


            (
                reconstructions,
                z_fused,
                raw_embeddings,
                normalized_embeddings
            ) = model(
                masked_family_inputs
            )


            (
                total_loss,
                family_losses
            ) = physics_balanced_masked_loss(

                reconstructions,

                target_family_inputs,

                spatial_masks
            )


            total_loss.backward()


            optimizer.step()


            batch_n = (
                batch_x.size(
                    0
                )
            )


            total_epoch_loss += (
                total_loss.item()
                *
                batch_n
            )


            for family in FAMILY_NAMES:

                family_epoch_loss[
                    family
                ] += (

                    family_losses[
                        family
                    ].item()

                    *
                    batch_n
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


        # ----------------------------------------------------
        # Epoch means
        # ----------------------------------------------------

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


        record = {

            "Fold":
                fold,

            "Epoch":
                epoch,

            "Seed":
                fold_seed,

            "Total_PB_Loss":
                mean_total_loss,

            "Whole_modality_fraction":
                empirical_whole_mask_fraction
        }


        for family in FAMILY_NAMES:

            record[
                f"{family}_loss"
            ] = (
                mean_family_losses[
                    family
                ]
            )


        history_records.append(
            record
        )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            epoch == 1
            or
            epoch % 25 == 0
            or
            epoch == EPOCHS
        ):

            print(
                f"\nFold {fold} | "
                f"Epoch {epoch:3d}/{EPOCHS}"
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


        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        if (
            epoch %
            CHECKPOINT_EVERY
            == 0
        ):

            checkpoint_file = (

                fold_dir /
                f"PBMMAE_LINEAR64_"
                f"Fold{fold}_13x13_"
                f"epoch{epoch:03d}.pt"
            )


            torch.save(

                {

                    "fold":
                        fold,

                    "epoch":
                        epoch,

                    "seed":
                        fold_seed,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "architecture_frozen":
                        True,

                    "development_fold":
                        3,

                    "branch_embed_dim":
                        BRANCH_EMBED_DIM,

                    "branch_layernorm":
                        True,

                    "fusion_type":
                        FUSION_TYPE,

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
                        WEIGHT_DECAY

                },

                checkpoint_file
            )


    # --------------------------------------------------------
    # Complete fold
    # --------------------------------------------------------

    training_seconds = (

        time.time()
        -
        start_time
    )


    history_df = (
        pd.DataFrame(
            history_records
        )
    )


    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_file = (

        fold_dir /
        f"PBMMAE_LINEAR64_"
        f"Fold{fold}_13x13_"
        f"training_history.csv"
    )


    history_df.to_csv(

        history_file,

        index=False
    )


    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    final_model_file = (

        fold_dir /
        f"PBMMAE_LINEAR64_"
        f"Fold{fold}_13x13_FINAL.pt"
    )


    torch.save(

        {

            "fold":
                fold,

            "seed":
                fold_seed,

            "model_state_dict":
                model.state_dict(),

            "architecture_frozen":
                True,

            "development_fold":
                3,

            "patch_size":
                PATCH_SIZE,

            "n_channels":
                N_CHANNELS,

            "physics_families":
                N_FAMILIES,

            "branch_embed_dim":
                BRANCH_EMBED_DIM,

            "branch_layernorm":
                True,

            "fusion_type":
                FUSION_TYPE,

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

            "final_total_pb_loss":
                float(
                    history_df[
                        "Total_PB_Loss"
                    ].iloc[-1]
                ),

            "minimum_total_pb_loss":
                float(
                    history_df[
                        "Total_PB_Loss"
                    ].min()
                ),

            "minimum_loss_epoch":
                int(
                    history_df.loc[
                        history_df[
                            "Total_PB_Loss"
                        ].idxmin(),
                        "Epoch"
                    ]
                ),

            "training_seconds":
                training_seconds

        },

        final_model_file
    )


    # --------------------------------------------------------
    # Fold summary
    # --------------------------------------------------------

    summary_file = (

        fold_dir /
        f"PBMMAE_LINEAR64_"
        f"Fold{fold}_13x13_"
        f"training_summary.txt"
    )


    final_loss = float(

        history_df[
            "Total_PB_Loss"
        ].iloc[-1]
    )


    minimum_loss = float(

        history_df[
            "Total_PB_Loss"
        ].min()
    )


    minimum_epoch = int(

        history_df.loc[
            history_df[
                "Total_PB_Loss"
            ].idxmin(),
            "Epoch"
        ]
    )


    final25_mean = float(

        history_df[
            "Total_PB_Loss"
        ].tail(
            25
        ).mean()
    )


    mean_mask_probability = float(

        history_df[
            "Whole_modality_fraction"
        ].mean()
    )


    with open(
        summary_file,
        "w"
    ) as f:


        f.write(
            "Paper 1 frozen PB-MMAE LinearFusion64 outer-fold training\n"
        )

        f.write(
            f"Fold: {fold}\n"
        )

        f.write(
            "Development fold excluded from final evaluation: Fold 3\n"
        )

        f.write(
            "Architecture frozen before outer-fold training: True\n"
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
            "Per-family LayerNorm before fusion: True\n"
        )

        f.write(
            "Fusion type: single linear projection\n"
        )

        f.write(
            "Fusion input dimension: 96\n"
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
            f"Seed: {fold_seed}\n"
        )

        f.write(
            f"Epochs: {EPOCHS}\n"
        )

        f.write(
            f"TRAIN patches: {len(train_dataset)}\n"
        )

        f.write(
            f"Final PB loss: {final_loss:.8f}\n"
        )

        f.write(
            f"Minimum PB loss: {minimum_loss:.8f}\n"
        )

        f.write(
            f"Minimum-loss epoch: {minimum_epoch}\n"
        )

        f.write(
            f"Final-25-epoch mean PB loss: {final25_mean:.8f}\n"
        )

        f.write(
            f"Mean whole-modality mask fraction: "
            f"{mean_mask_probability:.6f}\n"
        )

        f.write(
            f"Training seconds: {training_seconds:.2f}\n"
        )


    print(
        "\n" +
        "-" * 90
    )

    print(
        f"FOLD {fold} COMPLETED"
    )

    print(
        "-" * 90
    )

    print(
        "Final PB loss     :",
        round(
            final_loss,
            6
        )
    )

    print(
        "Minimum PB loss   :",
        round(
            minimum_loss,
            6
        )
    )

    print(
        "Minimum-loss epoch:",
        minimum_epoch
    )

    print(
        "Final-25 mean     :",
        round(
            final25_mean,
            6
        )
    )

    print(
        "Training seconds  :",
        round(
            training_seconds,
            2
        )
    )

    print(
        "Final model       :",
        final_model_file
    )


    return {

        "Fold":
            fold,

        "Seed":
            fold_seed,

        "TRAIN_patches":
            len(
                train_dataset
            ),

        "Final_PB_loss":
            final_loss,

        "Minimum_PB_loss":
            minimum_loss,

        "Minimum_loss_epoch":
            minimum_epoch,

        "Final25_mean_PB_loss":
            final25_mean,

        "Mean_whole_modality_fraction":
            mean_mask_probability,

        "Training_seconds":
            training_seconds
    }


# ============================================================
# 13. TRAIN ALL UNTOUCHED OUTER FOLDS
# ============================================================

print(
    "=" * 90
)

print(
    "FROZEN PB-MMAE OUTER-FOLD EVALUATION TRAINING"
)

print(
    "=" * 90
)

print(
    "Development fold : 3"
)

print(
    "Outer folds      :",
    OUTER_FOLDS
)

print(
    "Architecture     : "
    "LayerNorm + linear 96->64 fusion"
)

print(
    "No validation data will be loaded."
)


all_fold_records = []


total_start = (
    time.time()
)


for fold in OUTER_FOLDS:

    fold_record = train_fold(
        fold
    )

    all_fold_records.append(
        fold_record
    )


total_seconds = (
    time.time()
    -
    total_start
)


# ============================================================
# 14. SAVE OUTER-FOLD TRAINING SUMMARY
# ============================================================

outer_summary_df = pd.DataFrame(
    all_fold_records
)


outer_summary_file = (

    MODEL_ROOT /
    "PBMMAE_LINEAR64_outer_folds_training_summary.csv"
)


outer_summary_df.to_csv(

    outer_summary_file,

    index=False
)


# ============================================================
# 15. PRINT FINAL TABLE
# ============================================================

print(
    "\n\n" +
    "=" * 100
)

print(
    "ALL FROZEN OUTER-FOLD MODELS COMPLETED"
)

print(
    "=" * 100
)


print(

    outer_summary_df
    .round(
        {
            "Final_PB_loss": 6,

            "Minimum_PB_loss": 6,

            "Final25_mean_PB_loss": 6,

            "Mean_whole_modality_fraction": 4,

            "Training_seconds": 2
        }
    )
    .to_string(
        index=False
    )
)


print(
    "\nTotal training time:",
    round(
        total_seconds,
        2
    ),
    "seconds"
)


print(
    "\nCombined summary:"
)

print(
    outer_summary_file
)


print(
    "\nOuter-model directory:"
)

print(
    MODEL_ROOT
)


print(
    "=" * 100
)