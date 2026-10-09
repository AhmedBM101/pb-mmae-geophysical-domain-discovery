import os
# ============================================================
# Paper 1 — PB-MMAE 9x9 representation sensitivity evaluation
#
# Frozen architecture:
#   six family encoders
#   + per-family LayerNorm
#   + Linear(96 -> 64) fusion
#
# Purpose:
#   Extract unmasked TRAIN and VALIDATION embeddings and assess
#   geographic transfer for the 9x9 patch-size sensitivity test.
#
# IMPORTANT:
#   Validation data are used only for post-training evaluation.
# ============================================================

import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd


# ============================================================
# 1. PATHS / CONFIG
# ============================================================

PROJECT_ROOT = Path(os.environ["PBMMAE_PROJECT_ROOT"]).expanduser().resolve()

PATCH_ROOT = (
    PROJECT_ROOT /
    "03_outputs" /
    "patches" /
    "9x9"
)

MODEL_ROOT = (
    PROJECT_ROOT /
    "03_outputs" /
    "models" /
    "PBMMAE" /
    "OuterFolds_9x9_LinearFusion64_Sensitivity"
)

OUTPUT_ROOT = (
    MODEL_ROOT /
    "representation_evaluation"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

FOLDS = [1, 2, 4, 5]

PATCH_SIZE = 9
BRANCH_EMBED_DIM = 16
FUSED_LATENT_DIM = 64

DEVICE = torch.device("cpu")

torch.set_num_threads(2)


# ============================================================
# 2. PHYSICS FAMILIES
# ============================================================

PHYSICS_FAMILIES = {

    "magnetic": [0],

    "gravity": [1, 2, 3],

    "radiometric": [4, 5, 6],

    "thermal": [7],

    "structural": [8, 9, 10, 11],

    "terrain": [12, 13]
}

FAMILY_NAMES = list(
    PHYSICS_FAMILIES.keys()
)


# ============================================================
# 3. DATASET
# ============================================================

class PatchDataset(Dataset):

    def __init__(self, array):

        if array.ndim != 4:

            raise ValueError(
                f"Expected 4-D patch tensor, got {array.shape}"
            )

        if array.shape[-1] != 14:

            raise ValueError(
                f"Expected 14 channels, got {array.shape[-1]}"
            )

        self.data = torch.from_numpy(
            np.asarray(
                array,
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
# 4. MODEL DEFINITION
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


class PBMMAE(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()

        for family_name in FAMILY_NAMES:

            indices = PHYSICS_FAMILIES[
                family_name
            ]

            n_channels = len(indices)

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

        return fused, family_embeddings


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
# 5. EXTRACT REPRESENTATIONS
# ============================================================

def extract_representations(
    model,
    array,
    batch_size=128
):

    dataset = PatchDataset(
        array
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    fused_list = []

    family_lists = {
        family_name: []
        for family_name in FAMILY_NAMES
    }

    model.eval()

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(
                DEVICE
            )

            fused, family_embeddings = (
                model.encode(
                    batch
                )
            )

            fused_list.append(
                fused.cpu()
            )

            for family_name, family_z in zip(
                FAMILY_NAMES,
                family_embeddings
            ):

                family_lists[
                    family_name
                ].append(
                    family_z.cpu()
                )

    fused = torch.cat(
        fused_list,
        dim=0
    )

    families = {

        family_name:
            torch.cat(
                family_lists[
                    family_name
                ],
                dim=0
            )

        for family_name
        in FAMILY_NAMES
    }

    return fused, families


# ============================================================
# 6. DIAGNOSTICS
# ============================================================

def mean_dimension_sd(
    tensor
):

    sd = torch.std(
        tensor,
        dim=0,
        unbiased=True
    )

    return float(
        sd.mean().item()
    )


def near_zero_dimensions(
    tensor,
    threshold=1e-6
):

    sd = torch.std(
        tensor,
        dim=0,
        unbiased=True
    )

    return int(
        torch.sum(
            sd < threshold
        ).item()
    )


def mean_abs_offdiag_corr(
    tensor
):

    X = tensor.T

    corr = torch.corrcoef(
        X
    )

    n = corr.shape[0]

    mask = ~torch.eye(
        n,
        dtype=torch.bool,
        device=corr.device
    )

    values = torch.abs(
        corr[
            mask
        ]
    )

    values = values[
        torch.isfinite(
            values
        )
    ]

    if values.numel() == 0:

        return np.nan

    return float(
        values.mean().item()
    )


def family_sd_ratio(
    train_tensor,
    val_tensor
):

    train_sd = mean_dimension_sd(
        train_tensor
    )

    val_sd = mean_dimension_sd(
        val_tensor
    )

    if train_sd == 0:

        ratio = np.nan

    else:

        ratio = (
            val_sd /
            train_sd
        )

    return (
        train_sd,
        val_sd,
        ratio
    )


# ============================================================
# 7. EVALUATE ONE FOLD
# ============================================================

fused_records = []
family_records = []


for fold in FOLDS:

    print(
        "\n" +
        "=" * 100
    )

    print(
        f"FOLD {fold} — 9x9 REPRESENTATION EVALUATION"
    )

    print(
        "=" * 100
    )


    fold_patch_dir = (
        PATCH_ROOT /
        f"Fold_{fold}"
    )

    fold_model_dir = (
        MODEL_ROOT /
        f"Fold{fold}"
    )

    fold_output_dir = (
        OUTPUT_ROOT /
        f"Fold{fold}"
    )

    fold_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    train_file = (
        fold_patch_dir /
        f"Fold{fold}_TRAIN_9x9_14ch.npy"
    )

    val_file = (
        fold_patch_dir /
        f"Fold{fold}_VALIDATION_9x9_14ch.npy"
    )

    model_file = (
        fold_model_dir /
        f"Fold{fold}_9x9_PBMMAE_LinearFusion64_FINAL.pt"
    )


    for path in [
        train_file,
        val_file,
        model_file
    ]:

        if not path.exists():

            raise FileNotFoundError(
                path
            )


    train_array = np.load(
        train_file
    )

    val_array = np.load(
        val_file
    )


    print(
        "TRAIN:",
        train_array.shape
    )

    print(
        "VALIDATION:",
        val_array.shape
    )


    checkpoint = torch.load(
        model_file,
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


    # --------------------------------------------------------
    # Extract UNMASKED representations
    # --------------------------------------------------------

    train_fused, train_families = (
        extract_representations(
            model,
            train_array
        )
    )

    val_fused, val_families = (
        extract_representations(
            model,
            val_array
        )
    )


    # --------------------------------------------------------
    # Save latent arrays
    # --------------------------------------------------------

    train_fused_np = (
        train_fused.numpy()
    )

    val_fused_np = (
        val_fused.numpy()
    )


    np.save(

        fold_output_dir /
        f"PBMMAE_LINEAR64_Fold{fold}_9x9_TRAIN_latent64.npy",

        train_fused_np
    )


    np.save(

        fold_output_dir /
        f"PBMMAE_LINEAR64_Fold{fold}_9x9_VALIDATION_latent64.npy",

        val_fused_np
    )


    for family_name in FAMILY_NAMES:

        np.save(

            fold_output_dir /
            f"PBMMAE_LINEAR64_Fold{fold}_9x9_"
            f"TRAIN_{family_name}_latent16.npy",

            train_families[
                family_name
            ].numpy()
        )

        np.save(

            fold_output_dir /
            f"PBMMAE_LINEAR64_Fold{fold}_9x9_"
            f"VALIDATION_{family_name}_latent16.npy",

            val_families[
                family_name
            ].numpy()
        )


    # --------------------------------------------------------
    # Fused diagnostics
    # --------------------------------------------------------

    train_sd = mean_dimension_sd(
        train_fused
    )

    val_sd = mean_dimension_sd(
        val_fused
    )

    sd_ratio = (
        val_sd /
        train_sd
    )


    train_zero = near_zero_dimensions(
        train_fused
    )

    val_zero = near_zero_dimensions(
        val_fused
    )


    train_corr = mean_abs_offdiag_corr(
        train_fused
    )

    val_corr = mean_abs_offdiag_corr(
        val_fused
    )


    fused_records.append({

        "Fold":
            fold,

        "Patch_size":
            PATCH_SIZE,

        "Train_N":
            train_fused.shape[0],

        "Validation_N":
            val_fused.shape[0],

        "Latent_dim":
            train_fused.shape[1],

        "Train_mean_dimension_SD":
            train_sd,

        "Validation_mean_dimension_SD":
            val_sd,

        "Validation_to_Train_SD_ratio":
            sd_ratio,

        "Train_near_zero_dimensions":
            train_zero,

        "Validation_near_zero_dimensions":
            val_zero,

        "Train_mean_abs_offdiag_corr":
            train_corr,

        "Validation_mean_abs_offdiag_corr":
            val_corr
    })


    # --------------------------------------------------------
    # Family diagnostics
    # --------------------------------------------------------

    for family_name in FAMILY_NAMES:

        family_train = train_families[
            family_name
        ]

        family_val = val_families[
            family_name
        ]


        family_train_sd, family_val_sd, ratio = (
            family_sd_ratio(
                family_train,
                family_val
            )
        )


        family_records.append({

            "Fold":
                fold,

            "Patch_size":
                PATCH_SIZE,

            "Physics_family":
                family_name,

            "Train_mean_dimension_SD":
                family_train_sd,

            "Validation_mean_dimension_SD":
                family_val_sd,

            "Validation_to_Train_SD_ratio":
                ratio,

            "Train_near_zero_dimensions":
                near_zero_dimensions(
                    family_train
                ),

            "Validation_near_zero_dimensions":
                near_zero_dimensions(
                    family_val
                ),

            "Train_mean_abs_offdiag_corr":
                mean_abs_offdiag_corr(
                    family_train
                ),

            "Validation_mean_abs_offdiag_corr":
                mean_abs_offdiag_corr(
                    family_val
                )
        })


    print(
        "\nFused latent diagnostics:"
    )

    print(
        " TRAIN mean SD:",
        round(
            train_sd,
            4
        )
    )

    print(
        " VALIDATION mean SD:",
        round(
            val_sd,
            4
        )
    )

    print(
        " VAL/TRAIN SD ratio:",
        round(
            sd_ratio,
            4
        )
    )

    print(
        " Near-zero dims TRAIN/VAL:",
        train_zero,
        "/",
        val_zero
    )

    print(
        " Mean |offdiag corr| TRAIN:",
        round(
            train_corr,
            4
        )
    )

    print(
        " Mean |offdiag corr| VAL:",
        round(
            val_corr,
            4
        )
    )


# ============================================================
# 8. SAVE COMBINED TABLES
# ============================================================

fused_df = pd.DataFrame(
    fused_records
)

family_df = pd.DataFrame(
    family_records
)


fused_file = (
    OUTPUT_ROOT /
    "PBMMAE_9x9_outer_folds_fused_representation_diagnostics.csv"
)

family_file = (
    OUTPUT_ROOT /
    "PBMMAE_9x9_outer_folds_family_representation_diagnostics.csv"
)


fused_df.to_csv(
    fused_file,
    index=False
)

family_df.to_csv(
    family_file,
    index=False
)


# ============================================================
# 9. SUMMARY STATISTICS
# ============================================================

summary = pd.DataFrame(
    [{
        "Patch_size":
            PATCH_SIZE,

        "N_outer_folds":
            len(
                FOLDS
            ),

        "Mean_VAL_TRAIN_SD_ratio":
            fused_df[
                "Validation_to_Train_SD_ratio"
            ].mean(),

        "SD_VAL_TRAIN_SD_ratio":
            fused_df[
                "Validation_to_Train_SD_ratio"
            ].std(
                ddof=1
            ),

        "Min_VAL_TRAIN_SD_ratio":
            fused_df[
                "Validation_to_Train_SD_ratio"
            ].min(),

        "Max_VAL_TRAIN_SD_ratio":
            fused_df[
                "Validation_to_Train_SD_ratio"
            ].max(),

        "Total_train_near_zero_dimensions":
            fused_df[
                "Train_near_zero_dimensions"
            ].sum(),

        "Total_validation_near_zero_dimensions":
            fused_df[
                "Validation_near_zero_dimensions"
            ].sum(),

        "Mean_train_abs_offdiag_corr":
            fused_df[
                "Train_mean_abs_offdiag_corr"
            ].mean(),

        "Mean_validation_abs_offdiag_corr":
            fused_df[
                "Validation_mean_abs_offdiag_corr"
            ].mean()
    }]
)


summary_file = (
    OUTPUT_ROOT /
    "PBMMAE_9x9_outer_folds_representation_summary.csv"
)


summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# 10. PRINT FINAL RESULTS
# ============================================================

print(
    "\n\n" +
    "=" * 110
)

print(
    "9x9 OUTER-FOLD FUSED REPRESENTATION DIAGNOSTICS"
)

print(
    "=" * 110
)


print(

    fused_df[
        [
            "Fold",
            "Train_N",
            "Validation_N",
            "Train_mean_dimension_SD",
            "Validation_mean_dimension_SD",
            "Validation_to_Train_SD_ratio",
            "Train_near_zero_dimensions",
            "Validation_near_zero_dimensions",
            "Train_mean_abs_offdiag_corr",
            "Validation_mean_abs_offdiag_corr"
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
    "9x9 REPRESENTATION SUMMARY"
)

print(
    "=" * 110
)


print(
    summary
    .round(
        4
    )
    .to_string(
        index=False
    )
)


print(
    "\nSaved fused diagnostics:"
)

print(
    fused_file
)


print(
    "\nSaved family diagnostics:"
)

print(
    family_file
)


print(
    "\nSaved summary:"
)

print(
    summary_file
)


print(
    "\nIMPORTANT:"
)

print(
    "All representations were extracted from UNMASKED patches."
)

print(
    "Validation data were used only after training."
)


print(
    "=" * 110
)