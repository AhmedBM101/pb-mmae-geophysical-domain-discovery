import os
# ============================================================
# Paper 1 — PB-MMAE Ablation C representation evaluation
#
# Ablation:
#   Remove MAG_LD
#
# Purpose:
#   Extract UNMASKED TRAIN and VALIDATION representations
#   from the 13-channel structural-redundancy ablation and
#   compare geographic transfer against the frozen 14-channel
#   PB-MMAE.
#
# Metrics:
#   - fused latent mean dimension SD
#   - VAL/TRAIN SD ratio
#   - near-zero latent dimensions
#   - mean absolute off-diagonal correlation
#   - family-level transfer ratios
#
# Validation data are used only after training.
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
# 1. PATHS / CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(os.environ["PBMMAE_PROJECT_ROOT"]).expanduser().resolve()

PATCH_ROOT = (
    PROJECT_ROOT /
    "03_outputs" /
    "patches" /
    "13x13"
)

MODEL_ROOT = (
    PROJECT_ROOT /
    "03_outputs" /
    "models" /
    "PBMMAE" /
    "ablations" /
    "C_Remove_MAG_LD_13x13"
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

PATCH_SIZE = 13

BRANCH_EMBED_DIM = 16
FUSED_LATENT_DIM = 64

REMOVE_ORIGINAL_CHANNEL_INDEX = 8

DEVICE = torch.device("cpu")

torch.set_num_threads(2)


# ============================================================
# 2. ABLATED CHANNELS
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
    "GRAV_LD",
    "DEM_LD",
    "ID",
    "DEM",
    "SLOPE"
]

PHYSICS_FAMILIES = {

    "magnetic": [0],

    "gravity": [1, 2, 3],

    "radiometric": [4, 5, 6],

    "thermal": [7],

    "structural": [8, 9, 10],

    "terrain": [11, 12]
}

FAMILY_NAMES = list(
    PHYSICS_FAMILIES.keys()
)


# ============================================================
# 3. DATASET
# ============================================================

class PatchDataset(Dataset):

    def __init__(self, array_14ch):

        if array_14ch.ndim != 4:
            raise ValueError(
                f"Expected 4-D tensor; got {array_14ch.shape}"
            )

        if array_14ch.shape[-1] != 14:
            raise ValueError(
                f"Expected original 14 channels; "
                f"got {array_14ch.shape[-1]}"
            )

        array_13ch = np.delete(
            array_14ch,
            REMOVE_ORIGINAL_CHANNEL_INDEX,
            axis=-1
        )

        if array_13ch.shape[-1] != 13:
            raise RuntimeError(
                f"Unexpected ablated shape: {array_13ch.shape}"
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
# 4. MODEL
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

            fused, family_embeddings = model.encode(
                batch
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

def mean_dimension_sd(tensor):

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

    corr = torch.corrcoef(
        tensor.T
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


# ============================================================
# 7. OUTER-FOLD EVALUATION
# ============================================================

fused_records = []
family_records = []


for fold in FOLDS:

    print(
        "\n" +
        "=" * 100
    )

    print(
        f"ABLATION C — FOLD {fold} REPRESENTATION EVALUATION"
    )

    print(
        "=" * 100
    )


    patch_dir = (
        PATCH_ROOT /
        f"Fold_{fold}"
    )

    model_dir = (
        MODEL_ROOT /
        f"Fold{fold}"
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
        patch_dir /
        f"Fold{fold}_TRAIN_13x13_14ch.npy"
    )

    validation_file = (
        patch_dir /
        f"Fold{fold}_VALIDATION_13x13_14ch.npy"
    )

    model_file = (
        model_dir /
        f"Fold{fold}_AblationC_Remove_MAG_LD_FINAL.pt"
    )


    for path in [
        train_file,
        validation_file,
        model_file
    ]:

        if not path.exists():
            raise FileNotFoundError(
                path
            )


    train_array = np.load(
        train_file
    )

    validation_array = np.load(
        validation_file
    )


    print(
        "Original TRAIN:",
        train_array.shape
    )

    print(
        "Original VALIDATION:",
        validation_array.shape
    )

    print(
        "MAG_LD removed during Dataset construction."
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
    # UNMASKED representations
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
            validation_array
        )
    )


    # --------------------------------------------------------
    # Save fused embeddings
    # --------------------------------------------------------

    np.save(
        fold_output /
        f"AblationC_Fold{fold}_13x13_TRAIN_latent64.npy",
        train_fused.numpy()
    )

    np.save(
        fold_output /
        f"AblationC_Fold{fold}_13x13_VALIDATION_latent64.npy",
        val_fused.numpy()
    )


    # --------------------------------------------------------
    # Save family embeddings
    # --------------------------------------------------------

    for family_name in FAMILY_NAMES:

        np.save(
            fold_output /
            f"AblationC_Fold{fold}_13x13_"
            f"TRAIN_{family_name}_latent16.npy",
            train_families[
                family_name
            ].numpy()
        )

        np.save(
            fold_output /
            f"AblationC_Fold{fold}_13x13_"
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

        "Ablation":
            "Remove_MAG_LD",

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

        family_train_sd = mean_dimension_sd(
            family_train
        )

        family_val_sd = mean_dimension_sd(
            family_val
        )

        family_ratio = (
            family_val_sd /
            family_train_sd
        )


        family_records.append({

            "Fold":
                fold,

            "Ablation":
                "Remove_MAG_LD",

            "Physics_family":
                family_name,

            "Train_mean_dimension_SD":
                family_train_sd,

            "Validation_mean_dimension_SD":
                family_val_sd,

            "Validation_to_Train_SD_ratio":
                family_ratio,

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
        "TRAIN mean SD:",
        round(
            train_sd,
            4
        )
    )

    print(
        "VALIDATION mean SD:",
        round(
            val_sd,
            4
        )
    )

    print(
        "VAL/TRAIN SD ratio:",
        round(
            sd_ratio,
            4
        )
    )

    print(
        "Near-zero dimensions TRAIN/VAL:",
        train_zero,
        "/",
        val_zero
    )

    print(
        "Mean |offdiag corr| TRAIN:",
        round(
            train_corr,
            4
        )
    )

    print(
        "Mean |offdiag corr| VALIDATION:",
        round(
            val_corr,
            4
        )
    )


# ============================================================
# 8. SAVE DIAGNOSTIC TABLES
# ============================================================

fused_df = pd.DataFrame(
    fused_records
)

family_df = pd.DataFrame(
    family_records
)


fused_file = (
    OUTPUT_ROOT /
    "PBMMAE_AblationC_Remove_MAG_LD_fused_representation_diagnostics.csv"
)

family_file = (
    OUTPUT_ROOT /
    "PBMMAE_AblationC_Remove_MAG_LD_family_representation_diagnostics.csv"
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
# 9. SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    [{
        "Ablation":
            "Remove_MAG_LD",

        "Removed_channel":
            "MAG_LD",

        "Remaining_channels":
            13,

        "Patch_size":
            PATCH_SIZE,

        "N_outer_folds":
            len(FOLDS),

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
    "PBMMAE_AblationC_Remove_MAG_LD_representation_summary.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# 10. PRINT RESULTS
# ============================================================

print(
    "\n\n" +
    "=" * 115
)

print(
    "ABLATION C — FUSED REPRESENTATION DIAGNOSTICS"
)

print(
    "=" * 115
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
    "=" * 115
)

print(
    "ABLATION C — REPRESENTATION SUMMARY"
)

print(
    "=" * 115
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
    "MAG_LD was removed before inference."
)

print(
    "Validation data were used only after training."
)

print(
    "The structural branch now contains only "
    "GRAV_LD, DEM_LD, and ID."
)


print(
    "=" * 115
)