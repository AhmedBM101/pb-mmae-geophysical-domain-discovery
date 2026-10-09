import os
# ============================================================
# Evaluate final CAE + export latent embeddings
# Paper 1
# Fold 3, 13x13, 14 channels
# ============================================================

import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

from pathlib import Path

# Import PyTorch first
import torch
import torch.nn as nn

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

CAE_DIR = (
    PROJECT_ROOT /
    "03_outputs" /
    "baselines" /
    "CAE"
)

MODEL_FILE = (
    CAE_DIR /
    "Fold3_13x13_CAE_latent32_FINAL.pt"
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
# CONFIGURATION
# ============================================================

N_CHANNELS = 14
LATENT_DIM = 32

DEVICE = torch.device("cpu")

torch.set_num_threads(2)

print("=" * 72)
print("CAE HELD-OUT EVALUATION")
print("=" * 72)

print("Device:", DEVICE)
print("Model :", MODEL_FILE)


# ============================================================
# MODEL DEFINITION
# Must exactly match training architecture
# ============================================================

class ConvAutoencoder(nn.Module):

    def __init__(
        self,
        in_channels=14,
        latent_dim=32
    ):
        super().__init__()

        self.encoder_conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(
                (3, 3)
            )
        )

        self.encoder_fc = nn.Linear(
            64 * 3 * 3,
            latent_dim
        )

        self.decoder_fc = nn.Linear(
            latent_dim,
            64 * 3 * 3
        )

        self.decoder_conv = nn.Sequential(

            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                in_channels,
                kernel_size=3,
                padding=1
            )
        )

    def encode(self, x):

        x = self.encoder_conv(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        return self.encoder_fc(x)

    def decode(
        self,
        z,
        output_size
    ):

        x = self.decoder_fc(z)

        x = x.view(
            -1,
            64,
            3,
            3
        )

        x = nn.functional.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False
        )

        return self.decoder_conv(x)

    def forward(self, x):

        output_size = (
            x.shape[-2],
            x.shape[-1]
        )

        z = self.encode(x)

        reconstruction = self.decode(
            z,
            output_size
        )

        return reconstruction, z


# ============================================================
# CHECK FILES
# ============================================================

for f in [
    MODEL_FILE,
    TRAIN_FILE,
    VAL_FILE
]:

    print(
        f"\nExists: {f.exists()} | {f}"
    )

    if not f.exists():

        raise FileNotFoundError(f)


# ============================================================
# LOAD DATA
# ============================================================

X_train = np.load(
    TRAIN_FILE
).astype(np.float32)

X_val = np.load(
    VAL_FILE
).astype(np.float32)

print(
    "\nOriginal TRAIN:",
    X_train.shape
)

print(
    "Original VALIDATION:",
    X_val.shape
)

# NHWC -> NCHW
X_train_t = torch.from_numpy(
    np.transpose(
        X_train,
        (0, 3, 1, 2)
    )
)

X_val_t = torch.from_numpy(
    np.transpose(
        X_val,
        (0, 3, 1, 2)
    )
)

print(
    "\nPyTorch TRAIN:",
    X_train_t.shape
)

print(
    "PyTorch VALIDATION:",
    X_val_t.shape
)

assert torch.isfinite(
    X_train_t
).all()

assert torch.isfinite(
    X_val_t
).all()


# ============================================================
# LOAD FINAL MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False
)

model = ConvAutoencoder(
    in_channels=N_CHANNELS,
    latent_dim=LATENT_DIM
).to(DEVICE)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()

print(
    "\nSaved final TRAIN MSE:",
    checkpoint.get(
        "final_train_mse",
        "not found"
    )
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

criterion = nn.MSELoss(
    reduction="mean"
)


def evaluate_and_encode(
    tensor,
    batch_size=64
):

    recon_loss_sum = 0.0
    total_n = 0

    latent_batches = []

    with torch.no_grad():

        for start in range(
            0,
            len(tensor),
            batch_size
        ):

            batch = tensor[
                start:
                start + batch_size
            ].to(DEVICE)

            recon, latent = model(
                batch
            )

            batch_loss = criterion(
                recon,
                batch
            )

            n = batch.size(0)

            recon_loss_sum += (
                batch_loss.item() * n
            )

            total_n += n

            latent_batches.append(
                latent.cpu().numpy()
            )

    mean_mse = (
        recon_loss_sum /
        total_n
    )

    latent_array = np.concatenate(
        latent_batches,
        axis=0
    )

    return mean_mse, latent_array


# ============================================================
# TRAIN + VALIDATION
# ============================================================

train_mse, Z_train = (
    evaluate_and_encode(
        X_train_t
    )
)

val_mse, Z_val = (
    evaluate_and_encode(
        X_val_t
    )
)

qe_ratio = (
    val_mse /
    train_mse
)

print(
    "\nRecomputed TRAIN MSE:",
    round(train_mse, 6)
)

print(
    "Held-out VALIDATION MSE:",
    round(val_mse, 6)
)

print(
    "VAL / TRAIN MSE ratio:",
    round(qe_ratio, 4)
)

print(
    "\nTRAIN latent shape:",
    Z_train.shape
)

print(
    "VALIDATION latent shape:",
    Z_val.shape
)

print(
    "TRAIN latent finite:",
    np.isfinite(
        Z_train
    ).all()
)

print(
    "VALIDATION latent finite:",
    np.isfinite(
        Z_val
    ).all()
)


# ============================================================
# LATENT DIAGNOSTICS
# ============================================================

train_latent_std = (
    Z_train.std(
        axis=0,
        ddof=1
    )
)

val_latent_std = (
    Z_val.std(
        axis=0,
        ddof=1
    )
)

near_zero_train_dims = int(
    np.sum(
        train_latent_std <
        1e-8
    )
)

near_zero_val_dims = int(
    np.sum(
        val_latent_std <
        1e-8
    )
)

print(
    "\nNear-zero TRAIN latent dimensions:",
    near_zero_train_dims
)

print(
    "Near-zero VALIDATION latent dimensions:",
    near_zero_val_dims
)


# ============================================================
# SAVE LATENT EMBEDDINGS
# ============================================================

train_latent_file = (
    CAE_DIR /
    "Fold3_13x13_CAE_TRAIN_latent32.npy"
)

val_latent_file = (
    CAE_DIR /
    "Fold3_13x13_CAE_VALIDATION_latent32.npy"
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
# SAVE RECONSTRUCTION METRICS
# ============================================================

metrics_df = pd.DataFrame(
    [{
        "Fold": 3,
        "Patch_size": 13,
        "Latent_dim": 32,

        "TRAIN_patches":
            Z_train.shape[0],

        "VALIDATION_patches":
            Z_val.shape[0],

        "TRAIN_reconstruction_MSE":
            train_mse,

        "VALIDATION_reconstruction_MSE":
            val_mse,

        "VAL_to_TRAIN_MSE_ratio":
            qe_ratio,

        "TRAIN_near_zero_latent_dims":
            near_zero_train_dims,

        "VALIDATION_near_zero_latent_dims":
            near_zero_val_dims
    }]
)

metrics_file = (
    CAE_DIR /
    "Fold3_13x13_CAE_reconstruction_metrics.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False
)


# ============================================================
# ALSO SAVE LATENTS AS CSV FOR AUDITABILITY
# ============================================================

latent_columns = [
    f"Z{i+1:02d}"
    for i in range(
        LATENT_DIM
    )
]

train_latent_csv = (
    CAE_DIR /
    "Fold3_13x13_CAE_TRAIN_latent32.csv"
)

val_latent_csv = (
    CAE_DIR /
    "Fold3_13x13_CAE_VALIDATION_latent32.csv"
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
# FINISH
# ============================================================

print("\n" + "=" * 72)
print("Evaluation completed.")

print("\nSaved TRAIN latent:")
print(train_latent_file)

print("\nSaved VALIDATION latent:")
print(val_latent_file)

print("\nSaved metrics:")
print(metrics_file)

print("\nSaved TRAIN latent CSV:")
print(train_latent_csv)

print("\nSaved VALIDATION latent CSV:")
print(val_latent_csv)

print("=" * 72)