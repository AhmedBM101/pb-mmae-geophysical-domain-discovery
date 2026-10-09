import os
# ============================================================
# Standalone CAE training
# Paper 1
# Fold 3, 13x13, 14 channels
# ============================================================

import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

from pathlib import Path
import random
import time

# Import torch first
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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

CAE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

PATCH_SIZE = 13
FOLD = 3

N_CHANNELS = 14
LATENT_DIM = 32

SEED = 42

EPOCHS = 200
BATCH_SIZE = 32

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# Use CPU for maximum process stability.
DEVICE = torch.device("cpu")

torch.set_num_threads(2)

print("=" * 70)
print("PAPER 1 CAE TRAINING")
print("=" * 70)

print("Device      :", DEVICE)
print("Patch size  :", PATCH_SIZE)
print("Fold        :", FOLD)
print("Channels    :", N_CHANNELS)
print("Latent dim  :", LATENT_DIM)
print("Epochs      :", EPOCHS)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# MODEL
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
# LOAD TRAIN DATA
# ============================================================

train_file = (
    PATCH_DIR /
    "13x13" /
    "Fold_3" /
    "Fold3_TRAIN_13x13_14ch.npy"
)

print("\nTRAIN file:")
print(train_file)

if not train_file.exists():

    raise FileNotFoundError(
        train_file
    )


X_train = np.load(
    train_file
).astype(np.float32)

print(
    "\nOriginal array:",
    X_train.shape
)

# NHWC -> NCHW
X_train = np.transpose(
    X_train,
    (0, 3, 1, 2)
)

X_train_tensor = torch.from_numpy(
    X_train
)

print(
    "PyTorch tensor:",
    X_train_tensor.shape
)

print(
    "Finite:",
    torch.isfinite(
        X_train_tensor
    ).all().item()
)


# ============================================================
# DATALOADER
# ============================================================

dataset = TensorDataset(
    X_train_tensor
)

generator = torch.Generator()
generator.manual_seed(SEED)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator,
    num_workers=0,
    pin_memory=False
)


# ============================================================
# INITIALIZE MODEL
# ============================================================

model = ConvAutoencoder(
    in_channels=N_CHANNELS,
    latent_dim=LATENT_DIM
).to(DEVICE)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# TRAINING
# ============================================================

loss_history = []

start_time = time.time()

for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    total_loss = 0.0
    total_n = 0

    for (batch_x,) in loader:

        batch_x = batch_x.to(
            DEVICE
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        reconstruction, _ = model(
            batch_x
        )

        loss = criterion(
            reconstruction,
            batch_x
        )

        loss.backward()

        optimizer.step()

        n = batch_x.size(0)

        total_loss += (
            loss.item() * n
        )

        total_n += n


    epoch_loss = (
        total_loss /
        total_n
    )

    loss_history.append(
        epoch_loss
    )


    if (
        epoch == 1
        or epoch % 10 == 0
        or epoch == EPOCHS
    ):

        print(
            f"Epoch "
            f"{epoch:3d}/{EPOCHS} | "
            f"MSE = {epoch_loss:.6f}",
            flush=True
        )


    # --------------------------------------------------------
    # Lightweight model-state checkpoint every 20 epochs
    # --------------------------------------------------------

    if epoch % 20 == 0:

        checkpoint = (
            CAE_DIR /
            f"Fold3_13x13_CAE_epoch{epoch:03d}.pt"
        )

        torch.save(
            {
                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "loss_history":
                    loss_history
            },
            checkpoint
        )


elapsed = time.time() - start_time


# ============================================================
# FINAL TRAIN MSE
# ============================================================

model.eval()

total_loss = 0.0
total_n = 0

with torch.no_grad():

    for (batch_x,) in loader:

        batch_x = batch_x.to(
            DEVICE
        )

        reconstruction, _ = model(
            batch_x
        )

        loss = criterion(
            reconstruction,
            batch_x
        )

        n = batch_x.size(0)

        total_loss += (
            loss.item() * n
        )

        total_n += n


final_train_mse = (
    total_loss /
    total_n
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_file = (
    CAE_DIR /
    "Fold3_13x13_CAE_latent32_FINAL.pt"
)

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "patch_size":
            PATCH_SIZE,

        "fold":
            FOLD,

        "latent_dim":
            LATENT_DIM,

        "n_channels":
            N_CHANNELS,

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

        "final_train_mse":
            final_train_mse
    },
    final_model_file
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame({

    "Epoch":
        np.arange(
            1,
            EPOCHS + 1
        ),

    "TRAIN_MSE":
        loss_history
})

history_file = (
    CAE_DIR /
    "Fold3_13x13_CAE_training_history.csv"
)

history_df.to_csv(
    history_file,
    index=False
)


# ============================================================
# SAVE TEXT SUMMARY
# ============================================================

summary_file = (
    CAE_DIR /
    "Fold3_13x13_CAE_training_summary.txt"
)

with open(
    summary_file,
    "w"
) as f:

    f.write(
        "Paper 1 CAE baseline\n"
    )

    f.write(
        "Patch size: 13x13\n"
    )

    f.write(
        "Fold: 3\n"
    )

    f.write(
        "Channels: 14\n"
    )

    f.write(
        "Latent dimension: 32\n"
    )

    f.write(
        f"Epochs: {EPOCHS}\n"
    )

    f.write(
        f"Final TRAIN MSE: "
        f"{final_train_mse:.8f}\n"
    )

    f.write(
        f"Training seconds: "
        f"{elapsed:.2f}\n"
    )


print("\n" + "=" * 70)

print(
    "Training completed."
)

print(
    "Training time:",
    round(elapsed, 2),
    "seconds"
)

print(
    "Final TRAIN MSE:",
    round(
        final_train_mse,
        6
    )
)

print("\nFinal model:")
print(final_model_file)

print("\nHistory:")
print(history_file)

print("\nSummary:")
print(summary_file)

print("=" * 70)