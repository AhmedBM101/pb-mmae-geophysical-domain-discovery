import os
# ============================================================
# Paper 1 — Full CAE baseline
# 9x9 and 13x13
# Folds 1–5
# CAE latent=32 + KMeans(k=2)
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
from torch.utils.data import TensorDataset, DataLoader

# Remaining libraries
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


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

FULL_DIR = (
    CAE_DIR /
    "full_5fold"
)

FULL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

PATCH_SIZES = [9, 13]
FOLDS = [1, 2, 3, 4, 5]

N_CHANNELS = 14
LATENT_DIM = 32

EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

K = 2
KMEANS_N_INIT = 50

BASE_SEED = 42

DEVICE = torch.device("cpu")

torch.set_num_threads(2)

print("=" * 78)
print("FULL CAE BASELINE")
print("=" * 78)

print("Device      :", DEVICE)
print("Patch sizes :", PATCH_SIZES)
print("Folds       :", FOLDS)
print("Latent dim  :", LATENT_DIM)
print("Epochs      :", EPOCHS)
print("KMeans k    :", K)


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

        recon = self.decode(
            z,
            output_size
        )

        return recon, z


# ============================================================
# HELPERS
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_torch_tensor(array):

    # NHWC -> NCHW
    array = np.transpose(
        array,
        (0, 3, 1, 2)
    )

    return torch.from_numpy(
        array.astype(np.float32)
    )


def evaluate_and_encode(
    model,
    tensor,
    batch_size=64
):

    criterion = nn.MSELoss(
        reduction="mean"
    )

    model.eval()

    total_loss = 0.0
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

            loss = criterion(
                recon,
                batch
            )

            n = batch.size(0)

            total_loss += (
                loss.item() * n
            )

            total_n += n

            latent_batches.append(
                latent.cpu().numpy()
            )

    mean_mse = (
        total_loss /
        total_n
    )

    latent_array = np.concatenate(
        latent_batches,
        axis=0
    )

    return mean_mse, latent_array


# ============================================================
# MAIN LOOP
# ============================================================

records = []

global_start = time.time()

for patch_size in PATCH_SIZES:

    print("\n" + "=" * 78)
    print(
        f"PATCH SIZE {patch_size}x{patch_size}"
    )
    print("=" * 78)

    for fold in FOLDS:

        run_seed = (
            BASE_SEED
            + patch_size * 100
            + fold
        )

        set_seed(
            run_seed
        )

        print(
            f"\nPatch {patch_size}x{patch_size} | "
            f"Fold {fold}"
        )

        # ----------------------------------------------------
        # Paths
        # ----------------------------------------------------

        fold_dir = (
            PATCH_DIR /
            f"{patch_size}x{patch_size}" /
            f"Fold_{fold}"
        )

        train_file = (
            fold_dir /
            f"Fold{fold}_TRAIN_"
            f"{patch_size}x{patch_size}_14ch.npy"
        )

        val_file = (
            fold_dir /
            f"Fold{fold}_VALIDATION_"
            f"{patch_size}x{patch_size}_14ch.npy"
        )

        if not train_file.exists():
            raise FileNotFoundError(
                train_file
            )

        if not val_file.exists():
            raise FileNotFoundError(
                val_file
            )

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        X_train = np.load(
            train_file
        ).astype(np.float32)

        X_val = np.load(
            val_file
        ).astype(np.float32)

        X_train_t = to_torch_tensor(
            X_train
        )

        X_val_t = to_torch_tensor(
            X_val
        )

        assert torch.isfinite(
            X_train_t
        ).all()

        assert torch.isfinite(
            X_val_t
        ).all()

        print(
            "TRAIN:",
            X_train_t.shape,
            "| VAL:",
            X_val_t.shape
        )

        # ----------------------------------------------------
        # DataLoader
        # ----------------------------------------------------

        dataset = TensorDataset(
            X_train_t
        )

        generator = (
            torch.Generator()
        )

        generator.manual_seed(
            run_seed
        )

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=False
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        run_start = time.time()

        train_history = []

        for epoch in range(
            1,
            EPOCHS + 1
        ):

            model.train()

            running_loss = 0.0
            n_samples = 0

            for (batch_x,) in loader:

                batch_x = batch_x.to(
                    DEVICE
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                recon, _ = model(
                    batch_x
                )

                loss = criterion(
                    recon,
                    batch_x
                )

                loss.backward()

                optimizer.step()

                n = batch_x.size(0)

                running_loss += (
                    loss.item() * n
                )

                n_samples += n

            epoch_loss = (
                running_loss /
                n_samples
            )

            train_history.append(
                epoch_loss
            )

            if (
                epoch == 1
                or epoch % 50 == 0
                or epoch == EPOCHS
            ):

                print(
                    f"  Epoch "
                    f"{epoch:3d}/{EPOCHS} | "
                    f"MSE={epoch_loss:.6f}"
                )

        training_seconds = (
            time.time()
            - run_start
        )

        # ----------------------------------------------------
        # Evaluate + latent embeddings
        # ----------------------------------------------------

        train_mse, Z_train = (
            evaluate_and_encode(
                model,
                X_train_t
            )
        )

        val_mse, Z_val = (
            evaluate_and_encode(
                model,
                X_val_t
            )
        )

        mse_ratio = (
            val_mse /
            train_mse
        )

        # ----------------------------------------------------
        # Latent variance diagnostics
        # ----------------------------------------------------

        train_std = Z_train.std(
            axis=0,
            ddof=1
        )

        val_std = Z_val.std(
            axis=0,
            ddof=1
        )

        train_near_zero_dims = int(
            np.sum(
                train_std < 1e-8
            )
        )

        val_near_zero_dims = int(
            np.sum(
                val_std < 1e-8
            )
        )

        # ----------------------------------------------------
        # KMeans on TRAIN latent
        # ----------------------------------------------------

        km = KMeans(
            n_clusters=K,
            random_state=42,
            n_init=KMEANS_N_INIT,
            algorithm="lloyd"
        )

        train_labels = km.fit_predict(
            Z_train
        )

        val_labels = km.predict(
            Z_val
        )

        # ----------------------------------------------------
        # Internal metrics
        # ----------------------------------------------------

        sil = silhouette_score(
            Z_train,
            train_labels
        )

        dbi = davies_bouldin_score(
            Z_train,
            train_labels
        )

        ch = calinski_harabasz_score(
            Z_train,
            train_labels
        )

        # ----------------------------------------------------
        # Cluster occupancy
        # ----------------------------------------------------

        train_counts = np.bincount(
            train_labels,
            minlength=K
        )

        val_counts = np.bincount(
            val_labels,
            minlength=K
        )

        empty_val = int(
            np.sum(
                val_counts == 0
            )
        )

        # ----------------------------------------------------
        # Save model + embeddings
        # ----------------------------------------------------

        run_dir = (
            FULL_DIR /
            f"{patch_size}x{patch_size}" /
            f"Fold_{fold}"
        )

        run_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        model_file = (
            run_dir /
            f"CAE_{patch_size}x{patch_size}_"
            f"Fold{fold}_latent32_FINAL.pt"
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "patch_size":
                    patch_size,

                "fold":
                    fold,

                "latent_dim":
                    LATENT_DIM,

                "n_channels":
                    N_CHANNELS,

                "epochs":
                    EPOCHS,

                "seed":
                    run_seed,

                "train_mse":
                    train_mse,

                "val_mse":
                    val_mse
            },
            model_file
        )

        np.save(
            run_dir /
            "TRAIN_latent32.npy",
            Z_train
        )

        np.save(
            run_dir /
            "VALIDATION_latent32.npy",
            Z_val
        )

        np.save(
            run_dir /
            "TRAIN_k2_labels.npy",
            train_labels
        )

        np.save(
            run_dir /
            "VALIDATION_k2_labels.npy",
            val_labels
        )

        history_df = pd.DataFrame({
            "Epoch":
                np.arange(
                    1,
                    EPOCHS + 1
                ),

            "TRAIN_MSE":
                train_history
        })

        history_df.to_csv(
            run_dir /
            "training_history.csv",
            index=False
        )

        # ----------------------------------------------------
        # Record
        # ----------------------------------------------------

        records.append({

            "Patch_size":
                patch_size,

            "Validation_Fold":
                fold,

            "Seed":
                run_seed,

            "TRAIN_patches":
                len(X_train),

            "VALIDATION_patches":
                len(X_val),

            "Latent_dim":
                LATENT_DIM,

            "Epochs":
                EPOCHS,

            "TRAIN_reconstruction_MSE":
                train_mse,

            "VALIDATION_reconstruction_MSE":
                val_mse,

            "VAL_to_TRAIN_MSE_ratio":
                mse_ratio,

            "TRAIN_near_zero_latent_dims":
                train_near_zero_dims,

            "VALIDATION_near_zero_latent_dims":
                val_near_zero_dims,

            "KMeans_k":
                K,

            "Silhouette":
                sil,

            "Davies_Bouldin":
                dbi,

            "Calinski_Harabasz":
                ch,

            "TRAIN_cluster_min":
                int(
                    train_counts.min()
                ),

            "TRAIN_cluster_max":
                int(
                    train_counts.max()
                ),

            "VALIDATION_cluster_min":
                int(
                    val_counts.min()
                ),

            "VALIDATION_cluster_max":
                int(
                    val_counts.max()
                ),

            "Empty_validation_clusters":
                empty_val,

            "Training_seconds":
                training_seconds
        })

        print(
            f"  TRAIN MSE={train_mse:.4f} | "
            f"VAL MSE={val_mse:.4f} | "
            f"Ratio={mse_ratio:.2f} | "
            f"Sil={sil:.3f} | "
            f"DBI={dbi:.3f} | "
            f"VAL min={val_counts.min()} | "
            f"Empty VAL={empty_val}"
        )


# ============================================================
# SAVE FULL RESULTS
# ============================================================

results_df = pd.DataFrame(
    records
)

results_file = (
    FULL_DIR /
    "Paper1_CAE32_KMeans_k2_full_5fold_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)


# ============================================================
# PATCH-SIZE SUMMARY
# ============================================================

summary_df = (
    results_df
    .groupby(
        "Patch_size"
    )
    .agg(

        mean_train_MSE=(
            "TRAIN_reconstruction_MSE",
            "mean"
        ),

        mean_validation_MSE=(
            "VALIDATION_reconstruction_MSE",
            "mean"
        ),

        mean_MSE_ratio=(
            "VAL_to_TRAIN_MSE_ratio",
            "mean"
        ),

        mean_Silhouette=(
            "Silhouette",
            "mean"
        ),

        std_Silhouette=(
            "Silhouette",
            "std"
        ),

        mean_Davies_Bouldin=(
            "Davies_Bouldin",
            "mean"
        ),

        mean_Calinski_Harabasz=(
            "Calinski_Harabasz",
            "mean"
        ),

        min_validation_cluster=(
            "VALIDATION_cluster_min",
            "min"
        ),

        folds_with_empty_validation_clusters=(
            "Empty_validation_clusters",
            lambda x:
                int(
                    np.sum(
                        np.asarray(x) > 0
                    )
                )
        ),

        mean_training_seconds=(
            "Training_seconds",
            "mean"
        )
    )
    .reset_index()
)

summary_file = (
    FULL_DIR /
    "Paper1_CAE32_KMeans_k2_patchsize_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

elapsed_total = (
    time.time()
    - global_start
)

print("\n" + "=" * 78)
print("FULL CAE BASELINE COMPLETED")
print("=" * 78)

print("\nFull results:")
print(results_file)

print("\nPatch-size summary:")
print(summary_file)

print(
    "\nTotal runtime:",
    round(
        elapsed_total / 60,
        2
    ),
    "minutes"
)

print("\nSUMMARY")
print(summary_df.round(4).to_string(index=False))

print("=" * 78)