"""Deterministic canonical PB-MMAE outer-fold training."""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset

from .config import Paper1Config
from .model import FAMILY_INDICES, FAMILY_NAMES, PBMMAELinear64, split_into_families


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def apply_batch_masks(x: torch.Tensor, spatial_mask_ratio: float, modality_mask_probability: float) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Apply exactly one independent spatial mask per family and at most one whole-family mask."""
    batch_size, _, height, width = x.shape
    masked_x = x.clone()
    n_spatial_mask = int(round(height * width * spatial_mask_ratio))
    spatial_masks = {family: torch.zeros(batch_size, height, width, dtype=torch.bool, device=x.device) for family in FAMILY_NAMES}
    whole_family_masks = {family: torch.zeros(batch_size, dtype=torch.bool, device=x.device) for family in FAMILY_NAMES}

    for batch_index in range(batch_size):
        whole_family = None
        if torch.rand(1, device=x.device).item() < modality_mask_probability:
            whole_family = FAMILY_NAMES[torch.randint(0, len(FAMILY_NAMES), (1,), device=x.device).item()]
            whole_family_masks[whole_family][batch_index] = True
        for family in FAMILY_NAMES:
            indices = FAMILY_INDICES[family]
            if family == whole_family:
                masked_x[batch_index, indices, :, :] = 0.0
                spatial_masks[family][batch_index] = True
                continue
            positions = torch.randperm(height * width, device=x.device)[:n_spatial_mask]
            family_mask = torch.zeros(height * width, dtype=torch.bool, device=x.device)
            family_mask[positions] = True
            family_mask = family_mask.view(height, width)
            spatial_masks[family][batch_index] = family_mask
            for channel_index in indices:
                masked_x[batch_index, channel_index][family_mask] = 0.0
    return masked_x, spatial_masks, whole_family_masks


def physics_balanced_masked_loss(reconstructions: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], spatial_masks: dict[str, torch.Tensor], beta: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    family_weight = 1.0 / len(FAMILY_NAMES)
    losses: dict[str, torch.Tensor] = {}
    total = torch.zeros((), device=next(iter(reconstructions.values())).device)
    for family in FAMILY_NAMES:
        prediction = reconstructions[family]
        target = targets[family]
        mask = spatial_masks[family].unsqueeze(1).expand_as(prediction)
        if not torch.any(mask):
            raise RuntimeError(f"No masked positions available for family {family}.")
        loss = functional.smooth_l1_loss(prediction[mask], target[mask], beta=beta, reduction="mean")
        losses[family] = loss
        total = total + family_weight * loss
    return total, losses


def _checkpoint_payload(model: PBMMAELinear64, optimizer: torch.optim.Optimizer, fold: int, epoch: int, config: Paper1Config) -> dict[str, object]:
    settings = config.train
    return {
        "fold": fold, "epoch": epoch, "seed": settings.base_seed + fold,
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "architecture_frozen": True, "development_fold": settings.development_fold,
        "branch_embed_dim": settings.branch_embed_dim, "branch_layernorm": True,
        "fusion_type": "single_linear_projection", "fusion_input_dim": 96,
        "fused_latent_dim": settings.latent_dim, "spatial_mask_ratio": settings.spatial_mask_ratio,
        "modality_mask_prob": settings.modality_mask_probability, "smooth_l1_beta": settings.smooth_l1_beta,
        "epochs": settings.epochs, "batch_size": settings.batch_size,
        "learning_rate": settings.learning_rate, "weight_decay": settings.weight_decay,
    }


def _training_tensor(config: Paper1Config, fold: int) -> torch.Tensor:
    filename = config.paths.patch_root / "13x13" / f"Fold_{fold}" / f"Fold{fold}_TRAIN_13x13_14ch.npy"
    if not filename.exists():
        raise FileNotFoundError(f"Required training tensor not found: {filename}")
    array = np.load(filename).astype(np.float32)
    if array.ndim != 4 or array.shape[1:] != (13, 13, 14):
        raise ValueError(f"Expected NHWC tensor (*, 13, 13, 14); received {array.shape} from {filename}")
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite values found in {filename}")
    return torch.from_numpy(np.transpose(array, (0, 3, 1, 2)))


def train_fold(config: Paper1Config, fold: int) -> dict[str, float | int]:
    settings = config.train
    set_seed(settings.base_seed + fold)
    torch.set_num_threads(settings.num_threads)
    device = torch.device(settings.device)
    train_tensor = _training_tensor(config, fold)  # Deliberately never loads validation patches.
    loader_generator = torch.Generator().manual_seed(settings.base_seed + fold)
    loader = DataLoader(TensorDataset(train_tensor), batch_size=settings.batch_size, shuffle=True, generator=loader_generator, num_workers=0, pin_memory=False)
    model = PBMMAELinear64(settings.branch_embed_dim, settings.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    fold_dir = config.model_root / f"Fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    started = time.time()

    for epoch in range(1, settings.epochs + 1):
        total_loss = 0.0
        family_totals = {family: 0.0 for family in FAMILY_NAMES}
        total_samples = 0
        whole_masks = 0
        model.train()
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            masked_x, spatial_masks, whole_family_masks = apply_batch_masks(batch_x, settings.spatial_mask_ratio, settings.modality_mask_probability)
            optimizer.zero_grad(set_to_none=True)
            reconstructions, _, _, _ = model(split_into_families(masked_x))
            loss, family_losses = physics_balanced_masked_loss(reconstructions, split_into_families(batch_x), spatial_masks, settings.smooth_l1_beta)
            loss.backward()
            optimizer.step()
            batch_n = batch_x.size(0)
            total_loss += loss.item() * batch_n
            total_samples += batch_n
            whole_masks += sum(int(mask.sum().item()) for mask in whole_family_masks.values())
            for family in FAMILY_NAMES:
                family_totals[family] += family_losses[family].item() * batch_n
        record: dict[str, float | int] = {
            "Fold": fold, "Epoch": epoch, "Seed": settings.base_seed + fold,
            "Total_PB_Loss": total_loss / total_samples,
            "Whole_modality_fraction": whole_masks / total_samples,
        }
        record.update({f"{family}_loss": family_totals[family] / total_samples for family in FAMILY_NAMES})
        history.append(record)
        if epoch % settings.checkpoint_every == 0:
            torch.save(_checkpoint_payload(model, optimizer, fold, epoch, config), fold_dir / f"PBMMAE_LINEAR64_Fold{fold}_13x13_epoch{epoch:03d}.pt")

    history_df = pd.DataFrame(history)
    history_df.to_csv(fold_dir / f"PBMMAE_LINEAR64_Fold{fold}_13x13_training_history.csv", index=False)
    elapsed = time.time() - started
    payload = _checkpoint_payload(model, optimizer, fold, settings.epochs, config)
    payload.update({
        "patch_size": settings.patch_size, "n_channels": 14, "physics_families": len(FAMILY_NAMES),
        "final_total_pb_loss": float(history_df["Total_PB_Loss"].iloc[-1]),
        "minimum_total_pb_loss": float(history_df["Total_PB_Loss"].min()),
        "minimum_loss_epoch": int(history_df.loc[history_df["Total_PB_Loss"].idxmin(), "Epoch"]),
        "training_seconds": elapsed,
    })
    torch.save(payload, fold_dir / f"PBMMAE_LINEAR64_Fold{fold}_13x13_FINAL.pt")
    return {
        "Fold": fold, "Seed": settings.base_seed + fold, "TRAIN_patches": len(train_tensor),
        "Final_PB_loss": float(history_df["Total_PB_Loss"].iloc[-1]),
        "Minimum_PB_loss": float(history_df["Total_PB_Loss"].min()),
        "Minimum_loss_epoch": int(history_df.loc[history_df["Total_PB_Loss"].idxmin(), "Epoch"]),
        "Final25_mean_PB_loss": float(history_df["Total_PB_Loss"].tail(25).mean()),
        "Mean_whole_modality_fraction": float(history_df["Whole_modality_fraction"].mean()),
        "Training_seconds": elapsed,
    }


def train_outer_folds(config: Paper1Config) -> Path:
    """Train canonical folds 1, 2, 4, and 5, then return the aggregate summary path."""
    config.model_root.mkdir(parents=True, exist_ok=True)
    records = [train_fold(config, fold) for fold in config.train.evaluation_folds]
    summary = config.model_root / "PBMMAE_LINEAR64_outer_folds_training_summary.csv"
    pd.DataFrame(records).to_csv(summary, index=False)
    return summary
