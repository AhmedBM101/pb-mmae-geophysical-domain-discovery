"""Frozen PB-MMAE LinearFusion64 architecture used for outer-fold evaluation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .config import CHANNEL_NAMES, PHYSICS_FAMILIES


FAMILY_NAMES = tuple(PHYSICS_FAMILIES)
FAMILY_INDICES = {
    family: [CHANNEL_NAMES.index(channel) for channel in channels]
    for family, channels in PHYSICS_FAMILIES.items()
}


class FamilyEncoder(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.fc = nn.Linear(32 * 3 * 3, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(self.features(x), start_dim=1))


class FamilyDecoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, 32 * 3 * 3)
        self.conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        x = self.fc(z).view(-1, 32, 3, 3)
        x = functional.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return self.conv(x)


class PBMMAELinear64(nn.Module):
    """Six-branch PB-MMAE with LayerNorm and a single 96→64 fusion layer."""

    def __init__(self, branch_embed_dim: int = 16, latent_dim: int = 64) -> None:
        super().__init__()
        self.encoders = nn.ModuleDict({
            family: FamilyEncoder(len(channels), branch_embed_dim)
            for family, channels in PHYSICS_FAMILIES.items()
        })
        self.branch_norms = nn.ModuleDict({
            family: nn.LayerNorm(branch_embed_dim) for family in FAMILY_NAMES
        })
        self.fusion = nn.Linear(len(FAMILY_NAMES) * branch_embed_dim, latent_dim)
        self.decoders = nn.ModuleDict({
            family: FamilyDecoder(latent_dim, len(channels))
            for family, channels in PHYSICS_FAMILIES.items()
        })

    def encode(self, family_inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        raw_embeddings = [self.encoders[family](family_inputs[family]) for family in FAMILY_NAMES]
        normalized_embeddings = [self.branch_norms[family](embedding) for family, embedding in zip(FAMILY_NAMES, raw_embeddings)]
        return self.fusion(torch.cat(normalized_embeddings, dim=1)), raw_embeddings, normalized_embeddings

    def forward(self, family_inputs: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        first_family = FAMILY_NAMES[0]
        output_size = family_inputs[first_family].shape[-2:]
        fused, raw_embeddings, normalized_embeddings = self.encode(family_inputs)
        reconstructions = {family: self.decoders[family](fused, output_size) for family in FAMILY_NAMES}
        return reconstructions, fused, raw_embeddings, normalized_embeddings


def split_into_families(x: torch.Tensor) -> dict[str, torch.Tensor]:
    return {family: x[:, indices, :, :] for family, indices in FAMILY_INDICES.items()}
