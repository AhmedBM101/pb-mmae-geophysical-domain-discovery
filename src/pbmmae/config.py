"""Configuration and invariant checks for the canonical PB-MMAE experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


CHANNEL_NAMES: tuple[str, ...] = (
    "RTE_TMI",
    "CBG", "CBG_RES", "HGM",
    "K", "eTh", "eU",
    "CPD",
    "MAG_LD", "GRAV_LD", "DEM_LD", "ID",
    "DEM", "SLOPE",
)

PHYSICS_FAMILIES: dict[str, tuple[str, ...]] = {
    "magnetic": ("RTE_TMI",),
    "gravity": ("CBG", "CBG_RES", "HGM"),
    "radiometric": ("K", "eTh", "eU"),
    "thermal": ("CPD",),
    "structural": ("MAG_LD", "GRAV_LD", "DEM_LD", "ID"),
    "terrain": ("DEM", "SLOPE"),
}


@dataclass(frozen=True)
class Paths:
    project_root: Path
    patch_root: Path
    output_root: Path


@dataclass(frozen=True)
class TrainSettings:
    patch_size: int = 13
    evaluation_folds: tuple[int, ...] = (1, 2, 4, 5)
    development_fold: int = 3
    branch_embed_dim: int = 16
    latent_dim: int = 64
    spatial_mask_ratio: float = 0.40
    modality_mask_probability: float = 0.30
    smooth_l1_beta: float = 1.0
    epochs: int = 300
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    base_seed: int = 42
    checkpoint_every: int = 25
    device: str = "cpu"
    num_threads: int = 2


@dataclass(frozen=True)
class Paper1Config:
    paths: Paths
    train: TrainSettings

    @property
    def model_root(self) -> Path:
        return self.paths.output_root / "models" / "PBMMAE" / "OuterFolds_13x13_LinearFusion64"


def _as_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in value)


def load_config(path: str | Path) -> Paper1Config:
    """Load a YAML experiment file and validate the frozen paper invariants."""
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency error
        raise RuntimeError("Install package dependencies before loading YAML configuration.") from exc

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    base_dir = config_path.parent.parent
    paths_raw = raw.get("paths", {})
    train_raw = raw.get("train", {})
    project_root = Path(paths_raw.get("project_root", "."))
    if not project_root.is_absolute():
        project_root = (base_dir / project_root).resolve()
    patch_root = Path(paths_raw.get("patch_root", "03_outputs/patches"))
    output_root = Path(paths_raw.get("output_root", "03_outputs"))
    if not patch_root.is_absolute():
        patch_root = (project_root / patch_root).resolve()
    if not output_root.is_absolute():
        output_root = (project_root / output_root).resolve()

    settings = TrainSettings(
        patch_size=int(train_raw.get("patch_size", 13)),
        evaluation_folds=_as_tuple(train_raw.get("evaluation_folds", [1, 2, 4, 5])),
        development_fold=int(train_raw.get("development_fold", 3)),
        branch_embed_dim=int(train_raw.get("branch_embed_dim", 16)),
        latent_dim=int(train_raw.get("latent_dim", 64)),
        spatial_mask_ratio=float(train_raw.get("spatial_mask_ratio", 0.40)),
        modality_mask_probability=float(train_raw.get("modality_mask_probability", 0.30)),
        smooth_l1_beta=float(train_raw.get("smooth_l1_beta", 1.0)),
        epochs=int(train_raw.get("epochs", 300)),
        batch_size=int(train_raw.get("batch_size", 32)),
        learning_rate=float(train_raw.get("learning_rate", 1e-3)),
        weight_decay=float(train_raw.get("weight_decay", 1e-4)),
        base_seed=int(train_raw.get("base_seed", 42)),
        checkpoint_every=int(train_raw.get("checkpoint_every", 25)),
        device=str(train_raw.get("device", "cpu")),
        num_threads=int(train_raw.get("num_threads", 2)),
    )
    _validate(settings)
    return Paper1Config(paths=Paths(project_root=project_root, patch_root=patch_root, output_root=output_root), train=settings)


def _validate(settings: TrainSettings) -> None:
    if settings.patch_size != 13:
        raise ValueError("The canonical paper configuration requires 13x13 patches.")
    if settings.development_fold in settings.evaluation_folds:
        raise ValueError("Development Fold 3 must not be included in outer-fold evaluation.")
    if settings.evaluation_folds != (1, 2, 4, 5):
        raise ValueError("Canonical outer folds must be exactly (1, 2, 4, 5).")
    if settings.branch_embed_dim != 16 or settings.latent_dim != 64:
        raise ValueError("The frozen canonical architecture is 6x16 branches with a 64-D latent representation.")
    if not 0 < settings.spatial_mask_ratio < 1:
        raise ValueError("spatial_mask_ratio must lie strictly between zero and one.")
    if not 0 <= settings.modality_mask_probability <= 1:
        raise ValueError("modality_mask_probability must lie between zero and one.")
    family_channels = tuple(channel for family in PHYSICS_FAMILIES.values() for channel in family)
    if family_channels != CHANNEL_NAMES:
        raise RuntimeError("Physics-family definitions no longer match the frozen 14-channel order.")
