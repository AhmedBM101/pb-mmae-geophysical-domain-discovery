"""Stage runner for the frozen PB-MMAE paper workflow.

The original post-processing scripts are retained apart from their
project-root parameter. This runner gives them one controlled entry point.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from .config import Paper1Config


STAGES: tuple[tuple[str, str | None], ...] = (
    ("train", None),
    ("evaluate", "evaluate_pbmmae_linear64_outer_folds_13x13.py"),
    ("consensus", "consensus_pbmmae_linear64_outer_folds_k8_k9.py"),
    ("assemble-labels", "assemble_k9_geographic_labels.py"),
    ("align", "align_k9_outerfold_labels_hungarian.py"),
    ("derive-metadomains", "derive_k9_metadomains_from_crossfold_confusion.py"),
    ("build-maps", "build_pbmmae_4domain_consensus_maps.py"),
    ("characterize", "characterize_pbmmae_4domains_physical_signatures.py"),
    ("prepare-external", "prepare_external_validation_master_table.py"),
    ("validate-geology", "validate_geology_domain_association.py"),
    ("validate-gpi", "validate_GPI_domain_association.py"),
    ("validate-springs", "validate_warm_spring_domain_association.py"),
    ("synthesize-external", "synthesize_external_validation.py"),
    ("lock-results", "build_final_quantitative_results_lock.py"),
)


def available_stage_names() -> tuple[str, ...]:
    return tuple(name for name, _ in STAGES)


def _select_stages(requested: Iterable[str] | None) -> tuple[tuple[str, str | None], ...]:
    if not requested:
        return STAGES
    stage_map = dict(STAGES)
    unknown = [name for name in requested if name not in stage_map]
    if unknown:
        raise ValueError(f"Unknown stage(s): {', '.join(unknown)}. Available: {', '.join(available_stage_names())}")
    return tuple((name, stage_map[name]) for name in requested)


def run_pipeline(config: Paper1Config, stages: Iterable[str] | None = None, dry_run: bool = False) -> list[str]:
    """Run selected stages in their supplied order.

    ``project_root`` is the authorised local analysis tree containing
    ``02_data`` and ``03_outputs``; it can live outside this repository.
    """
    selected = _select_stages(stages)
    workflow_root = Path(__file__).resolve().parent / "workflow"
    environment = os.environ.copy()
    environment["PBMMAE_PROJECT_ROOT"] = str(config.paths.project_root)
    executed: list[str] = []
    for stage, filename in selected:
        command = ["pbmmae", "train-outer", "--config", "<configuration>"]
        if stage != "train":
            assert filename is not None
            script = workflow_root / filename
            if not script.exists():
                raise FileNotFoundError(f"Missing packaged workflow stage: {script}")
            command = [sys.executable, str(script)]
        print(f"[{stage}] {' '.join(command)}")
        if not dry_run:
            if stage == "train":
                from .training import train_outer_folds
                train_outer_folds(config)
            else:
                subprocess.run(command, check=True, cwd=config.paths.project_root, env=environment)
        executed.append(stage)
    return executed
