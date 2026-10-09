"""Command-line interface for PB-MMAE reproducibility tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .runner import available_stage_names, run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pbmmae", description="PB-MMAE reproducibility workflow")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train-outer", help="Train the frozen canonical 13x13 outer folds")
    train.add_argument("--config", type=Path, required=True, help="YAML configuration file")
    inspect = commands.add_parser("preflight", help="Validate configuration and required canonical training tensors")
    inspect.add_argument("--config", type=Path, required=True, help="YAML configuration file")
    run = commands.add_parser("run", help="Run the canonical model-to-results workflow")
    run.add_argument("--config", type=Path, required=True, help="YAML configuration file")
    run.add_argument("--stages", nargs="+", choices=available_stage_names(), help="Ordered subset of stages to run")
    run.add_argument("--dry-run", action="store_true", help="Print the selected stages without executing them")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    required = [config.paths.patch_root / "13x13" / f"Fold_{fold}" / f"Fold{fold}_TRAIN_13x13_14ch.npy" for fold in config.train.evaluation_folds]
    missing = [path for path in required if not path.exists()]
    if args.command == "preflight":
        if missing:
            parser.error("Missing canonical training tensors:\n" + "\n".join(str(path) for path in missing))
        print("Preflight passed. Canonical PB-MMAE training tensors are available.")
        return 0
    if args.command == "run":
        if missing and not args.dry_run and (not args.stages or "train" in args.stages):
            parser.error("Cannot run training; missing canonical training tensors:\n" + "\n".join(str(path) for path in missing))
        executed = run_pipeline(config, stages=args.stages, dry_run=args.dry_run)
        print(f"Pipeline {'planned' if args.dry_run else 'completed'}: {', '.join(executed)}")
        return 0
    if missing:
        parser.error("Cannot train; missing canonical training tensors:\n" + "\n".join(str(path) for path in missing))
    from .training import train_outer_folds
    summary = train_outer_folds(config)
    print(f"Canonical outer-fold training completed: {summary}")
    return 0


if __name__ == "__main__":  # pragma: no cover - console-script equivalent
    raise SystemExit(main())
