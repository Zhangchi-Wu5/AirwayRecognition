"""Reproduce the historical full-data baseline before selecting few-shot methods."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline_reproduction import PROJECT_ROOT, main


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "run", "pack", "_worker"))
    parser.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data_splits")
    parser.add_argument("--historical-splits-dir", type=Path,
                        help="Original historical CSVs if retained; defaults to --splits-dir")
    parser.add_argument("--image-root", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--reference-root", type=Path, default=PROJECT_ROOT / "outputs/resnet_fewshot_suite_20261022")
    parser.add_argument("--historical-reference", type=Path, default=PROJECT_ROOT / "configs/resnet_baseline_reference_20261022.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/resnet_baseline_reproduction_20261022")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--job", choices=("historical_42", "historical_2026", "historical_3407", "current_2026", "current_3407"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.num_workers < 0:
        parser.error("num-workers must be nonnegative")
    if (args.mode == "_worker") != (args.job is not None):
        parser.error("--job is required only for the internal _worker mode")
    args.historical_splits_dir = args.historical_splits_dir or args.splits_dir
    for name in ("splits_dir", "historical_splits_dir", "image_root", "reference_root", "historical_reference", "output_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


if __name__ == "__main__":
    main(parse_args())
