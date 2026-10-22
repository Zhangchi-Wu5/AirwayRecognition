"""Screen fixed-class low-label performance with frozen ImageNet ResNet-50.

Example: python -m scripts.run_resnet_fewshot --splits-dir data_splits \
    --image-root dataset --output-root outputs/fewshot_screening
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fewshot import PROJECT_ROOT, execute_experiment, prepare_experiment


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data_splits")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--sampling-seed", type=int, default=20261022)
    parser.add_argument("--prepare-only", action="store_true", help="Validate identities and generate supports without importing torch or fitting models")
    args = parser.parse_args(argv)
    if len(set(args.shots)) != len(args.shots) or any(k < 1 for k in args.shots):
        parser.error("shots must be distinct positive integers")
    if args.repeats < 2 or args.sampling_seed < 0 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("repeats >= 2, sampling-seed >= 0, batch-size >= 1, num-workers >= 0 required")
    return args


def main(argv=None):
    args = parse_args(argv)
    frames, runs, protocol = prepare_experiment(args)
    print(f"Prepared {len(runs)} runs ({len(runs) - 1} support draws + full reference once): {args.output_root}", flush=True)
    if args.prepare_only:
        print("Plan only: no features extracted and no performance result.", flush=True)
        return
    execute_experiment(args, frames, runs, protocol)
    print(f"Completed. Report: {Path(args.output_root).resolve() / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
