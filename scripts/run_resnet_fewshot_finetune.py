"""Run the fixed-support, two-stage ResNet-50 label-efficiency experiment."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fewshot_finetune import execute, prepare


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Validate existing outputs and resume unfinished runs from their initial weights")
    parser.add_argument("--run-ids", nargs="+", help="Run a named subset; omit on a later --resume to finish all supports")
    parser.add_argument("--smoke-test", action="store_true", help="One support, 1+1 rounds of two batches. Requires a separate output; never a formal result")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size >= 1 and num-workers >= 0 required")
    if args.smoke_test and args.run_ids:
        parser.error("--smoke-test cannot be combined with --run-ids")
    return args


def main(argv=None):
    args = parse_args(argv)
    frames, runs, protocol = prepare(args)
    print(f"Validated {len(runs)} fixed supports; mode={protocol['mode']}; output={args.output_root}", flush=True)
    if args.prepare_only:
        print("Prepared only; no model loaded or trained. Repeat with --resume to run.", flush=True)
        return
    execute(args, frames, runs, protocol)


if __name__ == "__main__":
    main()
