"""Run the paired ResNet-50 layer4 + classifier 5/10-shot experiment."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fewshot_partial import PROJECT_ROOT, main


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "run", "pack"))
    parser.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data_splits")
    parser.add_argument("--image-root", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--reference-root", type=Path, default=PROJECT_ROOT / "outputs/resnet_fewshot_suite_20261022")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/resnet_layer4_suite_20261022")
    args = parser.parse_args(argv)
    for name in ("splits_dir", "image_root", "reference_root", "output_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


if __name__ == "__main__":
    main(parse_args())
