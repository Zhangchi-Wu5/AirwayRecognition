"""Run the isolated renew ResNet-50 development cross-validation experiment."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.renew_experiment import PROJECT_ROOT, main


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "preview", "run", "pack"))
    parser.add_argument("--image-root", type=Path, default=PROJECT_ROOT / "dataset/renew")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/renew_resnet_cv")
    parser.add_argument("--phase", choices=("full", "fewshot", "all"), default="full")
    args = parser.parse_args(argv)
    for name in ("image_root", "output_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


if __name__ == "__main__":
    main(parse_args())
