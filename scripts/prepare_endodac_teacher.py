"""Create immutable per-image targets from a locally installed frozen EndoDAC."""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision.transforms import functional as TF

from src.geometry_distillation import (VERSION, cache_predictions, file_hash,
                                       atomic_json, GeometryImageDataset, relation_loss)
from src.renew_data import audit_dataset

PINNED_COMMIT = "22ec911b270d056fdd7f430474e4be407cce8287"


def support_rows(args):
    rows, metadata = audit_dataset(args.image_root)
    by_name = {r["image_name"]: r for r in rows}
    if args.support_csv:
        with Path(args.support_csv).open(newline="", encoding="utf-8-sig") as f:
            selected = list(csv.DictReader(f))
        if not selected or len({r["image_name"] for r in selected}) != len(selected):
            raise ValueError("Support CSV must contain distinct image identities")
        for row in selected:
            if row["image_name"] not in by_name or row.get("image_sha256") != by_name[row["image_name"]]["image_sha256"]:
                raise ValueError("Support CSV does not match audited source images")
        names = [r["image_name"] for r in selected]
    else:
        suite = json.loads((Path(args.suite_root) / "suite.json").read_text(encoding="utf-8"))
        identity = suite["identity"]
        # Use the existing digest convention, not the cache-specific hash.
        from src.fewshot_finetune import _digest
        if suite.get("identity_sha256") != _digest(identity):
            raise ValueError("Suite identity checksum mismatch")
        jobs = [j for j in identity["jobs"] if str(j["budget"]) in ("5", "10")]
        if args.scope == "qa":
            jobs = [j for j in jobs if str(j["budget"]) == "5" and int(j["fold"]) == 0
                    and int(j.get("repeat", 0)) == 0][:1]
        if not jobs:
            raise ValueError("Suite has no matching few-shot support jobs")
        for job in jobs:
            if set(job["support"]) & set(job["eval"]):
                raise ValueError("Support/evaluation overlap in suite")
        names = sorted({name for j in jobs for name in j["support"]})
        declared = {r["image_name"]: r["image_sha256"] for r in identity["data"]}
        if any(name not in by_name or declared.get(name) != by_name[name]["image_sha256"] for name in names):
            raise ValueError("Suite source hashes differ from audited source")
    # Frozen, independent target generation over the union is allowed; the
    # training loader later permits each run to read only its own support.
    return [by_name[name] for name in names], metadata


def load_tensor_checkpoint(path):
    try:
        result = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError("A PyTorch version supporting weights_only=True is required") from error
    if not isinstance(result, dict):
        raise ValueError("Expected a tensor state dictionary")
    return result


def load_teacher(repo, weights, backbone_weights, device):
    repo, weights = Path(repo).resolve(), Path(weights).resolve()
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != PINNED_COMMIT:
        raise ValueError(f"Teacher checkout must be pinned to {PINNED_COMMIT}")
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty.strip():
        raise ValueError("Teacher tracked source has local modifications")
    source_files = sorted(repo.glob("models/**/*.py"))
    if not source_files:
        raise ValueError("Teacher repository contains no model source files")
    source_hashes = {str(p.relative_to(repo)): file_hash(p) for p in source_files}
    os.environ["XFORMERS_DISABLED"] = "1"
    if "models" in sys.modules:
        raise RuntimeError("Run teacher preparation in a fresh Python process (models namespace already loaded)")
    sys.path.insert(0, str(repo))
    module = importlib.import_module("models.endodac")
    model = module.endodac(backbone_size="base", r=4, lora_type="dvlora",
        image_shape=(224, 280), pretrained_path=None,
        residual_block_indexes=[2, 5, 8, 11], include_cls_token=True, use_cls_token=False)
    checkpoint = load_tensor_checkpoint(weights)
    height, width = checkpoint.get("height"), checkpoint.get("width")
    if not isinstance(height, int) or not isinstance(width, int) or not (1 <= min(height, width) <= max(height, width) <= 4096):
        raise ValueError("Teacher checkpoint must declare valid integer height/width")
    model_state = {k: v for k, v in checkpoint.items() if torch.is_tensor(v)}
    if set(checkpoint) - set(model_state) - {"height", "width", "use_stereo"}:
        raise ValueError("Unexpected non-tensor metadata in teacher checkpoint")
    # Complete trained state is required. Unlike the upstream strict=False
    # route, no missing layers are silently left with random initialization.
    model.load_state_dict(model_state, strict=True)
    backbone_hash = None
    if backbone_weights:
        # A complete checkpoint supersedes the optional initialization file.
        # Verify the supplied artifact but do not execute/load it unnecessarily.
        backbone_hash = file_hash(backbone_weights)
    model.eval().requires_grad_(False).to(device)
    identity = {"kind": "EndoDAC_full_checkpoint", "commit": head,
        "source_sha256": source_hashes, "checkpoint_sha256": file_hash(weights),
        "backbone_sha256": backbone_hash, "backbone_initialization_used": False,
        "strict_complete_state": True, "feed_size": [height, width],
        "internal_image_shape": [224, 280], "dtype": "float32",
        "torch_version": str(torch.__version__), "device_type": str(device).split(":")[0],
        "xformers_disabled": True, "output": "sigmoid disparity; larger is near proxy, not metric depth",
        "cache_code_sha256": file_hash(Path(__file__).parents[1] / "src/geometry_distillation.py"),
        "preparation_code_sha256": file_hash(__file__)}

    def predict(image):
        tensor = TF.to_tensor(image.resize((width, height), resample=3)).unsqueeze(0).to(device)
        with torch.inference_mode():
            value = model(tensor)[("disp", 0)]
        if value.ndim != 4 or tuple(value.shape[:2]) != (1, 1):
            raise ValueError("Unexpected EndoDAC disparity output shape")
        return value[0, 0].float().cpu().numpy()
    return predict, identity


def write_qa(rows, cache_root):
    """Private contact sheet: appearance/coordinate check, not depth validation."""
    root = Path(cache_root)
    dataset = GeometryImageDataset(rows, root, train=False)
    columns, tile_width, tile_height = 3, 330, 134
    sheet = Image.new("RGB", (columns * tile_width, 40 + ((len(rows) + columns - 1) // columns) * tile_height), "white")
    drawing = ImageDraw.Draw(sheet)
    drawing.text((8, 8), "PRIVATE QA: image | predicted disparity (brighter = nearer proxy, NOT measured depth)", fill="black")
    records = []
    for index, row in enumerate(rows):
        sample = dataset[index]
        with np.load(root / f"{row['image_sha256']}.npz", allow_pickle=False) as cache:
            disparity = cache["disparity"][0]
        low, high = np.percentile(disparity, [2, 98])
        gray = ((disparity - low) / max(float(high - low), 1e-8) * 255).clip(0, 255).astype(np.uint8)
        with Image.open(row["path"]) as opened:
            source = opened.convert("RGB")
        proxy = Image.fromarray(gray, mode="L").convert("RGB")
        x, y = (index % columns) * tile_width + 5, 40 + (index // columns) * tile_height
        for offset, image in ((0, source), (165, proxy)):
            image.thumbnail((160, 96), Image.Resampling.BILINEAR)
            sheet.paste(image, (x + offset, y))
        _, stats = relation_loss(torch.zeros(1, 1, 14, 14), sample["geometry"][None],
                                 sample["valid"][None], variant="stable")
        drawing.text((x, y + 99), f"#{index + 1:02d} range {float(disparity.min()):.3f}-{float(disparity.max()):.3f}", fill="black")
        drawing.text((x, y + 114), f"Stable pair fraction: {stats['selection_rate']:.3f}", fill="black")
        records.append({"qa_index": index + 1, "image_sha256": row["image_sha256"],
                        "range": [float(disparity.min()), float(disparity.max())], **stats})
    path = root / "QA_CONTACT.png"
    temporary = root / "QA_CONTACT.tmp.png"
    sheet.save(temporary)
    temporary.replace(path)
    atomic_json(root / "QA_STATS.json", {"meaning": "engineering sanity only; stability is not depth accuracy",
                                         "records": records, "contact_sha256": file_hash(path)})
    return str(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--suite-root", help="Prepared geometry suite with suite.json")
    source.add_argument("--support-csv", help="Explicit image_name,image_sha256 support whitelist")
    parser.add_argument("--scope", choices=("qa", "all-supports"), default="qa",
                        help="qa uses fold0 5-shot repeat0; all-supports caches frozen per-image support union")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--teacher-repo", required=True)
    parser.add_argument("--teacher-weights", required=True, help="Complete depth_model.pth FILE")
    parser.add_argument("--backbone-weights", help="Optional initialization FILE, fingerprinted but superseded by complete checkpoint")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu explicitly for teacher-only checks")
    rows, data = support_rows(args)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    predict, identity = load_teacher(args.teacher_repo, args.teacher_weights,
                                     args.backbone_weights, torch.device(args.device))
    print(f"Preparing {len(rows)} frozen per-image targets; no training or labels used by teacher", flush=True)
    summary = cache_predictions(rows, args.cache_root, predict, identity)
    contact = write_qa(rows, args.cache_root) if args.scope == "qa" else None
    report = {"status": "cache_integrity_passed_not_depth_accuracy_validated", "scope": args.scope,
              "images": len(rows), "teacher_sha256": summary["teacher_sha256"],
              "qa_contact": contact,
              "selected_cache_sha256": summary["identity_sha256"],
              "dataset_fingerprint": data["dataset_fingerprint"] if "dataset_fingerprint" in data else data.get("fingerprint")}
    atomic_json(Path(args.cache_root) / f"PREPARATION_{args.scope}.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
