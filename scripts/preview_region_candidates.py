"""Audit label-free candidate regions on train/val with exact evaluation geometry.

Dark regions are algorithm candidates, not annotated or verified anatomical ROIs.
All split images are counted; visual examples use fixed identity ordering, never
candidate quality or model correctness. Test images cannot be selected.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from scripts.evaluate_region_relations import (
    CLASSES, PROJECT_ROOT, membership, read_identity_splits, resolve_split_images, require, sha256_file,
)
from src.data import get_eval_geometry_transforms

COLORS = ((255, 90, 60), (40, 220, 100), (75, 155, 255), (235, 90, 230), (255, 210, 30))


def mask_boundary(mask):
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, 1, constant_values=False)
    eroded = (padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
    return mask & ~eroded


def render_candidate_overlay(image, candidates):
    """Tint interiors strongly, rings lightly; draw candidate boundaries."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    valid = np.asarray(candidates["valid"], dtype=bool)
    h, w = rgb.shape[:2]
    for idx in np.flatnonzero(valid):
        color = np.asarray(COLORS[int(idx) % len(COLORS)], dtype=np.float32)
        masks = []
        for field in ("inner_masks", "ring_masks"):
            source = Image.fromarray((np.asarray(candidates[field][idx]) > .5).astype(np.uint8) * 255)
            masks.append(np.asarray(source.resize((w, h), Image.Resampling.NEAREST)) > 0)
        inner, ring = masks
        rgb[ring] = .82 * rgb[ring] + .18 * color
        rgb[inner] = .52 * rgb[inner] + .48 * color
        rgb[mask_boundary(inner)] = color
        rgb[mask_boundary(ring)] = .45 * color + .55 * 255
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def preview_tile(image, candidates, caption):
    size = 224
    tile = Image.new("RGB", (2 * size + 16, size + 70), "#161b22")
    tile.paste(image.resize((size, size)), (0, 25))
    tile.paste(render_candidate_overlay(image, candidates).resize((size, size)), (size + 16, 25))
    draw = ImageDraw.Draw(tile)
    draw.text((5, 5), "Model input", fill="white")
    draw.text((size + 20, 5), "Candidate + surrounding ring", fill="white")
    draw.text((5, size + 29), caption, fill="white")
    count = int(np.asarray(candidates["valid"]).sum())
    draw.text((5, size + 46), f"{count} candidate(s)" if count else "NO VALID CANDIDATE: global branch only", fill="#ffcf66" if not count else "#b8c8d8")
    return tile


def choose_examples(frame, per_class):
    """Deterministic label/identity selection independent of extraction outputs."""
    selected = []
    for label in CLASSES:
        part = frame[frame.label == label].sort_values(["patient_id", "image_name"], kind="stable")
        selected.extend(part.index[:per_class].tolist())
    return selected


def run(args):
    from src.region_candidates import CandidateConfig, extract_candidates
    output = args.output_dir.expanduser().resolve()
    require(not output.exists() or not any(output.iterdir()), "Output directory must be absent or empty")
    raw_config = {}
    if args.candidate_config:
        raw_config = json.loads(args.candidate_config.read_text(encoding="utf-8"))
        if "candidate_config" in raw_config:
            raw_config = raw_config["candidate_config"]
    if args.max_candidates is not None:
        raw_config["max_candidates"] = args.max_candidates
    config = CandidateConfig(**raw_config)
    frames = read_identity_splits(args.splits_dir)
    frame = resolve_split_images(frames, args.split, args.image_root)
    selected = choose_examples(frame, args.per_class)
    selected_set = set(selected)
    tiles, rows = {}, []
    geometry = get_eval_geometry_transforms(crop_border=False)
    for index, row in frame.iterrows():
        with Image.open(row.path) as source:
            image = geometry(source.convert("RGB"))
        candidates = extract_candidates(image, config)
        valid = np.asarray(candidates["valid"], dtype=bool)
        count = int(valid.sum())
        quality = np.asarray(candidates["quality"], dtype=float)
        fov = np.asarray(candidates["fov_mask"], dtype=bool)
        record = {"patient_id": row.patient_id, "image_name": row.image_name, "label": row.label,
                  "candidate_count": count, "has_two_candidates": int(count >= 2),
                  "mean_valid_quality": float(quality[valid].mean()) if count else 0.,
                  "fov_fraction": float(fov.mean()), "selected_for_preview": int(index in selected_set)}
        for slot in range(len(valid)):
            record[f"candidate_{slot}_valid"] = int(valid[slot])
            record[f"candidate_{slot}_quality"] = float(quality[slot])
            record[f"candidate_{slot}_area_fraction"] = float(np.asarray(candidates["inner_masks"])[slot].mean())
            record[f"candidate_{slot}_ring_fraction"] = float(np.asarray(candidates["ring_masks"])[slot].mean())
        rows.append(record)
        if index in selected_set:
            tiles[index] = preview_tile(image, candidates, f"{row.image_name} | class: {row.label}")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "candidate_summary.csv", index=False)
    n_columns = min(3, max(1, len(selected)))
    tile_width, tile_height = 464, 294
    n_rows = (len(selected) + n_columns - 1) // n_columns
    sheet = Image.new("RGB", (n_columns * tile_width, 75 + n_rows * tile_height), "#10141a")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), f"Algorithm candidates / split={args.split} / fixed first {args.per_class} identities per class", fill="white")
    draw.text((10, 29), "Strong tint: interior | light tint: surrounding ring | bright lines: mask boundaries", fill="#c8d7e6")
    draw.text((10, 48), "Resize(short side=256) + center crop(224): exact model evaluation view. NOT anatomical annotations.", fill="#ffcf66")
    for number, index in enumerate(selected):
        sheet.paste(tiles[index], ((number % n_columns) * tile_width, 75 + (number // n_columns) * tile_height))
    sheet.save(output / "candidate_contact_sheet.png")
    counts = pd.Series([row["candidate_count"] for row in rows]).value_counts().sort_index()
    metadata = {"candidate_config": asdict(config), "split": args.split, "test_images_loaded": False,
                "n_images_audited": len(frame), "n_patients": int(frame.patient_id.nunique()),
                "candidate_count_distribution": {str(key): int(value) for key, value in counts.items()},
                "n_images_with_two_candidates": int(sum(row["has_two_candidates"] for row in rows)),
                "geometry": "get_eval_geometry_transforms(crop_border=False): Resize(256), CenterCrop(224)",
                "selection": "first patient_id/image_name lexical identities per class; independent of candidate quality",
                "selected_images": frame.loc[selected, "image_name"].tolist(),
                "split_membership_sha256": {name: membership(data)[1] for name, data in frames.items()},
                "source_sha256": {name: sha256_file(PROJECT_ROOT / name) for name in (
                    "src/region_candidates.py", "src/data.py", "scripts/preview_region_candidates.py",
                    "scripts/evaluate_region_relations.py")}}
    (output / "preview_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# 候选区域预览\n\n"
        f"已审计 {args.split} 的全部 {len(frame)} 张图片；candidate_summary.csv 包含每张图片的计数。\n\n"
        "预览按类别、患者标识和文件名固定选取，不依赖候选好坏。强色填充为候选内部，浅色为周围环带，亮线为边界。"
        "没有候选的图片明确标记，并由模型全局分支处理。\n\n"
        "显示图经过短边缩放至 256 后中心裁剪至 224，与本轮模型验证输入一致；并非完整原始视野。"
        "这些区域是暗区启发式产生的候选，可能包含阴影、遮挡或其他组织，不是医生确认的管腔、分叉或新解剖 ROI。"
        "未加载测试集图片；未增加任何人工标注。\n", encoding="utf-8")
    print(f"Candidate audit: {len(frame)} images; counts={dict(counts)}; output={output}")
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument("--candidate-config", type=Path)
    parser.add_argument("--max-candidates", type=int)
    args = parser.parse_args(argv)
    if args.per_class < 1 or (args.max_candidates is not None and args.max_candidates < 1):
        parser.error("per-class and max-candidates must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
