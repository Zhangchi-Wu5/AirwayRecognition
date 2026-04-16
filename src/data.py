"""Data loading, manifest generation, and patient-level splitting."""
import re
from pathlib import Path
from typing import Optional

import pandas as pd


LABEL_TO_ID = {"lt": 0, "yz": 1, "zz": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
LABEL_NAMES_CN = {"lt": "隆突", "yz": "右总支气管", "zz": "左总支气管"}

_FILENAME_PATTERN = re.compile(r"^(\d+)\s*(lt|yz|zz)\.(png|jpg)$", re.IGNORECASE)


def parse_filename(filename: str) -> Optional[dict]:
    """Parse a bronchoscopy image filename into components.

    Returns None if the filename doesn't match the expected pattern.
    Pattern: {patient_id}[optional-space]{lt|yz|zz}.{png|jpg}
    """
    match = _FILENAME_PATTERN.match(filename.strip())
    if match is None:
        return None
    patient_id, label, ext = match.groups()
    return {
        "patient_id": patient_id,
        "label": label.lower(),
        "ext": ext.lower(),
    }


def build_manifest(dataset_dir: Path) -> pd.DataFrame:
    """Scan dataset_dir and build a manifest dataframe.

    Columns: patient_id, label, label_id, path
    Skips unparseable files with a warning.
    """
    dataset_dir = Path(dataset_dir)
    rows = []
    skipped = []
    for file_path in sorted(dataset_dir.iterdir()):
        if not file_path.is_file():
            continue
        parsed = parse_filename(file_path.name)
        if parsed is None:
            skipped.append(file_path.name)
            continue
        rows.append({
            "patient_id": parsed["patient_id"],
            "label": parsed["label"],
            "label_id": LABEL_TO_ID[parsed["label"]],
            "path": str(file_path.resolve()),
        })
    if skipped:
        print(f"[build_manifest] Skipped {len(skipped)} unparseable files:")
        for name in skipped[:10]:
            print(f"  - {name}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")
    return pd.DataFrame(rows)
