"""Generate manifest.csv from the real dataset."""
from pathlib import Path
from src.data import build_manifest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_CSV = PROJECT_ROOT / "data_splits" / "manifest.csv"

df = build_manifest(DATASET_DIR)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Manifest saved to {OUTPUT_CSV}")
print(f"Total rows: {len(df)}")
print(f"Unique patients: {df['patient_id'].nunique()}")
print(f"Label distribution:\n{df['label'].value_counts()}")
