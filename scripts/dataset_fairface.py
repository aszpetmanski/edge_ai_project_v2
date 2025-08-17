#!/usr/bin/env python3
from pathlib import Path
import shutil
import kagglehub

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "fairface" / "FairFace"

def count_images(d: Path) -> int:
    if not d.exists(): return 0
    return sum(1 for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

def find_fairface_root(download_root: Path) -> Path | None:
    # Look for the folder that has train/ val/ and CSVs
    for csv in download_root.rglob("train_labels.csv"):
        base = csv.parent
        if (base / "val_labels.csv").exists() and (base / "train").exists() and (base / "val").exists():
            return base
    # Some mirrors might put train/val directly under a "FairFace" folder
    for cand in download_root.rglob("FairFace"):
        if (cand / "train").exists() and (cand / "val").exists():
            return cand
    return None

def main():
    src_root = Path(kagglehub.dataset_download("aibloy/fairface"))
    ff_src = find_fairface_root(src_root)
    if ff_src is None:
        raise SystemExit("Could not locate FairFace structure (train/ val/ + CSVs) inside the downloaded package.")

    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Copy train/ and val/ directories (images)
    shutil.copytree(ff_src / "train", OUT / "train", dirs_exist_ok=True)
    shutil.copytree(ff_src / "val",   OUT / "val",   dirs_exist_ok=True)

    # Copy label CSVs
    shutil.copy2(ff_src / "train_labels.csv", OUT / "train_labels.csv")
    shutil.copy2(ff_src / "val_labels.csv",   OUT / "val_labels.csv")

    # Summary
    tr = count_images(OUT / "train")
    va = count_images(OUT / "val")
    print(f"OK: FairFace → {OUT} | train={tr} val={va} (CSV: train_labels.csv, val_labels.csv)")

if __name__ == "__main__":
    main()
