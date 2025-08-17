#!/usr/bin/env python3
from pathlib import Path
import shutil
import kagglehub

CLASSES = ["happy", "sad"]
SPLITS = ["train", "test"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "fer2013"

def find_split_dir(root: Path, split: str) -> Path | None:
    for d in root.rglob(split):
        if d.is_dir() and all((d / c).exists() for c in CLASSES):
            return d
    return None

def count_images(d: Path) -> int:
    if not d or not d.exists(): return 0
    return sum(1 for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

def main():
    src_root = Path(kagglehub.dataset_download("msambare/fer2013"))

    # locate split dirs inside the downloaded bundle
    split_dirs = {s: find_split_dir(src_root, s) for s in SPLITS}
    if any(v is None for v in split_dirs.values()):
        raise SystemExit("Could not locate expected 'train'/'test' folders with happy/sad inside the downloaded dataset.")

    # copy only happy/sad into our project
    for split in SPLITS:
        for cls in CLASSES:
            src = split_dirs[split] / cls
            dst = OUT / split / cls
            dst.mkdir(parents=True, exist_ok=True)
            for p in src.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    shutil.copy2(p, dst / p.name)

    # summary
    tr_h = count_images(OUT / "train" / "happy")
    tr_s = count_images(OUT / "train" / "sad")
    te_h = count_images(OUT / "test" / "happy")
    te_s = count_images(OUT / "test" / "sad")
    print(f"OK: fer2013 (happy/sad) → {OUT}")
    print(f"train: happy={tr_h} sad={tr_s} | test: happy={te_h} sad={te_s}")

if __name__ == "__main__":
    main()



