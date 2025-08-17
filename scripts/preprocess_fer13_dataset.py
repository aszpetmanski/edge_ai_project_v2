#!/usr/bin/env python3
from pathlib import Path
import argparse
from PIL import Image
from tqdm import tqdm

SPLITS = ("train", "test")
CLASSES = ("happy", "sad")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def resolve_path(p: str) -> Path:
    pth = Path(p)
    return pth if pth.is_absolute() else (PROJECT_ROOT / pth)

def is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def process_split(src_split: Path, dst_split: Path, size: int, overwrite: bool) -> int:
    if not src_split.exists():
        print(f"[warn] missing split folder: {src_split}")
        return 0
    total = 0
    for cls in CLASSES:
        src_cls = src_split / cls
        if not src_cls.exists():
            print(f"[warn] missing class folder: {src_cls}")
            continue
        dst_cls = dst_split / cls
        dst_cls.mkdir(parents=True, exist_ok=True)

        imgs = [p for p in src_cls.rglob("*") if is_img(p)]
        for p in tqdm(imgs, desc=f"{src_split.name}/{cls}", leave=False):
            out = dst_cls / (p.stem + ".jpg")
            if out.exists() and not overwrite:
                total += 1
                continue
            try:
                im = Image.open(p)
                im = im.convert("RGB").resize((size, size), Image.BICUBIC)
                im.save(out, quality=95, optimize=True)
                total += 1
            except Exception as e:
                print(f"[skip] {p}: {e}")
                continue
    return total

def main():
    ap = argparse.ArgumentParser(description="Upscale FER2013 happy/sad to RGB 224x224.")
    ap.add_argument("--src", default="data/raw/fer2013", help="source root with train/test/happy|sad")
    ap.add_argument("--dst", default="data/processed/fer2013_rgb224", help="destination root")
    ap.add_argument("--size", type=int, default=224, help="output size (square)")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args()

    src_root = resolve_path(args.src)
    dst_root = resolve_path(args.dst)
    dst_root.mkdir(parents=True, exist_ok=True)

    print(f"[info] project root: {PROJECT_ROOT}")
    print(f"[info] src: {src_root}")
    print(f"[info] dst: {dst_root}")

    grand_total = 0
    for split in SPLITS:
        n = process_split(src_root / split, dst_root / split, args.size, args.overwrite)
        print(f"{split}: {n} images processed")
        grand_total += n

    print(f"OK: wrote RGB {args.size}x{args.size} images to {dst_root} | total={grand_total}")

if __name__ == "__main__":
    main()

