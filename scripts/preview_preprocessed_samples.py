#!/usr/bin/env python3
import argparse, random
from pathlib import Path
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FF_CLEAN_T = PROJECT_ROOT / "data" / "splits" / "fairface_clean_train.csv"
FF_CLEAN_V = PROJECT_ROOT / "data" / "splits" / "fairface_clean_val.csv"
FER_PROC   = PROJECT_ROOT / "data" / "processed" / "fer2013_rgb224"
OUTP       = PROJECT_ROOT / "runs" / "preview" / "preview_mapped.png"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def load_fairface_clean(n: int, seed: int = 42):
    """Sample n rows from cleaned FairFace CSVs and return [(path, title), ...]."""
    frames = []
    for csv_path in (FF_CLEAN_T, FF_CLEAN_V):
        if csv_path.exists():
            df = pd.read_csv(csv_path, keep_default_na=False)
            need = {"img_path", "age_binary", "gender"}
            if not need.issubset(df.columns):
                continue
            frames.append(df[list(need)])
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed)

    items = []
    for _, r in df.iterrows():
        p = Path(str(r["img_path"]).strip())
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not is_img(p):
            continue
        age_b = str(r["age_binary"]).strip()
        gender = str(r["gender"]).strip()
        title = f"FairFace | {age_b} | {gender}"
        items.append((p, title))
        if len(items) >= n:
            break
    return items

def load_fer_proc(n: int, seed: int = 42):
    """Sample n images from processed FER2013 RGB224 train/test (happy/sad)."""
    cands = []
    for split in ("train", "test"):
        for cls in ("happy", "sad"):
            d = FER_PROC / split / cls
            if d.exists():
                for p in d.rglob("*"):
                    if is_img(p):
                        cands.append((p, f"FER2013 | {cls}"))
    if not cands:
        return []
    random.seed(seed)
    random.shuffle(cands)
    return cands[:n]

def main():
    ap = argparse.ArgumentParser(description="Preview mapped FairFace + preprocessed FER2013 (sanity check).")
    ap.add_argument("--ff",  type=int, default=4, help="FairFace samples (mapped adult/elderly)")
    ap.add_argument("--fer", type=int, default=4, help="FER2013 samples (RGB224 happy/sad)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", type=str, default=str(OUTP))
    ap.add_argument("--show", action="store_true", help="Show interactive window")
    args = ap.parse_args()

    ff_items  = load_fairface_clean(args.ff, args.seed)
    fer_items = load_fer_proc(args.fer, args.seed)

    if len(ff_items) < args.ff:
        raise SystemExit(f"Not enough FairFace cleaned samples found. Checked:\n  {FF_CLEAN_T}\n  {FF_CLEAN_V}")
    if len(fer_items) < args.fer:
        raise SystemExit(f"Not enough processed FER2013 samples found under:\n  {FER_PROC}")

    # 2 rows: top FairFace, bottom FER
    fig, axes = plt.subplots(2, max(args.ff, args.fer), figsize=(3.6*max(args.ff, args.fer), 7.2))

    # Ensure axes is 2 x C even if C==1
    if max(args.ff, args.fer) == 1:
        axes = axes.reshape(2, 1)

    # Top row: FairFace
    for i in range(max(args.ff, args.fer)):
        ax = axes[0, i]
        if i < len(ff_items):
            path, title = ff_items[i]
            img = Image.open(path).convert("RGB")
            w, h = img.size
            ax.imshow(img); ax.set_title(f"{title} | {w}x{h}", fontsize=9); ax.axis("off")
        else:
            ax.axis("off")

    # Bottom row: FER
    for i in range(max(args.ff, args.fer)):
        ax = axes[1, i]
        if i < len(fer_items):
            path, title = fer_items[i]
            img = Image.open(path).convert("RGB")
            w, h = img.size
            ax.imshow(img); ax.set_title(f"{title} | {w}x{h}", fontsize=9); ax.axis("off")
        else:
            ax.axis("off")

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    if args.show:
        plt.show()
    plt.close(fig)
    print(f"Saved preview to: {save_path}")

if __name__ == "__main__":
    main()
