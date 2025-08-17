#!/usr/bin/env python3
import random
from pathlib import Path
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FAIR = ROOT / "data" / "raw" / "fairface" / "FairFace"   # expects train/, val/, train_labels.csv, val_labels.csv
FER  = ROOT / "data" / "raw" / "fer2013"                 # expects train/test with happy/sad subfolders
OUTP = ROOT / "runs" / "preview" / "preview_8.png"

def is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}

def pick_fairface(n=4, seed=42):
    dfs = []
    for name in ("train_labels.csv", "val_labels.csv"):
        p = FAIR / name
        if p.exists():
            df = pd.read_csv(p)
            if "file" in df.columns:
                dfs.append(df[["file", "age", "gender"]])
    if not dfs:
        return []
    df = pd.concat(dfs, ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed)

    out = []
    for _, r in df.iterrows():
        rel = str(r["file"]).strip()                        # e.g., train/1.jpg
        img_path = FAIR / rel
        if is_img(img_path):
            title = f"FairFace | {str(r.get('age','')).strip()} | {str(r.get('gender','')).strip()}"
            out.append((img_path, title))
            if len(out) >= n: break
    return out

def pick_fer(n=4, seed=42):
    cands = []
    for split in ("train", "test"):
        for cls in ("happy", "sad"):
            d = FER / split / cls
            if d.exists():
                for p in d.rglob("*"):
                    if is_img(p):
                        cands.append((p, f"FER2013 | {cls}"))
    random.seed(seed)
    random.shuffle(cands)
    return cands[:n]

def main():
    fair = pick_fairface(4)
    fer  = pick_fer(4)
    if len(fair) < 4 or len(fer) < 4:
        raise SystemExit("Not enough images found. Check data/raw structure.")

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    items = fair + fer
    for ax, (path, title) in zip(axes.flat, items):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        ax.imshow(img)
        ax.set_title(f"{title} | {w}x{h}", fontsize=9)
        ax.axis("off")

    OUTP.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTP, dpi=150)
    plt.close(fig)
    print(f"Saved: {OUTP}")

if __name__ == "__main__":
    main()


