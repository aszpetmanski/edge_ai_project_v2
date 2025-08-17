#!/usr/bin/env python3
SEED = 42  # fixed seed for reproducibility

import argparse, math, random
from pathlib import Path
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR   = PROJECT_ROOT / "data" / "splits"
OUT_DIR      = PROJECT_ROOT / "runs" / "preview"
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp"}

def is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def to_abs(p):
    p = Path(str(p).strip())
    return p if p.is_absolute() else (PROJECT_ROOT / p)

def label_text(row):
    parts = []
    if str(row.get("age","")).strip() != "" or str(row.get("gender","")).strip() != "":
        age = int(row["age"]) if str(row["age"]).strip() != "" else None
        gender = int(row["gender"]) if str(row["gender"]).strip() != "" else None
        if age is not None:    parts.append("adult" if age == 0 else "elderly")
        if gender is not None: parts.append("male" if gender == 0 else "female")
    if str(row.get("expr","")).strip() != "":
        expr = int(row["expr"])
        parts.append("happy" if expr == 0 else "sad")
    return " | ".join(parts) if parts else "(no labels)"

def main():
    random.seed(SEED)
    ap = argparse.ArgumentParser(description="Balanced preview: half FairFace (age+gender), half FER (expr).")
    ap.add_argument("--split", choices=["train","val","test"], default="val")
    ap.add_argument("--n", type=int, default=25, help="total images to show")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    csv_path = SPLITS_DIR / f"{args.split}.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing split CSV: {csv_path}")

    df = pd.read_csv(csv_path, keep_default_na=False)
    need = {"path","age","gender","expr"}
    if not need.issubset(df.columns):
        raise SystemExit(f"CSV must have columns: {need}")

    # resolve and filter valid images
    df["abs_path"] = df["path"].map(to_abs)
    df = df[df["abs_path"].map(is_img)]
    if len(df) == 0:
        raise SystemExit("No valid image paths in this split.")

    # partition rows: FairFace-like vs FER-like
    is_ff  = (df["age"].astype(str).str.strip() != "") | (df["gender"].astype(str).str.strip() != "")
    is_fer = (df["expr"].astype(str).str.strip() != "")

    df_ff  = df[is_ff].sample(frac=1.0, random_state=SEED)
    df_fer = df[is_fer].sample(frac=1.0, random_state=SEED)

    if len(df_ff) == 0 and len(df_fer) == 0:
        raise SystemExit("Split has no labeled rows.")

    # target counts
    n_ff  = min(args.n // 2, len(df_ff))
    n_fer = min(args.n - n_ff, len(df_fer))

    # if one side is short, fill from the other
    if n_ff + n_fer < args.n:
        if n_ff < len(df_ff):
            take = min(args.n - (n_ff + n_fer), len(df_ff) - n_ff)
            n_ff += take
        elif n_fer < len(df_fer):
            take = min(args.n - (n_ff + n_fer), len(df_fer) - n_fer)
            n_fer += take

    df_sel = pd.concat([df_ff.iloc[:n_ff], df_fer.iloc[:n_fer]], ignore_index=True)
    df_sel = df_sel.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # grid
    n, cols = len(df_sel), max(1, args.cols)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6*cols, 3.6*rows))

    if rows == 1 and cols == 1:
        ax_list = [axes]
    elif rows == 1:
        ax_list = list(axes)
    elif cols == 1:
        ax_list = [ax for ax in axes]
    else:
        ax_list = [ax for row in axes for ax in row]

    for ax, (_, r) in zip(ax_list, df_sel.iterrows()):
        try:
            img = Image.open(r["abs_path"]).convert("RGB")
            w, h = img.size
            ax.imshow(img)
            ax.set_title(f"{args.split} | {label_text(r)} | {w}x{h}", fontsize=8)
            ax.axis("off")
        except Exception as e:
            ax.text(0.5, 0.5, f"Error\n{e}", ha="center", va="center", fontsize=8)
            ax.axis("off")

    for ax in ax_list[n:]:
        ax.axis("off")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"preview_unified_balanced_{args.split}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if args.show:
        plt.show()
    plt.close(fig)
    print(f"Saved preview to: {out_path}\nPicked: FairFace={n_ff}  FER={n_fer}")

if __name__ == "__main__":
    main()
