#!/usr/bin/env python3
from pathlib import Path
import csv
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR   = PROJECT_ROOT / "data" / "splits"
FF_CLEAN_T   = SPLITS_DIR / "fairface_clean_train.csv"
FF_CLEAN_V   = SPLITS_DIR / "fairface_clean_val.csv"
FER_PROC     = PROJECT_ROOT / "data" / "processed" / "fer2013_rgb224"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def rel_to_root(p: Path) -> str:
    p = p.resolve()
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)

def is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "age", "gender", "expr"])
        w.writeheader()
        w.writerows(rows)

def load_fairface_clean(csv_path: Path):
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path, keep_default_na=False)
    needed_cols = {"img_path", "gender", "age_bin"}
    if not needed_cols.issubset(df.columns):
        raise SystemExit(f"[error] {csv_path} missing columns {needed_cols}")

    rows = []
    for _, r in df.iterrows():
        img_path = Path(str(r["img_path"]).strip())
        if not is_img(img_path):
            continue
        # gender map: Male->0, Female->1 (case-insensitive)
        g = str(r.get("gender", "")).strip().lower()
        if g in {"female", "f"}:
            gender = 1
        elif g in {"male", "m"}:
            gender = 0
        else:
            gender = ""

        age = int(r["age_bin"]) if str(r["age_bin"]).strip() != "" else ""
        rows.append({
            "path": rel_to_root(img_path),
            "age": age,
            "gender": gender,
            "expr": ""
        })
    return rows

def load_fer_proc(split: str):
    base = FER_PROC / split
    rows = []
    for cls, label in [("happy", 0), ("sad", 1)]:
        d = base / cls
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if is_img(p):
                rows.append({
                    "path": rel_to_root(p),
                    "age": "",
                    "gender": "",
                    "expr": label
                })
    return rows

def main():
    # FairFace
    ff_train = load_fairface_clean(FF_CLEAN_T)
    ff_val   = load_fairface_clean(FF_CLEAN_V)

    # FER2013 processed
    fer_train = load_fer_proc("train")
    fer_test  = load_fer_proc("test")

    # Combine
    train_rows = ff_train + fer_train
    val_rows   = ff_val            # (no FER val yet; we’ll decide later)
    test_rows  = fer_test          # (FairFace has no test; keep test = FER)

    # Write
    write_csv(SPLITS_DIR / "train.csv", train_rows)
    write_csv(SPLITS_DIR / "val.csv",   val_rows)
    write_csv(SPLITS_DIR / "test.csv",  test_rows)

    # Tiny summary
    def count(rows, key):
        return sum(1 for r in rows if str(r[key]).strip() != "")
    print("OK: CSVs written to", SPLITS_DIR)
    print(f"  train: total={len(train_rows)} | age+gender={count(train_rows,'age')} | expr={count(train_rows,'expr')}")
    print(f"  val:   total={len(val_rows)}   | age+gender={count(val_rows,'age')}   | expr={count(val_rows,'expr')}")
    print(f"  test:  total={len(test_rows)}  | age+gender={count(test_rows,'age')}  | expr={count(test_rows,'expr')}")

if __name__ == "__main__":
    main()
