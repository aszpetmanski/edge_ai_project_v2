#!/usr/bin/env python3
from pathlib import Path
import argparse, random, math, csv
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Defaults
FF_CLEAN_T = PROJECT_ROOT / "data" / "splits" / "fairface_clean_train.csv"
FF_CLEAN_V = PROJECT_ROOT / "data" / "splits" / "fairface_clean_val.csv"   # used as FairFace TEST
FER_PROC   = PROJECT_ROOT / "data" / "processed" / "fer2013_rgb224"
OUT_DIR    = PROJECT_ROOT / "data" / "splits"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def rel_to_root(p: Path) -> str:
    p = p.resolve()
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)

# -------- FairFace: stratified val from TRAIN (by age_bin x gender) --------
def normalize_gender_bin(g: str):
    s = str(g).strip().lower()
    if s in {"male", "m"}: return 0
    if s in {"female", "f"}: return 1
    return None

def fairface_load_clean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, keep_default_na=False)
    need = {"img_path", "age_bin", "gender"}
    if not need.issubset(df.columns):
        raise SystemExit(f"[error] {csv_path} missing columns {need}")
    # ensure files exist
    df["img_path"] = df["img_path"].map(lambda x: str(x).strip())
    df["abs_path"] = df["img_path"].map(lambda x: (PROJECT_ROOT / x) if not Path(x).is_absolute() else Path(x))
    df = df[df["abs_path"].map(is_img)]
    # numeric bins for stratification
    df["gender_bin"] = df["gender"].map(normalize_gender_bin)
    df = df[df["gender_bin"].notna()]
    df["age_bin"] = df["age_bin"].astype(int)
    return df

def fairface_stratified_split(train_df: pd.DataFrame, val_frac: float, seed: int):
    random_state = pd.Series(range(len(train_df))).sample(frac=1.0, random_state=seed)  # just to use a seed
    # per (age_bin, gender_bin) group, take ceil(frac * n), but at least 1 if group has entries
    idx_val = []
    for (a, g), grp in train_df.groupby(["age_bin", "gender_bin"]):
        n = len(grp)
        k = max(1, int(round(n * val_frac))) if n > 0 else 0
        if k > 0 and k < n:
            idx_val.extend(grp.sample(n=k, random_state=seed).index.tolist())
        elif k >= n:
            idx_val.extend(grp.index.tolist())  # tiny group edge case
    mask_val = train_df.index.isin(idx_val)
    df_val = train_df[mask_val].copy()
    df_train = train_df[~mask_val].copy()
    return df_train, df_val

def fairface_rows_from_df(df: pd.DataFrame):
    rows = []
    for _, r in df.iterrows():
        gbin = normalize_gender_bin(r["gender"])
        if gbin is None:
            continue
        rows.append({
            "path": rel_to_root(r["abs_path"]),
            "age": int(r["age_bin"]),
            "gender": int(gbin),
            "expr": ""
        })
    return rows

# -------- FER: stratified val from TRAIN (by class happy/sad) --------
def fer_collect(split: str):
    base = FER_PROC / split
    files = {"happy": [], "sad": []}
    for cls in files.keys():
        d = base / cls
        if d.exists():
            files[cls] = [p for p in d.rglob("*") if is_img(p)]
    return files

def fer_split_train_val(fer_train_files: dict, val_frac: float, seed: int):
    rng = random.Random(seed)
    train_keep = {"happy": [], "sad": []}
    val_take   = {"happy": [], "sad": []}
    for cls in ("happy", "sad"):
        lst = list(fer_train_files.get(cls, []))
        rng.shuffle(lst)
        n = len(lst)
        k = max(1, int(round(n * val_frac))) if n > 0 else 0
        val_take[cls] = lst[:k]
        train_keep[cls] = lst[k:]
    return train_keep, val_take

def fer_rows_from_files(files_dict: dict):
    rows = []
    for cls, lab in [("happy", 0), ("sad", 1)]:
        for p in files_dict.get(cls, []):
            rows.append({
                "path": rel_to_root(p),
                "age": "",
                "gender": "",
                "expr": lab
            })
    return rows

# -------- Write CSV --------
def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "age", "gender", "expr"])
        w.writeheader()
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser(description="Rebuild unified train/val/test with stratified VAL from FairFace-train and FER-train.")
    ap.add_argument("--ff-train", default=str(FF_CLEAN_T))
    ap.add_argument("--ff-test",  default=str(FF_CLEAN_V), help="FairFace official val CSV (used as TEST)")
    ap.add_argument("--fer-root", default=str(FER_PROC))
    ap.add_argument("--val-frac-ff", type=float, default=0.10)
    ap.add_argument("--val-frac-fer", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    # FairFace
    ff_train_df = fairface_load_clean(Path(args.ff_train))
    ff_test_df  = fairface_load_clean(Path(args.ff_test))   # whole file becomes TEST set
    ff_train_rem, ff_val_from_train = fairface_stratified_split(ff_train_df, args.val_frac_ff, args.seed)

    ff_train_rows = fairface_rows_from_df(ff_train_rem)
    ff_val_rows   = fairface_rows_from_df(ff_val_from_train)
    ff_test_rows  = fairface_rows_from_df(ff_test_df)

    # FER
    fer_train_files = fer_collect("train")
    fer_test_files  = fer_collect("test")
    fer_train_keep, fer_val_take = fer_split_train_val(fer_train_files, args.val_frac_fer, args.seed)

    fer_train_rows = fer_rows_from_files(fer_train_keep)
    fer_val_rows   = fer_rows_from_files(fer_val_take)
    fer_test_rows  = fer_rows_from_files(fer_test_files)

    # Unified
    train_rows = ff_train_rows + fer_train_rows
    val_rows   = ff_val_rows   + fer_val_rows
    test_rows  = ff_test_rows  + fer_test_rows

    # Write
    write_csv(out_dir / "train.csv", train_rows)
    write_csv(out_dir / "val.csv",   val_rows)
    write_csv(out_dir / "test.csv",  test_rows)

    # Tiny summary
    def cnt(rows, key): return sum(1 for r in rows if str(r[key]).strip() != "")
    print("OK:")
    print(f"  train: total={len(train_rows)} | age+gender={cnt(train_rows,'age')} | expr={cnt(train_rows,'expr')}")
    print(f"  val:   total={len(val_rows)}   | age+gender={cnt(val_rows,'age')}   | expr={cnt(val_rows,'expr')}")
    print(f"  test:  total={len(test_rows)}  | age+gender={cnt(test_rows,'age')}  | expr={cnt(test_rows,'expr')}")

if __name__ == "__main__":
    main()