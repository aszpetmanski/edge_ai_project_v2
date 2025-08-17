#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FAIR = ROOT / "data" / "raw" / "fairface" / "FairFace"
OUT  = ROOT / "data" / "splits"
OUT.mkdir(parents=True, exist_ok=True)

MINOR_BUCKETS = {"0-2", "3-9", "10-19"}

def is_elderly(age_str: str) -> bool:
    s = str(age_str).strip().lower()
    # treat any bucket that includes 60+ as elderly
    return any(tok in s for tok in ("60", "70", "80", "90", "100"))

def process_split(name: str, out_name: str):
    csv_path = FAIR / f"{name}_labels.csv"
    if not csv_path.exists():
        return 0, 0, 0
    df = pd.read_csv(csv_path)

    if "file" not in df.columns or "age" not in df.columns:
        raise SystemExit(f"Expected columns 'file' and 'age' in {csv_path}")

    # ensure image paths exist
    def to_path(rel):
        rel = str(rel).strip()
        return FAIR / rel

    df["img_path"] = df["file"].map(to_path)
    df = df[df["img_path"].map(lambda p: Path(p).exists())]

    # drop minors
    df = df[~df["age"].astype(str).isin(MINOR_BUCKETS)]

    # map to adult/elderly
    df["age_binary"] = df["age"].map(lambda a: "elderly" if is_elderly(a) else "adult")
    df["age_bin"] = df["age_binary"].map({"adult": 0, "elderly": 1})

    # write cleaned CSV (keep original useful columns + our mappings)
    keep_cols = [c for c in ["file","age","gender","race","service_test","img_path","age_binary","age_bin"] if c in df.columns]
    out_csv = OUT / out_name
    df[keep_cols].to_csv(out_csv, index=False)

    # counts
    adults = int((df["age_binary"] == "adult").sum())
    elders = int((df["age_binary"] == "elderly").sum())
    return len(df), adults, elders

def main():
    n_tr, a_tr, e_tr = process_split("train", "fairface_clean_train.csv")
    n_va, a_va, e_va = process_split("val",   "fairface_clean_val.csv")

    print(f"FairFace cleaned:")
    print(f"  train: total={n_tr}  adult={a_tr}  elderly={e_tr}")
    print(f"  val:   total={n_va}  adult={a_va}  elderly={e_va}")
    print(f"CSV out → {OUT}")

if __name__ == "__main__":
    main()
