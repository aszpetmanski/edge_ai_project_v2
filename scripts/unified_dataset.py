# src/data/unified_dataset.py
from pathlib import Path
import csv
from typing import List, Tuple, Optional
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transforms import train_tfms, eval_tfms  # you already created this

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # -> /Users/.../edge_ai_project
SPLITS_DIR   = PROJECT_ROOT / "data" / "splits"

def _to_abs(p: str) -> Path:
    pp = Path(p.strip())
    return pp if pp.is_absolute() else (PROJECT_ROOT / pp)

def _to_int_or_none(x: str) -> Optional[int]:
    s = str(x).strip()
    if s == "": return None
    return int(s)

class UnifiedCsvDataset(Dataset):
    """
    Yields: image (C,H,W), labels (LongTensor[3]: age, gender, expr), mask (BoolTensor[3])
    Missing labels are filled with 0 and masked False.
    select: "all" (default), "ff" (rows with age/gender), "fer" (rows with expr)
    """
    def __init__(self, csv_path: Path, transform: transforms.Compose, select: str = "all"):
        self.transform = transform
        self.rows: List[Tuple[Path, Optional[int], Optional[int], Optional[int]]] = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                p = _to_abs(r["path"])
                age = _to_int_or_none(r.get("age", ""))
                gen = _to_int_or_none(r.get("gender", ""))
                expr = _to_int_or_none(r.get("expr", ""))
                is_ff = (age is not None) or (gen is not None)
                is_fer = (expr is not None)
                if select == "ff" and not is_ff:   continue
                if select == "fer" and not is_fer: continue
                self.rows.append((p, age, gen, expr))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        p, age, gen, expr = self.rows[idx]
        img = Image.open(p).convert("RGB")
        x = self.transform(img)

        labels = torch.tensor([
            age if age is not None else 0,
            gen if gen is not None else 0,
            expr if expr is not None else 0,
        ], dtype=torch.long)

        mask = torch.tensor([
            age is not None,
            gen is not None,
            expr is not None,
        ], dtype=torch.bool)

        return x, labels, mask

def build_loaders(split: str,
                  batch_size: int = 64,
                  num_workers: int = 4,
                  pin_memory: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Returns (ff_loader, fer_loader) for the given split ("train" | "val" | "test").
    """
    csv_path = SPLITS_DIR / f"{split}.csv"
    tfm = train_tfms if split == "train" else eval_tfms

    ff_ds  = UnifiedCsvDataset(csv_path, tfm, select="ff")
    fer_ds = UnifiedCsvDataset(csv_path, tfm, select="fer")

    shuffle = (split == "train")
    ff_loader = DataLoader(ff_ds,  batch_size=batch_size, shuffle=shuffle,
                           num_workers=num_workers, pin_memory=pin_memory)
    fer_loader = DataLoader(fer_ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, pin_memory=pin_memory)
    return ff_loader, fer_loader
