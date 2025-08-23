#!/usr/bin/env python3
import torch
from pathlib import Path
from unified_dataset import UnifiedCsvDataset, SPLITS_DIR, PROJECT_ROOT
from transforms import eval_tfms
from mobilenet_heads import MultiHeadMobileNetV3
from train_heads import evaluate, CKPT_PATH, DEVICE, BATCH_SIZE, NUM_WORKERS  # reuse

def main():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model = MultiHeadMobileNetV3(pretrained=False, freeze_backbone=True).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])

    ds = UnifiedCsvDataset(SPLITS_DIR / "test.csv", eval_tfms, select="all")
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                         num_workers=NUM_WORKERS, pin_memory=(DEVICE=="cuda"))

    acc, f1 = evaluate(model, loader, weights={"age":None,"gender":None,"expr":None})
    print("TEST metrics:")
    print(f"  acc  -> age={acc['age']}, gender={acc['gender']}, expr={acc['expr']}")
    print(f"  f1   -> age={f1['age']},  gender={f1['gender']},  expr={f1['expr']}")
    print(f"(from checkpoint: epoch={ckpt.get('epoch')}, val_macroF1={ckpt.get('val_f1_macro_avg')})")

if __name__ == "__main__":
    main()
