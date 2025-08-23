# scripts/train_heads.py
#!/usr/bin/env python3
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd

from unified_dataset import build_loaders, UnifiedCsvDataset, SPLITS_DIR, PROJECT_ROOT
from transforms import eval_tfms
from mobilenet_heads import MultiHeadMobileNetV3

# ---------------- basics ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = 8
BATCH_SIZE = 64
LR = 1e-3
WD = 1e-4
NUM_WORKERS = 4
SEED = 42

torch.manual_seed(SEED)
random.seed(SEED)

CKPT_DIR = PROJECT_ROOT / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = CKPT_DIR / "heads_mnv3.pt"
WEIGHTS_PATH = CKPT_DIR / "class_weights.json"

# ---------------- utils ----------------
def compute_class_weights_from_csv(csv_path: Path, col: str):
    """Return tensor([w0, w1]) with w_c = N / (2*N_c)."""
    df = pd.read_csv(csv_path, keep_default_na=False)
    s = df[col].astype(str).str.strip()
    n0 = (s == "0").sum()
    n1 = (s == "1").sum()
    N = n0 + n1
    eps = 1e-6
    w0 = N / (2.0 * max(n0, eps))
    w1 = N / (2.0 * max(n1, eps))
    return torch.tensor([w0, w1], dtype=torch.float32)

def masked_ce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, class_weights: torch.Tensor | None):
    """
    logits: (B,2), targets: (B,), mask: (B,)
    returns scalar loss (0 if no valid items)
    """
    if mask.sum() == 0:
        return logits.sum() * 0.0  # zero, stays on correct device/graph
    logits = logits[mask]
    targets = targets[mask]
    if class_weights is not None:
        ce = nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
    else:
        ce = nn.CrossEntropyLoss()
    return ce(logits, targets)

def batch_step(model, batch_ff, batch_fer, weights, optimizer):
    model.train()
    optimizer.zero_grad()

    loss_total = 0.0

    # FairFace batch: age + gender
    if batch_ff is not None:
        x, y, m = batch_ff  # labels: [age, gender, expr], mask same
        out = model(x.to(DEVICE))
        loss_age = masked_ce_loss(out["age"],    y[:,0].to(DEVICE), m[:,0].to(DEVICE), weights["age"])
        loss_gen = masked_ce_loss(out["gender"], y[:,1].to(DEVICE), m[:,1].to(DEVICE), weights["gender"])
        loss_total = loss_total + loss_age + loss_gen

    # FER batch: expr
    if batch_fer is not None:
        x, y, m = batch_fer
        out = model(x.to(DEVICE))
        loss_expr = masked_ce_loss(out["expr"], y[:,2].to(DEVICE), m[:,2].to(DEVICE), weights["expr"])
        loss_total = loss_total + loss_expr

    loss_total.backward()
    optimizer.step()
    return loss_total.item()

@torch.no_grad()
def evaluate(model, loader_all: DataLoader, weights):
    model.eval()
    # accumulators per head
    acc = {"age": None, "gender": None, "expr": None}
    f1  = {"age": None, "gender": None, "expr": None}

    # counters for accuracy & confusion (binary) per head
    def mkc():
        return {"tp0":0,"fp0":0,"fn0":0,"tp1":0,"fp1":0,"fn1":0,"correct":0,"total":0}
    C = {"age": mkc(), "gender": mkc(), "expr": mkc()}

    for x, y, m in loader_all:
        x = x.to(DEVICE)
        out = model(x)
        for head, idx in [("age",0), ("gender",1), ("expr",2)]:
            mask = m[:,idx].bool()
            if mask.sum() == 0:
                continue
            logits = out[head][mask]
            preds = logits.argmax(dim=1).cpu()
            t = y[:,idx][mask].cpu()

            C[head]["correct"] += int((preds == t).sum().item())
            C[head]["total"]   += int(len(t))

            # confusion for macro-F1
            # class 0
            tp0 = int(((preds==0)&(t==0)).sum())
            fp0 = int(((preds==0)&(t!=0)).sum())
            fn0 = int(((preds!=0)&(t==0)).sum())
            # class 1
            tp1 = int(((preds==1)&(t==1)).sum())
            fp1 = int(((preds==1)&(t!=1)).sum())
            fn1 = int(((preds!=1)&(t==1)).sum())

            C[head]["tp0"] += tp0; C[head]["fp0"] += fp0; C[head]["fn0"] += fn0
            C[head]["tp1"] += tp1; C[head]["fp1"] += fp1; C[head]["fn1"] += fn1

    def finalize(head):
        if C[head]["total"] == 0:
            return None, None
        accv = C[head]["correct"] / C[head]["total"]
        # macro F1
        def f1(tp, fp, fn):
            denom = (2*tp + fp + fn)
            return (2*tp / denom) if denom > 0 else 0.0
        f1_0 = f1(C[head]["tp0"], C[head]["fp0"], C[head]["fn0"])
        f1_1 = f1(C[head]["tp1"], C[head]["fp1"], C[head]["fn1"])
        f1m = (f1_0 + f1_1) / 2.0
        return accv, f1m

    for head in ["age","gender","expr"]:
        a, fm = finalize(head)
        acc[head] = a
        f1[head] = fm
    return acc, f1

# ---------------- main ----------------
def main():
    # loaders
    ff_tr, fer_tr = build_loaders("train", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    # for evaluation, we want both tasks in the same loader → select="all"
    val_all = DataLoader(
        UnifiedCsvDataset(SPLITS_DIR / "val.csv", eval_tfms, select="all"),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    # class weights from TRAIN
    train_csv = SPLITS_DIR / "train.csv"
    w_age    = compute_class_weights_from_csv(train_csv, "age")
    w_gender = compute_class_weights_from_csv(train_csv, "gender")
    w_expr   = compute_class_weights_from_csv(train_csv, "expr")
    # (optional) clip extreme weights (elderly may be very rare)
    w_age = torch.clamp(w_age, max=5.0)
    weights = {"age": w_age, "gender": w_gender, "expr": w_expr}
    # save for reference
    with open(WEIGHTS_PATH, "w") as f:
        json.dump({k: [float(x) for x in v.tolist()] for k,v in weights.items()}, f, indent=2)

    # model + optimizer (heads-only)
    model = MultiHeadMobileNetV3(pretrained=True, freeze_backbone=True, dropout=0.2).to(DEVICE)
    opt = torch.optim.AdamW(model.head_parameters(), lr=LR, weight_decay=WD)

    best_score = -1.0
    for epoch in range(1, EPOCHS+1):
        model.train()
        it_ff  = iter(ff_tr)
        it_fer = iter(fer_tr)
        steps = min(len(ff_tr), len(fer_tr))
        running = 0.0

        for _ in range(steps):
            try:
                batch_ff = next(it_ff)
            except StopIteration:
                it_ff = iter(ff_tr); batch_ff = next(it_ff)
            try:
                batch_fer = next(it_fer)
            except StopIteration:
                it_fer = iter(fer_tr); batch_fer = next(it_fer)

            loss = batch_step(model, batch_ff, batch_fer, weights, opt)
            running += loss

        # evaluate
        acc, f1 = evaluate(model, val_all, weights)
        # average macro-F1 across available heads
        f1_vals = [v for v in [f1["age"], f1["gender"], f1["expr"]] if v is not None]
        score = sum(f1_vals)/len(f1_vals) if f1_vals else 0.0

        print(f"Epoch {epoch:02d} | train_loss={running/steps:.4f} "
              f"| val_acc(age={acc['age']}, gen={acc['gender']}, expr={acc['expr']}) "
              f"| val_f1(age={f1['age']}, gen={f1['gender']}, expr={f1['expr']})")

        if score > best_score:
            best_score = score
            torch.save({
                "model_state": model.state_dict(),
                "weights": {k: v.tolist() for k,v in weights.items()},
                "epoch": epoch,
                "val_f1_macro_avg": score,
            }, CKPT_PATH)
            print(f"  ↳ saved best to {CKPT_PATH} (macro-F1={score:.4f})")

    # ---- optional fine-tune last MobileNet block ----
    FT_EPOCHS = 3  # run 3-5 is typical
    FT_LR = 1e-4  # 10x lower than heads-only LR

    model.unfreeze_last_blocks(n_blocks=1)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FT_LR, weight_decay=WD
    )

    for epoch in range(EPOCHS + 1, EPOCHS + FT_EPOCHS + 1):
        model.train()
        it_ff, it_fer = iter(ff_tr), iter(fer_tr)
        steps = min(len(ff_tr), len(fer_tr))
        running = 0.0
        for _ in range(steps):
            try:
                batch_ff = next(it_ff)
            except StopIteration:
                it_ff = iter(ff_tr); batch_ff = next(it_ff)
            try:
                batch_fer = next(it_fer)
            except StopIteration:
                it_fer = iter(fer_tr); batch_fer = next(it_fer)

            loss = batch_step(model, batch_ff, batch_fer, weights, opt)
            running += loss

        acc, f1 = evaluate(model, val_all, weights)
        f1_vals = [v for v in [f1["age"], f1["gender"], f1["expr"]] if v is not None]
        score = sum(f1_vals) / len(f1_vals) if f1_vals else 0.0

        print(f"[FT] Epoch {epoch:02d} | train_loss={running / steps:.4f} "
              f"| val_acc(age={acc['age']}, gen={acc['gender']}, expr={acc['expr']}) "
              f"| val_f1(age={f1['age']}, gen={f1['gender']}, expr={f1['expr']})")

        if score > best_score:
            best_score = score
            torch.save({
                "model_state": model.state_dict(),
                "weights": {k: v.tolist() for k, v in weights.items()},
                "epoch": epoch,
                "val_f1_macro_avg": score,
            }, CKPT_PATH)
            print(f"  ↳ saved best to {CKPT_PATH} (macro-F1={score:.4f})")

    print("Done.")

if __name__ == "__main__":
    main()
