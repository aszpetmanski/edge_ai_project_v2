from pathlib import Path
import torch
from torchvision import models

out_dir = Path("runs/checkpoints")
out_dir.mkdir(parents=True, exist_ok=True)

try:
    weights = models.MobileNet_V3_Large_Weights.DEFAULT  # torchvision >= 0.13
    model = models.mobilenet_v3_large(weights=weights)
except AttributeError:
    model = models.mobilenet_v3_large(pretrained=True)   # older torchvision

torch.save(model.state_dict(), out_dir / "mobilenet_v3_large_imagenet_full.pth")