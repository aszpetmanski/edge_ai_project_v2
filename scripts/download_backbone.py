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

save_path = out_dir / "mobilenet_v3_large_imagenet_full.pth"
torch.save(model.state_dict(), save_path)

# single final check + print
if save_path.exists() and save_path.stat().st_size > 0:
    print(f"OK: saved weights to {save_path} ({save_path.stat().st_size} bytes)")
else:
    raise RuntimeError("Save failed: file missing or empty.")

print('elo')