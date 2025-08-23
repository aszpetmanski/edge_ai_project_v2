# src/models/mobilenet_heads.py
from typing import Dict
import torch
from torch import nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

class MultiHeadMobileNetV3(nn.Module):
    """
    MobileNetV3-Large backbone (ImageNet), three 2-way heads: age, gender, expr.
    Forward returns dict of logits: {"age": ..., "gender": ..., "expr": ...}
    """
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True, dropout: float = 0.2):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = mobilenet_v3_large(weights=weights)
        in_feats = self.backbone.classifier[0].in_features  # 960

        # strip the classifier, keep conv features + global pool
        self.backbone.classifier = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)

        # minimal heads (Linear; add small dropout for stability)
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()
        self.head_age    = nn.Linear(in_feats, 2)
        self.head_gender = nn.Linear(in_feats, 2)
        self.head_expr   = nn.Linear(in_feats, 2)

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_blocks(self, n_blocks: int = 1):
        feats = list(self.backbone.features.children())
        for m in feats[-n_blocks:]:
            for p in m.parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.backbone.features(x)
        z = torch.flatten(self.pool(feats), 1)
        z = self.dropout(z)
        return {
            "age":    self.head_age(z),
            "gender": self.head_gender(z),
            "expr":   self.head_expr(z),
        }

    def head_parameters(self):
        return list(self.head_age.parameters()) + list(self.head_gender.parameters()) + list(self.head_expr.parameters())
