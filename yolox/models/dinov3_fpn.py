import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class dinov3FPN(nn.Module):
    def __init__(self, backbone_name="timm/vit_small_plus_patch16_dinov3.lvd1689m", out_channels=256):
        super().__init__()
        # timmバックボーンをfeatures_onlyで
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
        )

    def forward(self, x):
        feats = self.backbone(x)
        p_80 = F.interpolate(feats[-1], size=(80, 80), mode="bilinear", align_corners=False)
        return [p_80]

