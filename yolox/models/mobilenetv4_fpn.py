
import torch.nn as nn
import torch.nn.functional as F
import timm
from .fpn import FPN


class mobilenetv4FPN(nn.Module):
    def __init__(self, backbone_name="mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k", out_channels=256):
        super().__init__()
        # timmバックボーンをfeatures_onlyで
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            #out_indices=(2,3,4),  
        )
        chs = self.backbone.feature_info.channels()
        # 各段のチャネル数をtimmから取得
        self.used_idx = [-3, -2, -1]
        self.used_idx = [i % len(chs) for i in self.used_idx]
        in_channels_list = [chs[i] for i in self.used_idx]
        self.fpn = FPN(in_channels_list, out_channels=out_channels)

    def forward(self, x):
        # feats: [C2, C3, C4] みたいなリストで返る
        feats = self.backbone(x)
        # そのままFPNへ
        feats_for_fpn = [feats[i] for i in self.used_idx]
        fpn_feats = self.fpn(feats_for_fpn)
        return [fpn_feats[0]]  # [P2, P3, P4]
