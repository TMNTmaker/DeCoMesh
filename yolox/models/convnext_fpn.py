import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        """
        in_channels_list: バックボーンの各段のチャネル数（低解像度の段も含む）
        out_channels: FPNで揃えたい出力チャネル数
        """
        super().__init__()
        # lateral conv (1x1) と output conv (3x3) を段ごとに作る
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        for in_ch in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_ch, out_channels, kernel_size=1))
            self.output_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )

    def forward(self, feats):
        """
        feats: list of feature maps (大きい順でも小さい順でもいいが、ここでは「浅い→深い」を想定)
               例: [C2, C3, C4, C5]
        return: list of FPN feature maps [P2, P3, P4, P5]
        """
        # 一番最後（最も深くて解像度の小さい）特徴から処理
        last_inner = self.lateral_convs[-1](feats[-1])
        outs = [self.output_convs[-1](last_inner)]

        # 後ろから前に向かってFPNを組む
        for i in range(len(feats) - 2, -1, -1):
            lateral = self.lateral_convs[i](feats[i])
            # 上の段をアップサンプルして足す
            up = F.interpolate(last_inner, size=lateral.shape[-2:], mode="nearest")
            last_inner = lateral + up
            outs.insert(0, self.output_convs[i](last_inner))

        return outs


class ConvNeXtFPN(nn.Module):
    def __init__(self, backbone_name="convnextv2_tiny.fcmae_ft_in22k_in1k_384", out_channels=256):
        super().__init__()
        # timmバックボーンをfeatures_onlyで
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2,3),  # ConvNeXt V2では0,1,2を使用して192,384,768のチャネルを取得
        )
        # 各段のチャネル数をtimmから取得
        in_channels_list = [f["num_chs"] for f in self.backbone.feature_info]
        self.fpn = FPN(in_channels_list, out_channels=out_channels)

    def forward(self, x):
        # feats: [C2, C3, C4] みたいなリストで返る
        feats = self.backbone(x)
        # そのままFPNへ
        fpn_feats = self.fpn(feats)
        return [fpn_feats[1]]  # [P2, P3, P4]

#
#if __name__ == "__main__":
#    model = ConvNeXtFPN("convnext_base", out_channels=256)
#    x = torch.randn(1, 3, 224, 224)
#    outs = model(x)
#    for i, o in enumerate(outs):
#        print(i, o.shape)
#