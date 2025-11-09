import torch.nn as nn
import torch.nn.functional as F


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