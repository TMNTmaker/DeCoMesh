
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch.nn.functional as F

class EfficientNetFPN(nn.Module):
    def __init__(self, model_name="efficientnet-b2", out_channels=256, pretrained=True):
        super().__init__()
        if pretrained:
            self.backbone = EfficientNet.from_pretrained(model_name)
        else:
            self.backbone = EfficientNet.from_name(model_name)

        self.out_channels = out_channels

        # 後でendpointsのch数を見てから作るのでここはNoneで
        self.lateral = nn.ModuleDict()
        self.smooth = nn.ModuleDict({
            "p3": nn.Conv2d(out_channels, out_channels, 3, padding=1),
            "p4": nn.Conv2d(out_channels, out_channels, 3, padding=1),
            "p5": nn.Conv2d(out_channels, out_channels, 3, padding=1),
        })

    def _get_lateral(self, key, in_ch):
        if key not in self.lateral:
            self.lateral[key] = nn.Conv2d(in_ch, self.out_channels, 1)
            # 親モデルがGPUにある場合、新しいレイヤーもGPUに移動
            if next(self.parameters()).is_cuda:
                self.lateral[key] = self.lateral[key].cuda()
        return self.lateral[key]

    def forward(self, x):
        # ここで全部の中間特徴を取る
        endpoints = self.backbone.extract_endpoints(x)
        # endpoints はこんな感じのdictになることが多い:
        # {
        #   'reduction_1': ...,
        #   'reduction_2': ...,
        #   'reduction_3': ...,
        #   'reduction_4': ...,
        #   'reduction_5': ...,
        # }
        # reduction_x が進むにつれ解像度が下がります

        # 入力640前提で、
        #   reduction_3 → 80x80 くらい
        #   reduction_4 → 40x40 くらい
        #   reduction_5 → 20x20 くらい
        # になることが多いのでこれをC3/C4/C5にする
        c3 = endpoints["reduction_3"]
        c4 = endpoints["reduction_4"]
        c5 = endpoints["reduction_5"]

        # lateralでchそろえ
        p5 = self._get_lateral("p5", c5.shape[1])(c5)
        p4 = self._get_lateral("p4", c4.shape[1])(c4)
        p3 = self._get_lateral("p3", c3.shape[1])(c3)

        # top-down
        p4 = p4 + F.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        p3 = p3 + F.interpolate(p4, size=p3.shape[-2:], mode="nearest")

        # smooth
        p5 = self.smooth["p5"](p5)
        p4 = self.smooth["p4"](p4)
        p3 = self.smooth["p3"](p3)

        return [p3]
        #return p5,p4,p3
