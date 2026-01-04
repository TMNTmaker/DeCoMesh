import torch
import torch.nn as nn
import torch.nn.functional as F



from PIL import Image, ImageDraw
import numpy as np



class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False, groups=groups)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ResidualBlock(nn.Module):
    """出力チャネル数は一定、空間サイズを維持する3x3×2の基本ResBlock"""
    def __init__(self, ch):
        super().__init__()
        self.conv1 = ConvBNAct(ch, ch, k=3, s=1, p=1)
        self.conv2 = ConvBNAct(ch, ch, k=3, s=1, p=1, act=False)
        self.act   = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        return self.act(out)

class ClassNet(nn.Module):
    """
    入力: (B, 256, 80, 80)
    出力: (B, num_classes, 80, 80)
    """
    def __init__(self, in_channels=256, hidden=128, num_classes=20, dropout_p=0.1):
        super().__init__()
        # 前処理(チャネル圧縮はしない。必要ならhiddenを変えてOK)
        self.num_classes = num_classes
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, hidden, k=3, s=1, p=1),
            ResidualBlock(hidden),
            ResidualBlock(hidden),
            nn.Dropout2d(p=dropout_p),
            ConvBNAct(hidden, hidden, k=3, s=1, p=1),
        )
        # 1x1でクラス数へ
        self.classifier_xyp = nn.Conv2d(hidden, self.num_classes, kernel_size=1, bias=True)
        self.classifier_yzp = nn.Conv2d(hidden, self.num_classes, kernel_size=1, bias=True)
        self.classifier_zxp = nn.Conv2d(hidden, self.num_classes, kernel_size=1, bias=True)
        self.classifier_xym = nn.Conv2d(hidden, self.num_classes, kernel_size=1, bias=True)
        self.classifier_yzm = nn.Conv2d(hidden, self.num_classes, kernel_size=1, bias=True)
        self.classifier_zxm = nn.Conv2d(hidden, self.num_classes, kernel_size=1, bias=True)
        # NOTE:
        # - Softmax は「レイヤー(Conv2d)」ではなく「出力(logits)」に適用するもの。
        # - さらに学習時は `F.cross_entropy` が内部で log-softmax を取るため、
        #   ここで softmax を掛けると学習が壊れる（数値的にも不利）。
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _polygon_mask(self,H: int, W: int, poly_xy: np.ndarray) -> torch.Tensor:
        """
        poly_xy: (K, 2) の np.array、(x, y) 画素座標（特徴マップ解像度）
        戻り値: (H, W) の bool Tensor（True がポリゴン内部）
        """
        # 画素境界を超えないようクリップ
        poly_xy = np.asarray(poly_xy, dtype=np.float32)
        poly_xy[:, 0] = np.clip(poly_xy[:, 0], 0, W - 1)
        poly_xy[:, 1] = np.clip(poly_xy[:, 1], 0, H - 1)

        # PIL で塗りつぶし
        img = Image.new("1", (W, H), 0)  # (W, H) 注意: PILは横幅が先
        draw = ImageDraw.Draw(img)
        # PILは(x, y)順、float OK（内部で補間）。頂点数 >= 3 のときのみ描画
        if poly_xy.shape[0] >= 3:
            draw.polygon([tuple(p) for p in poly_xy], outline=1, fill=1)
        mask = np.array(img, dtype=bool)   # (H, W)
        return torch.from_numpy(mask)

    def build_targets_from_mesh(self,
        outputs: torch.Tensor,      # (B, C, H, W)
        category: list[list],     # (B, N)  各オブジェクトのクラスID
        mesh: torch.Tensor,         # (B, N, 6, 4, 3) = (バッチ, オブジェクト, 面, 頂点, xyz)
        direction: str = 'xyp' ,    # 'xy','yz','zx'のいずれか
        reduction_mag: int = 8,
        bg_id: int = 0,
    ) -> torch.Tensor:
        """
        各オブジェクトの面ポリゴン（元画像座標系のx,y）を HxW 解像度に縮小して塗り、クラスIDで埋める。
        近い面が手前に来るよう Z の平均で奥→手前に描画（奥から順に塗り、手前で上書き）。

        期待する前提:
        - mesh の x,y は元画像の画素座標（幅×高さ）で、z は深度（数値が大きいほど遠い想定）。
        - 4頂点の「面」×6 は、直方体メッシュを想定（四角面）。三角面等でも動作可（3頂点以上ならOK）。
        - 出力の (H, W) は `outputs.shape[2:]` に一致。
        """
        device = outputs.device
        B, C, H, W = outputs.shape
        assert direction in ('xyp','yzp','zxp','xym','yzm','zxm'), "direction must be 'xyp','yzp','zxp','xym','yzm','zxm'"
        if  'xy' in direction:
            coord_idx1, coord_idx2, coord_idx3 = 0, 1, 2  # x, y, z
        elif 'yz' in direction:
            coord_idx1, coord_idx2, coord_idx3 = 2, 1, 0  # y, z, x
        elif 'zx' in direction:  # 'zx'
            coord_idx1, coord_idx2, coord_idx3 = 2, 0, 1  # z, x, y
        if 'p' in direction:
            sign=1
        else:
            sign=-1
        # 初期化（背景ID）
        targets = torch.full((B, H, W), int(bg_id), dtype=torch.long, device=device)

        # numpy/PIL はCPUのみなので一旦CPUへ
        #category_np = category#.detach().cpu().numpy()
        mesh_np = mesh.detach().cpu().numpy()

        for b in range(B):
            cats_b = category[b]            # (N,)
            mesh_b = mesh_np[b]                # (N, 6, 4, 3)
            #6, 4, 3が全部ゼロのオブジェクトは無視
            mask_valid = (np.abs(mesh_b).sum(axis=(1,2,3)) > 0)  # (N,)
            mesh_b = mesh_b[mask_valid]
            N = mesh_b.shape[0]
            if N == 0:
                continue

            # オブジェクトの「平均Z」で奥→手前の順に並べる（Z大=遠い想定なので降順）
            # 形状: (N, 6, 4, 3) -> Zだけ抜き出して平均
            
            q_mean = mesh_b[..., coord_idx3].mean(axis=(1, 2))  # (N,)
            draw_order = np.argsort(-q_mean*sign)  # 降順（奥→手前）

            # このバッチの targets を一旦CPUで編集（最後にデバイスへ）
            tgt_b = targets[b].cpu()

            for idx in draw_order:
                if idx >= len(cats_b):
                    print("Warning: idx out of range in build_targets_from_mesh")
                    continue
                cls_id = int(cats_b[idx])
                faces = mesh_b[idx]            # (6, 4, 3)

                for f in range(faces.shape[0]):
                    # (4, 3) -> (4, 2) の (x, y) を使用
                    face_xyz = faces[f]        # (4, 3)
                    poly_nm = face_xyz[:, [coord_idx1,coord_idx2]]  # (4, 2) 画素座標（元画像）
                    # 特徴マップ解像度へ縮小
                    poly_nm = poly_nm / float(reduction_mag)
                    # マスク生成（HxW）
                    mask = self._polygon_mask(H, W, poly_nm)  # bool, CPU
                    # クラスで上書き（手前の面/物体が最後に上書きされる）
                    if mask.any():
                        tgt_b[mask] = cls_id

            # 戻す
            targets[b] = tgt_b.to(device)

        return targets

    
    def loss_all(self, outputs, category,mesh,reduction_mag, ignore_index: int = -1, bg_id: int = -1):
        """
        outputs: (B, num_classes, H, W) - モデルの出力
        category: (B, N) - 正解ラベル(クラスIDマップ)
        mesh: (B, N,6,4, 3) - メッシュ座標データ(B, N,面,点, xyz)
        reduction_mag: int - 特徴マップと元画像の縮小倍率
        
        targets: (B, H, W) - 正解ラベル(クラスIDマップ)
        """
        targets_xy_p = self.build_targets_from_mesh(outputs[:,0], category, mesh,direction="xyp", reduction_mag=reduction_mag, bg_id=bg_id)
        targets_xy_m = self.build_targets_from_mesh(outputs[:,1], category, mesh,direction="xym", reduction_mag=reduction_mag, bg_id=bg_id)
        targets_yz_p = self.build_targets_from_mesh(outputs[:,2], category, mesh,direction="yzp", reduction_mag=reduction_mag, bg_id=bg_id)
        targets_yz_m = self.build_targets_from_mesh(outputs[:,3], category, mesh,direction="yzm", reduction_mag=reduction_mag, bg_id=bg_id)
        targets_zx_p = self.build_targets_from_mesh(outputs[:,4], category, mesh,direction="zxp", reduction_mag=reduction_mag, bg_id=bg_id)
        targets_zx_m = self.build_targets_from_mesh(outputs[:,5], category, mesh,direction="zxm", reduction_mag=reduction_mag, bg_id=bg_id)
        
        
        loss  = F.cross_entropy(outputs[:,0], targets_xy_p, ignore_index=ignore_index)
        loss += F.cross_entropy(outputs[:,1], targets_xy_m, ignore_index=ignore_index)
        loss += F.cross_entropy(outputs[:,2], targets_yz_p, ignore_index=ignore_index)
        loss += F.cross_entropy(outputs[:,3], targets_yz_m, ignore_index=ignore_index)
        loss += F.cross_entropy(outputs[:,4], targets_zx_p, ignore_index=ignore_index)
        loss += F.cross_entropy(outputs[:,5], targets_zx_m, ignore_index=ignore_index)
        
        return loss
    
    
    def forward(self, x, category=None,mesh=None, reduction_mag=None):
        # x: (B, 256, 80, 80)
        feat = self.stem(x)                 # -> (B, hidden, 80, 80)
        logits_xyp = self.classifier_xyp(feat)      # -> (B, num_classes, 80, 80)
        logits_xym = self.classifier_xym(feat)      # -> (B, num_classes, 80, 80)
        logits_yzp = self.classifier_yzp(feat)      # -> (B, num_classes, 80, 80)
        logits_yzm = self.classifier_yzm(feat)      # -> (B, num_classes, 80, 80)
        logits_zxp = self.classifier_zxp(feat)      # -> (B, num_classes, 80, 80)
        logits_zxm = self.classifier_zxm(feat)      # -> (B, num_classes, 80, 80)
        #結合
        logits = torch.stack([logits_xyp,logits_xym,
                              logits_yzp,logits_yzm,
                              logits_zxp,logits_zxm],dim=1) #(B,6,num_classes,80,80)
        probs = torch.softmax(logits, dim=2)   # 確率が欲しいとき
        #pred  = logits.argmax(dim=1)           # ラベルマップが欲しいとき
        
        if self.training:
            assert category is not None, "category must be provided during training"
            assert mesh is not None, "mesh must be provided during training"
            assert reduction_mag is not None, "reduction_mag must be provided during training"
            
            
            loss = self.loss_all(logits, category,mesh,reduction_mag, ignore_index=-1, bg_id=self.num_classes-1)
            return loss,probs,feat
        else :
            return probs,feat
        
        
        
## --- 動作確認 ---
#if __name__ == "__main__":
#    B, C, H, W = 2, 256, 80, 80
#    num_classes = 20
#
#    model = ClassNet(in_channels=C, hidden=256, num_classes=num_classes, dropout_p=0.1)
#    x = torch.randn(B, C, H, W)
#    y = model(x)
#    print(y.shape)  # torch.Size([2, 21, 80, 80])
#
#    # 例: セグメンテーションの損失（CrossEntropy）
#    # target は (B, H, W) のクラスIDマップ
#    target = torch.randint(0, num_classes, (B, H, W))
#    loss = F.cross_entropy(y, target)
#    print("loss:", float(loss))
#