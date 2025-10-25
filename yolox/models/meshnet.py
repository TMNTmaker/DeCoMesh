import torch
import torch.nn as nn
from yolox.models.losses import *

import time
class MeshNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.stage_1 = Stage_1()
        self.stage_2 = Stage_x()
        self.stage_3 = Stage_x()
        #self.stage_4 = Stage_x()
        #self.stage_5 = Stage_x()
        #self.stage_6 = Stage_x()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x,labels=None,reduction_mag=None):
        pafs = []
        features = []
        feature_map = x
        h1,F1= self.stage_1(feature_map)
        pafs.append(h1)
        features.append(F1)
        h1,F1 = self.stage_2(torch.cat([h1, feature_map], dim = 1))
        pafs.append(h1)
        features.append(F1)
        h1,F1 = self.stage_3(torch.cat([h1, feature_map], dim = 1))
        pafs.append(h1)
        features.append(F1)
        #h1 = self.stage_4(torch.cat([h1, feature_map], dim = 1))
        #pafs.append(h1)
        #h1 = self.stage_5(torch.cat([h1, feature_map], dim = 1))
        #pafs.append(h1)
        #h1 = self.stage_6(torch.cat([h1, feature_map], dim = 1))
        #pafs.append(h1)
        if self.training:
            assert labels is not None, "labels must be provided during training"
            assert reduction_mag is not None, "reduction_mag must be provided during training"
            
            joint_list = labels
            torch.cuda.synchronize(); t0 = time.time()
            pafs_t,ignore_mask= self.data_procces_torch(feature_map,joint_list,reduction_mag)
            torch.cuda.synchronize(); t1 = time.time()
            loss_total,loss_offset,loss_target=self.loss_all(pafs,pafs_t,
                                                 ignore_mask,
                                                 reduction_mag)    
            torch.cuda.synchronize(); t2 = time.time()
            print(f"Data processing time: {t1-t0:.4f}s, Loss calculation time: {t2-t1:.4f}s")
            return loss_total,loss_offset,loss_target,pafs,features
        else:
            return pafs,features  
    
    
    #Data processing time: 1.3204s
    from typing import Tuple
    @torch.no_grad()
    def data_procces_torch(
        self,
        x: torch.Tensor,  
        joint_list: torch.Tensor,        # [B, N, F, 4, 3]  四角形(頂点×3D)
        reduction_mag: float,
        thickness: float = 0.5,
        out_dtype: torch.dtype | None = torch.float16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            pafs:         [B, 12, H, W]  (view: xy/yz/zx × 4ch = off_y, off_x, tgt_y, tgt_x)
            ignore_mask:  [B, 3,  H, W]  (0=valid, 1=ignore)
        """
        device = x.device
        B, _, H, W = x.shape
        r = float(reduction_mag)
        t2 = float(thickness * thickness)  # 距離判定は2乗で

        # --- 内部は float32 で計算、最後に out_dtype へ変換 ---
        pafs = torch.zeros((B, 12, H, W), device=device, dtype=torch.float32)

        jl = joint_list.to(device=device, dtype=torch.float32)  # [B,N,F,4,3]
        B_, N, F, V, C = jl.shape
        assert B_ == B and V == 4 and C == 3

        # --- 画像座標グリッドをキャッシュ（インスタンスごと） ---
        if not hasattr(self, "_paf_grid_cache"):
            self._paf_grid_cache = {}
        key = (device, H, W)
        if key in self._paf_grid_cache:
            Yc_full, Xc_full = self._paf_grid_cache[key]
        else:
            ys = torch.arange(H, device=device, dtype=torch.float32) + 0.5
            xs = torch.arange(W, device=device, dtype=torch.float32) + 0.5
            Yc_full, Xc_full = torch.meshgrid(ys, xs, indexing="ij")  # [H,W]
            self._paf_grid_cache[key] = (Yc_full, Xc_full)

        # --- エッジ列の抽出をベクトル化（各バッチ毎） ---
        def draw_edges_view(patch: torch.Tensor, A2: torch.Tensor, B2: torch.Tensor):
            """
            patch: [4, H, W]  (= off_y, off_x, tgt_y, tgt_x)
            A2,B2: [K, 2]     (この view の2D座標 / r)
            """
            K = A2.shape[0]
            if K == 0:
                return

            vec = B2 - A2                       # [K,2]
            norm2 = (vec * vec).sum(dim=1)      # [K]
            keep = norm2 >= 1e-8
            if not keep.any():
                return

            A2 = A2[keep]; B2 = B2[keep]
            vec = vec[keep]; norm2 = norm2[keep]
            vx = vec[:, 0]; vy = vec[:, 1]      # [K]

            # 各エッジの ROI（int）をまとめて計算し、個々に処理
            min_x = torch.clamp(torch.floor(torch.minimum(A2[:, 0], B2[:, 0]) - thickness), 0, W).to(torch.int64)
            max_x = torch.clamp(torch.ceil (torch.maximum(A2[:, 0], B2[:, 0]) + thickness), 0, W).to(torch.int64)
            min_y = torch.clamp(torch.floor(torch.minimum(A2[:, 1], B2[:, 1]) - thickness), 0, H).to(torch.int64)
            max_y = torch.clamp(torch.ceil (torch.maximum(A2[:, 1], B2[:, 1]) + thickness), 0, H).to(torch.int64)

            # 先着のみ書き込む（first-wins）ため、順序はそのまま
            for k in range(A2.shape[0]):
                x0 = int(min_x[k].item()); x1 = int(max_x[k].item())
                y0 = int(min_y[k].item()); y1 = int(max_y[k].item())
                if x1 <= x0 or y1 <= y0:
                    continue

                # 事前計算したフルグリッドからスライスを切り出すだけ（arange/meshgridを作らない）
                Xc = Xc_full[y0:y1, x0:x1]
                Yc = Yc_full[y0:y1, x0:x1]

                Ax, Ay = A2[k, 0], A2[k, 1]
                vxk, vyk = vx[k], vy[k]
                inv_norm2 = 1.0 / (norm2[k] + 1e-12)

                # 最近傍射影点 H = A + t v（線分内に clamp）
                APx = Xc - Ax
                APy = Yc - Ay
                t = (APx * vxk + APy * vyk) * inv_norm2
                t = t.clamp_(0.0, 1.0)
                Hx = Ax + t * vxk
                Hy = Ay + t * vyk

                # 厚み内判定（距離2乗）
                dx = Xc - Hx
                dy = Yc - Hy
                draw = (dx*dx + dy*dy) <= t2   # [h,w], bool

                # --- off ベクトル（A→P or 垂線）の分岐 ---
                off_x = Ax - Xc
                off_y = Ay - Yc
                # √2 の比較は 2 で代替（2乗ノルム）
                use_perp = (off_x*off_x + off_y*off_y) > 2.0
                # 垂線: (Hx-Xc, Hy-Yc)
                off_x = torch.where(use_perp, Hx - Xc, off_x)
                off_y = torch.where(use_perp, Hy - Yc, off_y)

                # --- tgt ベクトル（近い端点へ：A か B）---
                Bx, By = B2[k, 0], B2[k, 1]
                # C = (Xc+off) なので A/B から C へのベクトル
                Cx = Xc + off_x
                Cy = Yc + off_y
                ACx = Ax - Cx; ACy = Ay - Cy
                BCx = Bx - Cx; BCy = By - Cy

                # 近い方を採用
                distA2 = ACx*ACx + ACy*ACy
                distB2 = BCx*BCx + BCy*BCy
                chooseB = distB2 < distA2
                tgt_x = torch.where(chooseB, BCx, ACx)
                tgt_y = torch.where(chooseB, BCy, ACy)

                # クリップ（最大成分の絶対値が 1 を超えるときだけ正規化）
                max_abs = torch.maximum(tgt_x.abs(), tgt_y.abs())
                need_clip = max_abs > 1.0
                if need_clip.any():
                    tgt_norm = torch.sqrt(tgt_x*tgt_x + tgt_y*tgt_y) + 1e-12
                    scale = torch.where(need_clip, 1.0 / tgt_norm, torch.ones_like(tgt_norm))
                    tgt_x = tgt_x * scale
                    tgt_y = tgt_y * scale

                # --- 4 チャンネルへ first-wins 書き込み ---
                # 既に埋まっている画素は上書きしない
                # ここで draw を掛けることで帯域外は 0 を維持
                to_off_y = off_y * draw
                to_off_x = off_x * draw
                to_tgt_y = tgt_y * draw
                to_tgt_x = tgt_x * draw

                # それぞれ zeros の場所だけ更新
                p0 = patch[0, y0:y1, x0:x1]; m0 = (p0 == 0)
                p0[m0] = to_off_y[m0]
                p1 = patch[1, y0:y1, x0:x1]; m1 = (p1 == 0)
                p1[m1] = to_off_x[m1]
                p2 = patch[2, y0:y1, x0:x1]; m2 = (p2 == 0)
                p2[m2] = to_tgt_y[m2]
                p3 = patch[3, y0:y1, x0:x1]; m3 = (p3 == 0)
                p3[m3] = to_tgt_x[m3]

        # 面が全ゼロのものを除外
        valid_mask = jl.abs().sum(dim=(3, 4)) > 0  # [B,N,F]

        for b in range(B):
            if not valid_mask[b].any():
                continue

            faces_b = jl[b][valid_mask[b]]        # [M,4,3]
            if faces_b.numel() == 0:
                continue

            # 4 頂点から4エッジ (i-1 -> i) をベクトル化で生成
            A3 = torch.roll(faces_b, shifts=1, dims=1).reshape(-1, 3)  # [M*4,3]
            B3 = faces_b.reshape(-1, 3)                                # [M*4,3]

            # --- view ごとに 2D 座標を作って一括でエッジ描画 ---
            # xy
            A2_xy = (A3[:, [0, 1]] / r).contiguous()
            B2_xy = (B3[:, [0, 1]] / r).contiguous()
            draw_edges_view(pafs[b, 0:4], A2_xy, B2_xy)

            # yz
            A2_yz = (A3[:, [1, 2]] / r).contiguous()
            B2_yz = (B3[:, [1, 2]] / r).contiguous()
            draw_edges_view(pafs[b, 4:8], A2_yz, B2_yz)

            # zx
            A2_zx = (A3[:, [2, 0]] / r).contiguous()
            B2_zx = (B3[:, [2, 0]] / r).contiguous()
            draw_edges_view(pafs[b, 8:12], A2_zx, B2_zx)

        # --- ignore mask を作成（ch 合計が 0 の画素を ignore=1）---
        block_xy = pafs[:, 0:4].sum(dim=1).abs()
        block_yz = pafs[:, 4:8].sum(dim=1).abs()
        block_zx = pafs[:, 8:12].sum(dim=1).abs()
        ignore_mask = torch.stack([(block_xy > 1e-5),
                                (block_yz > 1e-5),
                                (block_zx > 1e-5)], dim=1).to(torch.uint8)

        if out_dtype is None:
            out_dtype = x.dtype
        return pafs.to(out_dtype), ignore_mask    
    
    
    
    def loss_all(self,
                     pafs_ys, 
                     pafs_t, 
                     ignore_mask,
                     lambda_offset=10.0,
                     lambda_target=10.0):
        
        loss_total = 0
        torch.cuda.synchronize(); t0 = time.time()
        
        # compute loss on each stage
        
        for pafs_y in pafs_ys:
           
           
            loss_offset = lambda_offset*mean_square_error(pafs_y[:, [0,1,4,5,8,9]], 
                                                        pafs_t[:, [0,1,4,5,8,9]])                      
            loss_target = lambda_target*mean_square_error(pafs_y[:, [2,3,6,7,10,11]], 
                                                        pafs_t[:, [2,3,6,7,10,11]])                      
            #
            #loss_offset_xy = lambda_offset*masked_charbonnier(pafs_y[:, [0,1]], 
            #                                            pafs_t[:, [0,1]],
            #                                            ignore_mask[:,0])           
            #loss_offset_yz = lambda_offset*masked_charbonnier(pafs_y[:, [4,5]], 
            #                                            pafs_t[:, [4,5]],
            #                                            ignore_mask[:,1])           
            #loss_offset_zx = lambda_offset*masked_charbonnier(pafs_y[:, [8,9]], 
            #                                            pafs_t[:, [8,9]],
            #                                            ignore_mask[:,2])           
            #
            #loss_target_xy = lambda_target*masked_charbonnier(pafs_y[:, [2,3]],
            #                                            pafs_t[:, [2,3]],
            #                                            ignore_mask[:,0])
            #loss_target_yz = lambda_target*masked_charbonnier(pafs_y[:, [6,7]],
            #                                            pafs_t[:, [6,7]],
            #                                            ignore_mask[:,1])
            #loss_target_zx = lambda_target*masked_charbonnier(pafs_y[:, [10,11]],
            #                                            pafs_t[:, [10,11]],
            #                                            ignore_mask[:,2])
            #
            #loss_offset = loss_offset_xy + loss_offset_yz + loss_offset_zx 
            #loss_target = loss_target_xy + loss_target_yz + loss_target_zx 
            loss_total += loss_offset + loss_target
        torch.cuda.synchronize(); t1 = time.time()
        print(f"mesh loss processing time: {t1-t0:.4f}s")
        return loss_total, loss_offset,loss_target



 
class Stage_1(nn.Module):
    def __init__(self):
        super(Stage_1, self).__init__()
        self.conv1_CPM_L1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        #self.conv2_CPM_L1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        #self.conv3_CPM_L1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4_CPM_L1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv5_CPM_L1 = nn.Conv2d(in_channels=256, out_channels=12, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.bn   = nn.BatchNorm2d(num_features=12)
        self.softsign = nn.Softsign()
        
        
    def forward(self, x):
        h1 = self.relu(self.conv1_CPM_L1(x)) # branch1
        #h1 = self.relu(self.conv2_CPM_L1(h1))
        #h1 = self.relu(self.conv3_CPM_L1(h1))
        F1 = self.relu(self.conv4_CPM_L1(h1))
        h1 = self.conv5_CPM_L1(F1)
        h1 = self.softsign(self.bn(h1))
        return h1,F1
    
class Stage_x(nn.Module):
    def __init__(self):
        super(Stage_x, self).__init__()
        self.conv1_L1 = nn.Conv2d(in_channels = 268, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        #self.conv2_L1 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        #self.conv3_L1 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        #self.conv4_L1 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        #self.conv5_L1 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.conv6_L1 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 1, stride = 1, padding = 0)
        self.conv7_L1 = nn.Conv2d(in_channels = 128, out_channels = 12, kernel_size = 1, stride = 1, padding = 0)
        self.relu = nn.ReLU()
        self.bn   = nn.BatchNorm2d(num_features=12)
        self.softsign = nn.Softsign()
        
    def forward(self, x):
        h1 = self.relu(self.conv1_L1(x)) # branch1
        #h1 = self.relu(self.conv2_L1(h1))
        #h1 = self.relu(self.conv3_L1(h1))
        #h1 = self.relu(self.conv4_L1(h1))
        #h1 = self.relu(self.conv5_L1(h1))
        F1 = self.relu(self.conv6_L1(h1))
        h1 = self.conv7_L1(F1)
        h1 = self.softsign(self.bn(h1))
        return h1,F1
