


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from yolox.utils import (
 postprocess3D
)
from typing import Tuple
import time
from yolox.models.losses import chamfer_distance,IoU3D


class TripleViewEncoder(nn.Module):
    """三面図を処理するカスタムエンコーダ"""
    def __init__(self):
        super().__init__()
        # 共有畳み込み層
        self.conv_shared = nn.Sequential(
            nn.Conv2d(12, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            )
        # ビュー別処理ブランチ
        self.conv_front = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.conv_top = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.conv_side = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        
    def forward(self, x):
        
        #x: (B, 6 ,H,W) -> (B, 2, H, W),(B, 2, H, W),(B, 2, H, W) 
        #x1, x2, x3 = torch.split(x, 2, dim=1)
        
        #xf = self.conv_shared(x1)
        #xs= self.conv_shared(x2)
        #xt = self.conv_shared(x3)
        x = self.conv_shared(x)
        # ビュー別処理
        front = self.conv_front(x)
        side = self.conv_side(x)
        top = self.conv_top(x)
        
        return front, top, side# (B, 128, H/4, W/4)

class LowRankVoxelFusion(nn.Module):
    """H×W×D座標グリッドを出力するデコーダ"""
    def __init__(self, in_channels):
        super().__init__()
        self.out_channels = 216
        assert self.out_channels % 6 == 0, "in_channels must be a multiple of 6"
        self.decoder_front = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels, self.out_channels, 3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            #nn.Flatten(2, 3)
        )
        self.decoder_side = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels, self.out_channels, 3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            #nn.Flatten(2, 3)
        )
        self.decoder_top = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels, self.out_channels, 3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            #nn.Flatten(2, 3)
        )
        
    def forward(self, front,side,top):
        front = self.decoder_front(front)
        side = self.decoder_side(side)
        top = self.decoder_top(top)
        H,W=front.shape[-2:]
        W,Z=side.shape[-2:]
        voxel = torch.einsum(
            'bcfxy, bcfyz, bcfzx -> bczyx',
            front.view(-1, 6, self.out_channels//6, H, W),
            side.view(-1, 6, self.out_channels//6, W, Z),
            top.view(-1, 6, self.out_channels//6, Z, H),
        ) / (self.out_channels/6)
        return  torch.tanh(voxel)

class TriView2CoordGrid(nn.Module):
    """エンドツーエンドネットワーク"""
    def __init__(self, in_channels=128):
        super().__init__()
        self.encoder = TripleViewEncoder()
        self.decoder = LowRankVoxelFusion(in_channels)
        
    def forward(self, x,labels=None,reduction_mag=None):
        
        front, top, side = self.encoder(x)
        coord_grid = self.decoder(front, side,top)
        if self.training:
            assert labels is not None, "Training requires target labels" 
            assert reduction_mag is not None, "Training requires reduction_mag"
            # 無効体素マスク（0=valid, 1=ignore）: 和ベクトルの絶対値 < 1e-5
            with torch.no_grad():
                # 4ch 和ノルムではなく、元コード同様「6ch 合計の絶対値 < eps」で判定
                mag = coord_grid.sum(dim=1).abs()                   # (B, D, H, W)
                ignore_mask = (mag < 1e-5)                          # bool でも可

            # ベクトル場の損失を計算
            
            joint_list = labels
            torch.cuda.synchronize(); t0 = time.time()    
            label_grid_3D,ignore_mask= self.data_procces_torch(coord_grid,joint_list,reduction_mag)
            torch.cuda.synchronize(); t1 = time.time()
            # 必要なら dtype を coord_grid に合わせる
            #label_grid_3D = label_grid_3D.to(coord_grid.dtype)
            torch.cuda.synchronize(); t1 = time.time()
            
            loss,loss_offset,loss_chamfer,loss_IoU3D= self.loss_all(
                    coord_grid, 
                    label_grid_3D, 
                    ignore_mask,
                    reduction_mag)
            torch.cuda.synchronize(); t2 = time.time()
            print(f"Label grid build time: {t1-t0:.4f}s, Loss calculation time: {t2-t1:.4f}s")
            return loss,loss_offset,loss_chamfer,loss_IoU3D  
        else:
            return coord_grid#(B,6,D,H,W)
    
    @torch.no_grad()
    def data_procces_torch(
        self,
        x: torch.Tensor,  
        joint_list: torch.Tensor,        # [B, N, F, 4, 3]  四角形x(x,y,z)
        reduction_mag: float,
        out_dtype: torch.dtype | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        """
        Returns:
            pafs:         (B, 6, D, H, W)[off_z, off_y, off_x, tgt_z, tgt_y, tgt_x,
                                            face_normal1_z, face_normal1_y, face_normal1_x]  
            ignore_mask:  [B, 1,  D,H, W]  (0=valid, 1=ignore)
        """
        device = x.device
        B, _,D, H, W = x.shape
        r = float(reduction_mag)
        pafs   = torch.zeros((B, 6, D, H, W), device=device, dtype=torch.float32)
        #normal_hits = torch.zeros((B, 1, D, H, W), device=device, dtype=torch.int32)    # 寄与数
        
        jl = joint_list.to(device=device, dtype=torch.float32)  # [B,N,F,4,3]
        
        
        
        B_, N, F, _, _ = jl.shape
        assert B_ == B
        # 事前に 26近傍の厚みテーブルを用意（半ボクセル単位: 0.5, √2/2, √3/2）
        sqrt3_2 = 0.866
        sqrt2_2 = 0.707
        demc2_2 = 0.500
        th_kernel = torch.tensor([
                [[sqrt3_2,sqrt2_2,sqrt3_2],
                [sqrt2_2,demc2_2,sqrt2_2],
                [sqrt3_2,sqrt2_2,sqrt3_2]],
                [[sqrt2_2,demc2_2,sqrt2_2],
                [demc2_2,demc2_2,demc2_2],
                [sqrt2_2,demc2_2,sqrt2_2]],
                [[sqrt3_2,sqrt2_2,sqrt3_2],
                [sqrt2_2,demc2_2,sqrt2_2],
                [sqrt3_2,sqrt2_2,sqrt3_2]],
            ], device=device, dtype=torch.float32)*0.75  # [3,3,3]
        # 全要素ゼロのエッジを除外（[B,N,F]）
        valid_mask = jl.abs().sum(dim=(3,4)) > 0
        
        def draw_view(b: int, vertices: torch.Tensor):
            #CCWなので任意の3点から法線ベクトルを求める
            v0 = vertices[0]
            v1 = vertices[1]
            v2 = vertices[2]

            # 辺ベクトルを計算
            e1 = v1 - v0
            e2 = v2 - v0

            # 外積で法線を計算
            cross = torch.cross(e1, e2)
            normal = cross / (torch.sqrt((cross*cross).sum()) + 1e-8) 

            for v_ix in range(len(vertices)):
                    
                # A3,B3 をスケーリング
                A3 = vertices[v_ix-1] / r
                B3 = vertices[v_ix] / r
                sqrt3 = 1.7320508075688772
                
                vec  = B3 - A3               # [3]
                norm2 = (vec *vec).sum()
                # scalar
                if (norm2 < 1e-8).item():
                    continue
                vx, vy, vz = vec

                min_x = int(torch.clamp(torch.floor(torch.min(A3[0], B3[0]) - 0.5), 0, W).item())
                max_x = int(torch.clamp(torch.ceil (torch.max(A3[0], B3[0]) + 0.5), 0, W).item())
                min_y = int(torch.clamp(torch.floor(torch.min(A3[1], B3[1]) - 0.5), 0, H).item())
                max_y = int(torch.clamp(torch.ceil (torch.max(A3[1], B3[1]) + 0.5), 0, H).item())
                min_z = int(torch.clamp(torch.floor(torch.min(A3[2], B3[2]) - 0.5), 0, D).item())
                max_z = int(torch.clamp(torch.ceil (torch.max(A3[2], B3[2]) + 0.5), 0, D).item())
                if (max_x <= min_x) or (max_y <= min_y) or (max_z <= min_z):
                    continue

                zs = torch.arange(min_z, max_z, device=device, dtype=torch.float32)
                ys = torch.arange(min_y, max_y, device=device, dtype=torch.float32)
                xs = torch.arange(min_x, max_x, device=device, dtype=torch.float32)
                Zc, Yc, Xc = torch.meshgrid(zs + 0.5, ys + 0.5, xs + 0.5, indexing="ij")  # (dz,dy,dx)

                
                APx = Xc - A3[0]; APy = Yc - A3[1]; APz = Zc - A3[2]
                denom = vx*vx + vy*vy + vz*vz + 1e-12
                t = (APx * vx + APy * vy + APz * vz) / denom
                t = t.clamp_(0.0, 1.0)
                Hx = A3[0] + t * vx
                Hy = A3[1] + t * vy
                Hz = A3[2] + t * vz
                dist = torch.sqrt((Xc - Hx)**2 + (Yc - Hy)**2 + (Zc - Hz)**2)
                
                dx_off = torch.clamp(torch.floor(Hx - Xc - 0.5), -1, 1).to(torch.long)  # ∈{-1,0,1}
                dy_off = torch.clamp(torch.floor(Hy - Yc - 0.5), -1, 1).to(torch.long)
                dz_off = torch.clamp(torch.floor(Hz - Zc - 0.5), -1, 1).to(torch.long)            # [0,1,2] へシフトして 3×3×3 テーブルを引く
                ix = dx_off + 1
                iy = dy_off + 1
                iz = dz_off + 1

                # 体積の各位置ごとにしきい値取得（broadcast でそのまま高次元インデクシング可）
                th_local = th_kernel[iz, iy, ix]  
                            
                off_xa_all = A3[0] - Xc
                off_ya_all = A3[1] - Yc
                off_za_all = A3[2] - Zc
                            
                
                off_xb_all = B3[0] - Xc
                off_yb_all = B3[1] - Yc
                off_zb_all = B3[2] - Zc
                
                
                distoffA2_3d = torch.sqrt(
                                    off_xa_all*off_xa_all + 
                                    off_ya_all*off_ya_all + 
                                    off_za_all*off_za_all)
                distoffB2_3d = torch.sqrt(
                                    off_xb_all*off_xb_all + 
                                    off_yb_all*off_yb_all + 
                                    off_zb_all*off_zb_all)
                mask_B_off = distoffB2_3d < distoffA2_3d

                min_dist_mask_offA3 = distoffA2_3d == distoffA2_3d.min()
                min_dist_mask_offB3 = distoffB2_3d == distoffB2_3d.min()
                min_dist_mask_off = min_dist_mask_offA3 | min_dist_mask_offB3
                
                off_x = torch.where(mask_B_off, off_xb_all, off_xa_all)
                off_y = torch.where(mask_B_off, off_yb_all, off_ya_all)
                off_z = torch.where(mask_B_off, off_zb_all, off_za_all)
                
                 
                off_x = torch.where(off_x==0, 1e-3, off_xa_all)
                off_y = torch.where(off_y==0, 1e-3, off_ya_all)
                off_z = torch.where(off_z==0, 1e-3, off_za_all)
                
                
                off_x = off_x*min_dist_mask_off 
                off_y = off_y*min_dist_mask_off
                off_z = off_z*min_dist_mask_off
                
                off_norm = torch.sqrt(off_x*off_x + off_y*off_y + off_z*off_z)
                use_perp = (off_norm > sqrt3)
                #off_x = torch.where(use_perp, 0, off_x)
                #off_y = torch.where(use_perp, 0, off_y)
                #off_z = torch.where(use_perp, 0, off_z)
                
                                        
                ACx = A3[0] - Hx; ACy = A3[1] - Hy; ACz = A3[2] - Hz
                BCx = B3[0] - Hx; BCy = B3[1] - Hy; BCz = B3[2] - Hz

                distA = torch.sqrt(ACx*ACx + ACy*ACy + ACz*ACz)
                distB = torch.sqrt(BCx*BCx + BCy*BCy + BCz*BCz)
                mask_B = distB < distA

                tgt_x = torch.where(mask_B, BCx, ACx)
                tgt_y = torch.where(mask_B, BCy, ACy)
                tgt_z = torch.where(mask_B, BCz, ACz)
                # tgt_*を単位ベクトルに変換
                tgt_norm = torch.sqrt(tgt_x*tgt_x + tgt_y*tgt_y + tgt_z*tgt_z + 1e-8)
                tgt_x = tgt_x / tgt_norm
                tgt_y = tgt_y / tgt_norm
                tgt_z = tgt_z / tgt_norm

                # --- 辺の中点（グリッド座標系で） ---
                Px = (A3[0] + B3[0]) * 0.5
                Py = (A3[1] + B3[1]) * 0.5
                Pz = (A3[2] + B3[2]) * 0.5

                # torchで丸め＆クリップ → グローバル整数座標
                gx = int(torch.clamp(torch.round(Px), 0, W - 1))
                gy = int(torch.clamp(torch.round(Py), 0, H - 1))
                gz = int(torch.clamp(torch.round(Pz), 0, D - 1))
                #lx = gx - min_x
                #ly = gy - min_y
                #lz = gz - min_z
                #if not (0 <= lx < (max_x - min_x) and 0 <= ly < (max_y - min_y) and 0 <= lz < (max_z - min_z)):
                #    continue  # 念のため
                # tgt_*ベクトルのノルム
                tgt_norm = torch.sqrt(tgt_x ** 2 + tgt_y ** 2 + tgt_z ** 2 + 1e-8)

                # tgt_*ベクトルの始点から中点Pへの距離
                tgt_end_x = Xc + tgt_x * tgt_norm
                tgt_end_y = Yc + tgt_y * tgt_norm
                tgt_end_z = Zc + tgt_z * tgt_norm
                dist_tgt_end_to_p = torch.sqrt((tgt_end_x - Px) ** 2 + (tgt_end_y - Py) ** 2 + (tgt_end_z - Pz) ** 2)

                # tgt_*ベクトルのうち、点Pに最も近いものだけ残す
                min_dist_mask = dist_tgt_end_to_p == dist_tgt_end_to_p.min()
                tgt_x = tgt_x * min_dist_mask
                tgt_y = tgt_y * min_dist_mask
                tgt_z = tgt_z * min_dist_mask
                lz,ly,lx = min_dist_mask.nonzero(as_tuple=False)[0]

                #off領域にも法線ベクトルマッピング
                normal_x = torch.where(use_perp, 0, normal[0])
                normal_y = torch.where(use_perp, 0, normal[1])
                normal_z = torch.where(use_perp, 0, normal[2])
                normal_x[lz,ly,lx] = normal[0]
                normal_y[lz,ly,lx] = normal[1]
                normal_z[lz,ly,lx] = normal[2]
                
                draw = dist <= th_local
                #if not draw.any().item():
                #    return

                patch = pafs[b, :, min_z:max_z,min_y:max_y, min_x:max_x]
                
                vals = [off_z,off_y, off_x,
                        tgt_z,tgt_y,tgt_x,
                        #normal_x,normal_y,normal_z,
                        ]
                for i, val in enumerate(vals):
                    mask = (patch[i] != 0)
                    if i < 3:
                        patch[i][~mask] = val[~mask] * draw[~mask]    
                    elif i < 6:    
                        patch[i][~mask] = val[~mask]
                    #else:
                    #    if i==6:
                    #        normal_hits[b,0, gz, gy, gx] += 1
                    #    patch[i][~mask] += val[~mask] #* draw[~mask] 
                blk=0
            return
        # 1面ずつ
        for b in range(B):
            # flatten: [N,F] -> [M]
            idxs = torch.nonzero(valid_mask[b], as_tuple=False)  # [K,2] (n,e)
            if idxs.numel() == 0:
                continue
            flag=False
            for nf in idxs:
                n, f = int(nf[0].item()), int(nf[1].item())
                #debug
                #if n in [0,1,6,7,8,9,10,11]:
                #    flag=True
                    #A3, B3 = jl[b, n, f, 0], jl[b, n, f, 1]  # [3]
                draw_view(b, jl[b, n, f])  # xyz
                #elif flag:
                #    break
            #hits = normal_hits.to(pafs.dtype)                   # float化
            #mask = hits > 0
#
#            # 0割対策（ヒット無しはそのままにしたい場合は where を使う）
            #safe_hits = torch.where(mask, hits, torch.ones_like(hits))
#
#            # 平均
            #pafs[:, 3, ...] = torch.where(mask[:,0], pafs[:, 3, ...] / safe_hits[:,0], pafs[:, 3, ...])
            #pafs[:, 4, ...] = torch.where(mask[:,0], pafs[:, 4, ...] / safe_hits[:,0], pafs[:, 4, ...])
            #pafs[:, 5, ...] = torch.where(mask[:,0], pafs[:, 5, ...] / safe_hits[:,0], pafs[:, 5, ...])

            block = pafs[:, :]              # [B,6,D,H,W]
            
            power = (block *block).sum(dim=1).sqrt()
            ignore_mask = (power < 1e-5).to(torch.uint8).unsqueeze(1)  # [B,1,D,H,W]

        
        if out_dtype is None:
            out_dtype = x.dtype
        return pafs.to(out_dtype), ignore_mask

            
    def loss_all(self,
                    vector_field_y, 
                    vector_field_t, 
                    vector_field_masks,mag):

        def mean_square_error(pred, target):
            assert pred.shape == target.shape, 'x and y should in same shape'
            return torch.sum((pred - target) ** 2) / target.nelement()
        

        loss_total = 0

        torch.cuda.synchronize(); t0 = time.time()
        groundpolygon = postprocess3D(vector_field_t,mag)
        torch.cuda.synchronize(); t1 = time.time()
        predictpolygon = postprocess3D(vector_field_y,mag)
        torch.cuda.synchronize(); t2 = time.time()
        print(f"Ground polygon processing time: {t1-t0:.4f}s  Predict polygon processing time: {t2-t1:.4f}s")
        loss_chamfer = chamfer_distance(predictpolygon, groundpolygon)
        _,loss_IoU3D = IoU3D(predictpolygon, groundpolygon)
        torch.cuda.synchronize(); t3 = time.time()
        print(f"Chamfer loss calculation time: {t3-t2:.4f}s")
        pafs_off_y = torch.stack((
            vector_field_y[:, 0],
            vector_field_y[:, 1],
            vector_field_y[:, 2],
        ), dim=1)
        pafs_off_t = torch.stack((
            vector_field_t[:, 0],
            vector_field_t[:, 1],
            vector_field_t[:, 2],
        ), dim=1)
        
        loss_offset = mean_square_error(pafs_off_y, pafs_off_t)
        loss_total += loss_offset + loss_chamfer + loss_IoU3D

        return loss_total,loss_offset,loss_chamfer, loss_IoU3D


def create_mask_from_target(target_grid, eps=1e-6):
    """
    target_gridから有効ピクセルマスクを生成
    
    引数:
        target_grid: ターゲット座標グリッド (B, H, W, 3)
        eps: 浮動小数点誤差を考慮した閾値 (デフォルト: 1e-6)
        
    戻り値:
        mask: 有効ピクセルを示すバイナリマスク (B, H, W)
    """
    # 1. 座標が無効値(0,0,0)かどうかを判定
    zero_mask = torch.all(torch.abs(target_grid) < eps, dim=-1)
    
    # 2. NaN値のチェック
    nan_mask = torch.isnan(target_grid).any(dim=-1)
    
    # 3. 無限大値のチェック
    inf_mask = torch.isinf(target_grid).any(dim=-1)
    
    # 4. すべてのチェックを統合: 有効な座標 = ゼロでない AND NaNでない AND 無限大でない
    valid_mask = ~(zero_mask | nan_mask | inf_mask)
    
    return valid_mask
def dice_loss(pred, target, eps=1e-6):
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    intersection = (pred * target).sum(1)
    union = pred.sum(1) + target.sum(1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()

def masked_bce_loss(pred, target, mask):
    loss = F.binary_cross_entropy(pred, target, reduction='none')
    return (loss * mask).sum() / mask.sum()

def asymmetric_loss(pred, target, gamma_pos=0.0, gamma_neg=4.0):
    eps = 1e-8
    x_pos = pred
    x_neg = 1 - pred

    pos_loss = target * torch.log(x_pos + eps) * (1 - x_pos) ** gamma_pos
    neg_loss = (1 - target) * torch.log(x_neg + eps) * (x_pos) ** gamma_neg
    loss = - (pos_loss + neg_loss)
    return loss.mean()

def asymmetric_mse_loss(pred, target, alpha=1.0, beta=0.1):
    """
    pred: (B, H, W, D) - sigmoid 出力など
    target: (B, H, W, D) - binary {0,1}
    alpha: 前景（target=1）の重み
    beta: 背景（target=0）の重み
    """
    foreground = (target == 1).float()
    background = 1.0 - foreground

    diff_sq = (pred - target) ** 2
    weighted_loss = alpha * foreground * diff_sq + beta * background * diff_sq
    return weighted_loss.mean()

def masked_mse_loss(pred, target, mask, eps=1e-6):
    """
    pred: (B, H, W, D)
    target: (B, H, W, D)
    mask: (B, H, W, D) または (B, H, W)
    """
    # mask を pred/target と同じ形状に拡張
    if mask.dim() == pred.dim() - 1:
        mask = mask.unsqueeze(-1).expand_as(pred)
    else:
        mask = mask.expand_as(pred)
    
    diff = (pred - target) ** 2
    masked_diff = diff * mask  # 無効領域は 0 に
    loss = masked_diff.sum() / (mask.sum() + eps)  # avoid divide by zero
    return loss
