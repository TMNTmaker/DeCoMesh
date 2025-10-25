


import torch
import torch.nn as nn
from yolox.utils import (
 postprocess3D
)
from typing import Tuple
import time
from yolox.models.losses import *


class TripleViewEncoder(nn.Module):
    """三面図を処理するカスタムエンコーダ"""
    def __init__(self):
        super().__init__()
        # 共有畳み込み層
        self.conv_shared = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
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
        self.norm3d = nn.GroupNorm(num_groups=2, num_channels=6, eps=1e-5, affine=True)

        self.softsign = nn.Softsign()
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
        
        
        from torch.cuda.amp import autocast

        # voxel: [B, C, D, H, W]
        voxel = torch.nan_to_num(voxel, nan=0.0, posinf=1e4, neginf=-1e4)
        voxel = voxel.clamp(-1e4, 1e4)

        # BNは確実にfp32で
        with autocast(enabled=False):
            
            y = self.norm3d(voxel.float())
        voxel_bn = y.to(voxel.dtype)
        
        return  self.softsign(voxel_bn)#torch.tanh(voxel_bn)

class TriView2CoordGrid(nn.Module):
    """エンドツーエンドネットワーク"""
    def __init__(self, in_channels=128):
        super().__init__()
        self.encoder = TripleViewEncoder()
        self.decoder = LowRankVoxelFusion(in_channels)


    def forward(self, x,labels=None,reduction_mag=None):
        
        front, top, side = self.encoder(x)
        vector_field_y = self.decoder(front, side,top)
        if self.training:
            assert labels is not None, "Training requires target labels" 
            assert reduction_mag is not None, "Training requires reduction_mag"
            # ベクトル場の損失を計算
            
            joint_list = labels
            torch.cuda.synchronize(); t0 = time.time()    
            vector_field_t,ignore_mask= self.data_procces_torch(vector_field_y,joint_list,reduction_mag)
            torch.cuda.synchronize(); t1 = time.time()
            print(f"Label grid build time: {t1-t0:.4f}s")            
            
            
            torch.cuda.synchronize(); t0 = time.time()
            groundpolygon = postprocess3D(vector_field_t,reduction_mag)
            torch.cuda.synchronize(); t1 = time.time()
            predictpolygon = postprocess3D(vector_field_y,reduction_mag)
            torch.cuda.synchronize(); t2 = time.time()
            print(f"Ground polygon processing time: {t1-t0:.4f}s  Predict polygon processing time: {t2-t1:.4f}s")

            
            loss,loss_offset,loss_target,loss_chamfer,loss_IoU3D=\
                self.loss_all(
                    vector_field_y,
                    vector_field_t,
                    groundpolygon, 
                    predictpolygon, 
                    ignore_mask,
                    reduction_mag)

            return loss,loss_offset,loss_target,loss_chamfer,loss_IoU3D  
        else:
            pred=postprocess3D(vector_field_y,8)
            #groups_as_list(vector_field_y[1])
            return pred[0],pred[1]#[0],groups_as_list(vector_field_y[1])
    
    
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
            pafs:         (B, 6, D, H, W)[off_z, off_y, off_x, tgt_z, tgt_y, tgt_x]  
            ignore_mask:  [B, 1,  D,H, W]  (0=valid, 1=ignore)
        """
        device = x.device
        B, _,D, H, W = x.shape
        r = float(reduction_mag)
        pafs   = torch.zeros((B, 6, D, H, W), device=device, dtype=torch.float32) 
        
        jl = joint_list.to(device=device, dtype=torch.float32)  # [B,N,F,4,3]
        
        
        
        B_, N, F, _, _ = jl.shape
        assert B_ == B
        # 事前に 26近傍の厚みテーブルを用意（半ボクセル単位: 0.5, √2/2, √3/2）
        import math
        # しきい値カーネル（z,y,x）
        sqrt3_2 = math.sqrt(3.0) * 0.5        # 0.866025403784...
        sqrt2_2 = math.sqrt(2.0) * 0.5        # 0.707106781186...
        demc2_2 = 0.5


        th_kernel = torch.tensor([
            [[sqrt3_2, sqrt2_2, sqrt3_2],
            [sqrt2_2, demc2_2, sqrt2_2],
            [sqrt3_2, sqrt2_2, sqrt3_2]],

            [[sqrt2_2, demc2_2, sqrt2_2],
            [demc2_2, demc2_2, demc2_2],
            [sqrt2_2, demc2_2, sqrt2_2]],

            [[sqrt3_2, sqrt2_2, sqrt3_2],
            [sqrt2_2, demc2_2, sqrt2_2],
            [sqrt3_2, sqrt2_2, sqrt3_2]],
        ], device=  "cuda", dtype=torch.float32)#*0.75  # (3,3,3)

        def _k_smallest_mask(dist: torch.Tensor, k: int = 1) -> torch.Tensor:
            """
            dist: 任意形状（最後は空間格子）。全体の中から距離の小さい順に k 個だけ True。
            """
            flat = dist.view(-1)
            k = min(k, flat.numel())
            # k個の閾値を topk で取得（小さい方からk個 → largest=False）
            vals, idx = torch.topk(flat, k, largest=False, sorted=False)
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask[idx] = True
            return mask.view_as(dist)
        def _single_argmin_mask(dist: torch.Tensor) -> torch.Tensor:
            """
            dist: (..., dz, dy, dx)
            戻り値: dist と同形状の bool。グローバル最小の1点のみ True。
            """
            flat_idx = dist.view(-1).argmin()
            mask = torch.zeros_like(dist, dtype=torch.bool)
            mask.view(-1)[flat_idx] = True
            return mask


        def draw_view(b: int, vertices: torch.Tensor):
            # 頂点列をループエッジとして扱う（v[i-1] -> v[i]）
            V = vertices.shape[0]
            if V < 2:
                return

            for v_ix in range(V):
                # A3,B3 をスケール1/rでグリッド座標へ
                A3 = vertices[(v_ix - 1) % V] / r
                B3 = vertices[v_ix] / r

                vec = B3 - A3               # [3]
                norm2 = (vec * vec).sum()
                if norm2.item() < 1e-8:
                    continue

                vx, vy, vz = vec

                # 対象領域（半ボクセルの余白付き）
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

                # 線分ABへの最近点H（各セル中心からの射影）
                APx = Xc - A3[0]; APy = Yc - A3[1]; APz = Zc - A3[2]
                denom = vx * vx + vy * vy + vz * vz + 1e-12
                t = (APx * vx + APy * vy + APz * vz) / denom
                t = t.clamp_(0.0, 1.0)
                Hx = A3[0] + t * vx
                Hy = A3[1] + t * vy
                Hz = A3[2] + t * vz

                # A/B への相対オフセット（最近の始点側を採用）
                off_xa = A3[0] - Xc; off_ya = A3[1] - Yc; off_za = A3[2] - Zc
                off_xb = B3[0] - Xc; off_yb = B3[1] - Yc; off_zb = B3[2] - Zc
                dA2 = off_xa*off_xa + off_ya*off_ya + off_za*off_za
                dB2 = off_xb*off_xb + off_yb*off_yb + off_zb*off_zb
                use_B = dB2 < dA2

                # A/Bの“最も近い3点”だけ残す（==min をやめ、argmin で単一点）
                # A側
                a_mask =_k_smallest_mask(dA2)
                # B側
                b_mask = _k_smallest_mask(dB2)
                keep_mask = a_mask | b_mask

                off_x = torch.where(use_B, off_xb, off_xa)
                off_y = torch.where(use_B, off_yb, off_ya)
                off_z = torch.where(use_B, off_zb, off_za)

                # 最小点のみ残す
                off_x = off_x * keep_mask
                off_y = off_y * keep_mask
                off_z = off_z * keep_mask

                # tgt（H→A もしくは H→B の単位ベクトル、近い方）
                ACx = A3[0] - Hx; ACy = A3[1] - Hy; ACz = A3[2] - Hz
                BCx = B3[0] - Hx; BCy = B3[1] - Hy; BCz = B3[2] - Hz
                dA = torch.sqrt(ACx*ACx + ACy*ACy + ACz*ACz + 1e-12)
                dB = torch.sqrt(BCx*BCx + BCy*BCy + BCz*BCz + 1e-12)
                use_B2 = dB < dA

                tgt_x = torch.where(use_B2, BCx, ACx)
                tgt_y = torch.where(use_B2, BCy, ACy)
                tgt_z = torch.where(use_B2, BCz, ACz)
                ms = int(math.ceil(math.sqrt(D*D + H*H + W*W)))
                    # 両方向なので片側あたり T = ceil(ms/2)
                T = max(1, (ms + 1)//2)
                tgt_x /= T
                tgt_y /= T
                tgt_z /= T
                

                # 中点Pに最も近い“tgtの終端”一つだけ残す（argminで単一点）
                Px = (A3[0] + B3[0]) * 0.5
                Py = (A3[1] + B3[1]) * 0.5
                Pz = (A3[2] + B3[2]) * 0.5
                # 単位ベクトルなのでそのまま1ステップ先を終端とする
                ex = Xc + tgt_x
                ey = Yc + tgt_y
                ez = Zc + tgt_z
                d_end = torch.sqrt((ex - Px)**2 + (ey - Py)**2 + (ez - Pz)**2)
                keep_tgt = _single_argmin_mask(d_end)
                tgt_x = tgt_x * keep_tgt
                tgt_y = tgt_y * keep_tgt
                tgt_z = tgt_z * keep_tgt

                # 書き込み：未使用セルかつ draw の位置だけに限定して書く
                patch = pafs[b, :, min_z:max_z, min_y:max_y, min_x:max_x]  # [6, dz, dy, dx]

                vals = [off_z, off_y, off_x, tgt_z, tgt_y, tgt_x]  # [6, dz, dy, dx]
                for i, val in enumerate(vals):
                    free = (patch[i] == 0) #& draw
                    patch[i][free] = val[free]
            return
        # 全要素ゼロのエッジを除外（[B,N,F]） 
        valid_mask = jl.abs().sum(dim=(3,4)) > 0
        for b in range(B):
            # flatten: [N,F] -> [M]
            idxs = torch.nonzero(valid_mask[b], as_tuple=False)  # [K,2] (n,e)
            if idxs.numel() == 0:
                continue
            for nf in idxs:
                n, f = int(nf[0].item()), int(nf[1].item())
                draw_view(b, jl[b, n, f])  # xyz
            block = pafs[:, :].sum(dim=1).abs()             # [B,6,D,H,W]
            
            ignore_mask = torch.stack([(block > 1e-5)], 
                                      dim=1).to(torch.uint8)
        
        if out_dtype is None:
            
            out_dtype = x.dtype
        return pafs.to(out_dtype), ignore_mask

            
    def loss_all(self,
                    vector_field_y, 
                    vector_field_t,
                    groundpolygon, 
                    predictpolygon,
                    ignore_mask,
                    lambda_offset=10.0,
                    lambda_target=10.0,
                    lambda_chamfer=1.0,
                    lambda_3dIoU=5.0
                    ):
        torch.cuda.synchronize(); t2 = time.time()
        loss_chamfer = chamfer_distance(predictpolygon, groundpolygon)
        _,loss_IoU3D = IoU3D_voxel(predictpolygon, groundpolygon)
        torch.cuda.synchronize(); t3 = time.time()
        print(f"Chamfer&IoU3D loss calculation time: {t3-t2:.4f}s")
        
        loss_offset = lambda_offset*masked_mse_loss(vector_field_y[:, [0,1,2]], 
                                                    vector_field_t[:, [0,1,2]],
                                                    ignore_mask)
        loss_target = lambda_target*masked_mse_loss(vector_field_y[:, [3,4,5]], 
                                                    vector_field_t[:, [3,4,5]],
                                                    ignore_mask)
        loss_total = loss_offset + loss_target  +\
        lambda_chamfer*torch.log(loss_chamfer) + lambda_3dIoU*loss_IoU3D

        return loss_total,loss_offset,loss_target,loss_chamfer, loss_IoU3D




class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

def groups_as_list(rows: torch.Tensor):
    """
    rows: (N, M) int tensor (cpu/gpu ok)
    Rule: Two rows are in the same group if they share >=1 number (transitively).
    Returns: Python list of groups -> list of row-lists (each row is a list[int])
    """
    rows_cpu = rows.detach().to("cpu").long()
    N, M = rows_cpu.shape

    uf = UnionFind(N)
    first_row_of_value = {}

    # link rows that share any value
    for i in range(N):
        vals = torch.unique(rows_cpu[i]).tolist()
        for v in vals:
            if v not in first_row_of_value:
                first_row_of_value[v] = i
            else:
                uf.union(i, first_row_of_value[v])

    # collect rows per root
    root_to_rows = {}
    for i in range(N):
        r = uf.find(i)
        root_to_rows.setdefault(r, []).append(i)

    # sort groups by smallest row index; keep original row order inside group
    groups_idx = [sorted(v) for v in root_to_rows.values()]
    groups_idx.sort(key=lambda xs: xs[0])

    # convert to pure Python list-of-lists-of-ints
    groups = [rows_cpu[idxs].tolist() for idxs in groups_idx]
    return groups




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