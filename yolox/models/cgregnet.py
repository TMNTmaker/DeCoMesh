


import torch
import torch.nn as nn
from yolox.utils import (
 postprocess3D
)
from typing import Tuple
import time
from yolox.models.losses import *
import math

class TripleViewEncoder(nn.Module):
    """三面図を処理するカスタムエンコーダ"""
    def __init__(self):
        super().__init__()
        # 共有畳み込み層
        self.conv_shared = nn.Sequential(
            nn.Conv2d(128*4, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3,stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3,stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=128, eps=1e-5, affine=True),
            nn.ReLU(),
            )
        # ビュー別処理ブランチ
        self.enc2d = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3,stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3,stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=128, eps=1e-5, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
                
        
    def forward(self, x):
        x = self.conv_shared(x)
        # ビュー別処理
        front = self.enc2d(x)
        side = self.enc2d(x)
        top = self.enc2d(x)
        
        return front, top, side# (B, 128, H/4, W/4)

class LowRankVoxelFusion(nn.Module):
    """H×W×D座標グリッドを出力するデコーダ"""
    def __init__(self, in_channels):
        super().__init__()
        self.out_channels = 216
        assert self.out_channels % 8 == 0, "in_channels must be a multiple of 8"
        
                # 2Dデコーダ（BN2d→GN2dに置換推奨）
        def dec2d():
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, self.out_channels, 3, padding=1),
                nn.GroupNorm(num_groups=8, num_channels=self.out_channels, eps=1e-5, affine=True),
            )
        self.decoder_front = dec2d()
        self.decoder_side  = dec2d()
        self.decoder_top   = dec2d()
        


        self.head3d = nn.Sequential(
            nn.GroupNorm(num_groups=24, num_channels=72, eps=1e-5, affine=True),
            nn.Conv3d(72, 32, kernel_size=1, bias=True),
            nn.GroupNorm(num_groups=8, num_channels=32, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 10, kernel_size=1, bias=True),  # [dir(3), mag(1), mask(1)]
        )
        self.softplus = nn.Softplus()
        
        self.softsign = nn.Softsign()
    def forward(self, front,side,top):
        front = self.decoder_front(front)
        side = self.decoder_side(side)
        top = self.decoder_top(top)
        H,W=front.shape[-2:]
        W,Z=side.shape[-2:]
        voxel = torch.einsum(
            'bcfxy, bcfyz, bcfzx -> bczyx',
            front.view(-1, 72, self.out_channels//72, H, W),
            side.view(-1, 72, self.out_channels//72, W, Z),
            top.view(-1, 72, self.out_channels//72, Z, H),
        ) / (self.out_channels/72)
        head = self.head3d(voxel) # B, 10, D, H, W
        dir_off_logit = head[:, 0:3]
        mag_off_logit = head[:, 3:4]
        dir_tgt_logit = head[:, 4:7]
        mag_tgt_logit = head[:, 7:8]
        mask_logit = head[:, 8:10]
        # 方向を単位化、マグはsoftplus→実スケール
        v_off_dir = dir_off_logit / (dir_off_logit.norm(dim=1, keepdim=True).clamp_min(1e-8))
        v_off_mag = self.softplus(mag_off_logit)
        v_pred_off = v_off_dir * v_off_mag
        
        v_tgt_dir = dir_tgt_logit / (dir_tgt_logit.norm(dim=1, keepdim=True).clamp_min(1e-8))
        v_tgt_mag = self.softplus(mag_tgt_logit)
        v_pred_tgt = v_tgt_dir * v_tgt_mag
        # 結合
        offtgt = torch.cat([v_pred_off, v_pred_tgt], dim= 1)  # B, 6, D, H, W
        return  offtgt,mask_logit

class TriView2CoordGrid(nn.Module):
    """エンドツーエンドネットワーク"""
    def __init__(self, in_channels=128):
        super().__init__()
        self.encoder = TripleViewEncoder()
        self.decoder = LowRankVoxelFusion(in_channels)


    def forward(self, x,labels=None,reduction_mag=None):
        
        front, top, side = self.encoder(x)
        vector_field_y,mask_y = self.decoder(front, side,top)
        if self.training:
            assert labels is not None, "Training requires target labels" 
            assert reduction_mag is not None, "Training requires reduction_mag"
            # ベクトル場の損失を計算
            
            joint_list = labels
            torch.cuda.synchronize(); t0 = time.time()    
            vector_field_t,mask_t= self.data_process_torch(vector_field_y,joint_list,reduction_mag)
            torch.cuda.synchronize(); t1 = time.time()
            #print(f"Label grid build time: {t1-t0:.4f}s")            
            
            
            torch.cuda.synchronize(); t0 = time.time()
            groundpolygon = postprocess3D(vector_field_t,
                            reduction_mag)
            torch.cuda.synchronize(); t1 = time.time()
            predictpolygon = postprocess3D(vector_field_y,
                            reduction_mag)
            torch.cuda.synchronize(); t2 = time.time()
            #print(f"Ground polygon processing time: {t1-t0:.4f}s  Predict polygon processing time: {t2-t1:.4f}s")

            
            loss,loss_dict=\
                self.loss_all(
                    vector_field_y,
                    vector_field_t,
                    mask_y,
                    mask_t,
                    groundpolygon, 
                    predictpolygon, 
                    reduction_mag)

            return loss,loss_dict  
        else:
            pred=postprocess3D(vector_field_y,8)
            #groups_as_list(vector_field_y[1])
            return pred[0],pred[1]#[0],groups_as_list(vector_field_y[1])
    
    
    @torch.no_grad()
    def data_process_torch(
        self,
        x: torch.Tensor,  
        joint_list: torch.Tensor,        # [B, N, F, 4, 3]  四角形x(x,y,z)
        reduction_mag: float,
        out_dtype: torch.dtype | None = None,
    ) -> Tuple [torch.Tensor , torch.Tensor]:        
        """
        Returns:
            pafs: (B, 8, D, H, W)[off_mask, off_z, off_y, off_x, off_mask, tgt_z, tgt_y, tgt_x]  
        """
        device = x.device
        B, _,D, H, W = x.shape
        r = float(reduction_mag)
        pafs   = torch.zeros((B, 8, D, H, W), device=device, dtype=torch.float32) 
        
        jl = joint_list.to(device=device, dtype=torch.float32)  # [B,N,F,4,3]
        
        
        
        B_, N, F, _, _ = jl.shape
        assert B_ == B
        # 事前に 26近傍の厚みテーブルを用意（半ボクセル単位: 0.5, √2/2, √3/2）
        
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
                keep_off = a_mask | b_mask

                off_x = torch.where(use_B, off_xb, off_xa)
                off_y = torch.where(use_B, off_yb, off_ya)
                off_z = torch.where(use_B, off_zb, off_za)

                # 最小点のみ残す
                off_x = off_x * keep_off
                off_y = off_y * keep_off
                off_z = off_z * keep_off

                # tgt（H→A もしくは H→B のベクトル、近い方）
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
                # そのまま1ステップ先を終端とする
                ex = Xc + tgt_x
                ey = Yc + tgt_y
                ez = Zc + tgt_z
                d_end = torch.sqrt((ex - Px)**2 + (ey - Py)**2 + (ez - Pz)**2)
                keep_tgt = _single_argmin_mask(d_end)
                tgt_x = tgt_x * keep_tgt
                tgt_y = tgt_y * keep_tgt
                tgt_z = tgt_z * keep_tgt


                off_mask = keep_off.to(dtype=torch.float32)
                tgt_mask = keep_tgt.to(dtype=torch.float32)
  

                # 書き込み：未使用セルかつ draw の位置だけに限定して書く
                patch = pafs[b, :, min_z:max_z, min_y:max_y, min_x:max_x]  # [8, dz, dy, dx]

                vals = [off_z, off_y, off_x, tgt_z, tgt_y, tgt_x,off_mask,tgt_mask]  # [8, dz, dy, dx]
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
            
        
        if out_dtype is None:
            
            out_dtype = x.dtype
        return pafs[:,:6].to(out_dtype),pafs[:,6:].to(out_dtype)

            
    def loss_all(self,
                    vector_field_y, 
                    vector_field_t,
                    mask_y,
                    mask_t,
                    groundpolygon, 
                    predictpolygon,
                    lambda_offset=10.0,
                    lambda_target=10.0,
                    lambda_chamfer=1.0,
                    lambda_3dIoU=5.0
                    ):
        torch.cuda.synchronize(); t2 = time.time()
        loss_chamfer = chamfer_distance(predictpolygon, groundpolygon)
        _,loss_IoU3D = IoU3D_voxel(predictpolygon, groundpolygon)
        torch.cuda.synchronize(); t3 = time.time()
        #print(f"Chamfer&IoU3D loss calculation time: {t3-t2:.4f}s")
        
        #offset
        loss_offset,L_det_offset,L_vec_offset,L_neg_offset =\
            loss_sparse_vector_field(mask_y[:,[0]], vector_field_y[:, [0,1,2]], 
                                 mask_t[:,[0]], vector_field_t[:, [0,1,2]], 
                             alpha=1.0, beta=0.2, lam_dir=1.0, 
                             lam_mag=1.0, gamma=2.0, eps=1e-6)
        #target
        loss_target,L_det_target,L_vec_target,L_neg_target =\
            loss_sparse_vector_field(mask_y[:,[1]], vector_field_y[:, [3,4,5]], 
                                 mask_t[:,[1]], vector_field_t[:, [3,4,5]], 
                             alpha=1.0, beta=0.2, lam_dir=1.0, 
                             lam_mag=1.0, gamma=2.0, eps=1e-6)
        
        

        loss_total = loss_offset + loss_target  
        #+lambda_chamfer*torch.log(loss_chamfer) + lambda_3dIoU*loss_IoU3D

        
        
        return loss_total,dict(loss_offset=loss_offset,
                               loss_det_offset= L_det_offset,
                               loss_vec_offset=L_vec_offset,
                               loss_neg_offset=L_neg_offset,
                        
                               loss_target=loss_target,       
                               loss_det_target= L_det_target,
                               loss_vec_target=L_vec_target,
                               loss_neg_target=L_neg_target,
                               
                               loss_chamfer=loss_chamfer, 
                               loss_IoU3D=loss_IoU3D,)




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
