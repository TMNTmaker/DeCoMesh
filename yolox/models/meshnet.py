import torch
from torch.nn import Conv2d, Module, ReLU, MaxPool2d, init
import torch.nn.functional as F
import numpy as np
from yolox.utils import postprocess2D
from yolox.models.losses import multi_shape_boundary_plus_area_loss_2d_batched
from typing import Tuple
import time
class MeshNet(Module):
    def __init__(self):
        super().__init__() 
        self.stage_1 = Stage_1()
        self.stage_2 = Stage_x()
        self.stage_3 = Stage_x()
        self.stage_4 = Stage_x()
        self.stage_5 = Stage_x()
        self.stage_6 = Stage_x()
        for m in self.modules():
            if isinstance(m, Conv2d):
                init.constant_(m.bias, 0)
        
    def forward(self, x,labels=None,reduction_mag=None):
        pafs = []
        feature_map = x
        h1= self.stage_1(feature_map)
        pafs.append(h1)
        h1 = self.stage_2(torch.cat([h1, feature_map], dim = 1))
        pafs.append(h1)
        h1 = self.stage_3(torch.cat([h1, feature_map], dim = 1))
        pafs.append(h1)
        h1 = self.stage_4(torch.cat([h1, feature_map], dim = 1))
        pafs.append(h1)
        h1 = self.stage_5(torch.cat([h1, feature_map], dim = 1))
        pafs.append(h1)
        h1 = self.stage_6(torch.cat([h1, feature_map], dim = 1))
        pafs.append(h1)
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
            return loss_total,loss_offset,loss_target,pafs#,loss_chamfer_xy,loss_chamfer_yz,loss_chamfer_zx,pafs
        else:
            return pafs  
    
    
    
    @torch.no_grad()
    def data_procces_torch(
        self,
        x: torch.Tensor,  
        joint_list: torch.Tensor,        # [B, N, F, 4, 3]  四角形(x,y,z) 
        reduction_mag: float,
        thickness: float = 0.5,
        out_dtype: torch.dtype | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            pafs:         [B, 12, H, W]  (view: xy/yz/zx × 4ch)
            ignore_mask:  [B, 3,  H, W]  (0=valid, 1=ignore)
        """
        device = x.device
        B, _, H, W = x.shape
        r = float(reduction_mag)
        pafs   = torch.zeros((B, 12, H, W), device=device, dtype=torch.float32)

        jl = joint_list.to(device=device, dtype=torch.float32)  # [B,N,E,2,3]
        B_, N, F, _, _ = jl.shape
        assert B_ == B

        # 全要素ゼロの面を除外（[B,N,F]）
        valid_mask = jl.abs().sum(dim=(3,4)) > 0
        
        def draw_view(b: int, A3: torch.Tensor, B3: torch.Tensor, ch_base: int, view_idx: int):
            # A,B を 2D 投影
            if ch_base == 0:      # xy
                A = A3[[0,1]] / r
                B = B3[[0,1]] / r

            elif ch_base == 4:    # yz
                A = A3[[1,2]] / r
                B = B3[[1,2]] / r                
            else:                  # zx
                A = A3[[2,0]] / r
                B = B3[[2,0]] / r
            vec  = B - A               # [2]
            norm2 = (vec * vec).sum()  # scalar
            if (norm2 < 1e-8).item():
                return
            vx, vy = vec[0], vec[1]

            min_x = int(torch.clamp(torch.floor(torch.min(A[0], B[0]) - thickness), 0, W).item())
            max_x = int(torch.clamp(torch.ceil (torch.max(A[0], B[0]) + thickness), 0, W).item())
            min_y = int(torch.clamp(torch.floor(torch.min(A[1], B[1]) - thickness), 0, H).item())
            max_y = int(torch.clamp(torch.ceil (torch.max(A[1], B[1]) + thickness), 0, H).item())
            if (max_x <= min_x) or (max_y <= min_y):
                return

            ys = torch.arange(min_y, max_y, device=device, dtype=torch.float32)
            xs = torch.arange(min_x, max_x, device=device, dtype=torch.float32)
            Yc, Xc = torch.meshgrid(ys + 0.5, xs + 0.5, indexing="ij")  # [h,w]

            APx = Xc - A[0]; APy = Yc - A[1]
            t = (APx * vx + APy * vy) / (norm2 + 1e-12)
            t = t.clamp_(0.0, 1.0)
            Hx = A[0] + t * vx
            Hy = A[1] + t * vy
            dist = torch.hypot(Xc - Hx, Yc - Hy)

            off_x = A[0] - Xc
            off_y = A[1] - Yc
            off_norm = torch.hypot(off_x, off_y)
            use_perp = off_norm > 1.4142135623730951  # sqrt(2)
            off_x = torch.where(use_perp, Hx - Xc, off_x)
            off_y = torch.where(use_perp, Hy - Yc, off_y)
            

            BCx =B[0] - (off_x + Xc)
            BCy =B[1] - (off_y + Yc)
            ACx = A[0] - (off_x + Xc)
            ACy = A[1] - (off_y + Yc)
            tgt_x = torch.where(
                torch.hypot(BCx, BCy) < torch.hypot(ACx, ACy),BCx,ACx)
            tgt_y = torch.where(
                torch.hypot(BCx, BCy) < torch.hypot(ACx, ACy),BCy,ACy)
            
            
            max_abs = torch.maximum(tgt_x.abs(), tgt_y.abs())
            tgt_norm = torch.hypot(tgt_x, tgt_y)
            need_clip = max_abs > 1.0

            tgt_x = torch.where(need_clip, tgt_x / tgt_norm, tgt_x)
            tgt_y = torch.where(need_clip, tgt_y / tgt_norm, tgt_y)

            draw = dist <= thickness
            if not draw.any().item():
                return

            patch = pafs[b, ch_base:ch_base+4, min_y:max_y, min_x:max_x]
            
            vals = [off_y, off_x, tgt_y, tgt_x]
            for i, val in enumerate(vals):
                mask = (patch[i] != 0)
                patch[i][~mask] = val[~mask] * draw[~mask] 
            blk=0

        # 1エッジずつ（bbox 部分はベクトル化）
        for b in range(B):
            # flatten: [N,F] -> [M]
            idxs = torch.nonzero(valid_mask[b], as_tuple=False)  # [K,2] (n,e)
            if idxs.numel() == 0:
                continue
            for nf in idxs:
                n, f = int(nf[0].item()), int(nf[1].item())
                for i in range(len(jl[b, n, f])):     
                    A3, B3 = jl[b, n, f, i-1], jl[b, n, f, i]  # [3]
                    draw_view(b, A3, B3, ch_base=0, view_idx=0)  # xy
                    draw_view(b, A3, B3, ch_base=4, view_idx=1)  # yz
                    draw_view(b, A3, B3, ch_base=8, view_idx=2)  # zx
            blk=0

        ignore = []
        for vi in range(3):
            block = pafs[:, vi*4:(vi+1)*4]              # [B,4,H,W]
            summed = block.sum(dim=1).abs()             # [B,H,W]
            ignore.append((summed < 1e-5).to(torch.uint8))
        
        ignore_mask = torch.stack(ignore, dim=1)        # [B,3,H,W]

        if out_dtype is None:
            out_dtype = x.dtype
        return pafs.to(out_dtype), ignore_mask
    
    
    
    
    def loss_all(self,
                     pafs_ys, 
                     pafs_t, 
                     ignore_mask,
                     reduction_mag):

        def mean_square_error(pred, target):
            assert pred.shape == target.shape, 'x and y should in same shape'
            return torch.sum((pred - target) ** 2) / target.nelement()

        
        loss_total = 0
        torch.cuda.synchronize(); t0 = time.time()
        #groundpolygon_xy = postprocess2D(pafs_t[:,0:4,...],reduction_mag)
        #groundpolygon_yz = postprocess2D(pafs_t[:,4:8,...],reduction_mag)
        #groundpolygon_zx = postprocess2D(pafs_t[:,8:12,...],reduction_mag)
        #torch.cuda.synchronize(); t1 = time.time()
        #print(f"Ground polygon processing time: {t1-t0:.4f}s")

    
        
        blk=0
        
        
        # compute loss on each stage
        
        pafs_off_t = torch.stack((
                pafs_t[:, 0],
                pafs_t[:, 1],
                pafs_t[:, 4],
                pafs_t[:, 5],
                pafs_t[:, 8],
                pafs_t[:, 9],
            ), dim=1)
        pafs_tgt_t = torch.stack((
                pafs_t[:, 2],
                pafs_t[:, 3],
                pafs_t[:, 6],
                pafs_t[:, 7],
                pafs_t[:, 10],
                pafs_t[:, 11],
            ), dim=1)
                
        for pafs_y in pafs_ys:
            stage_pafs_t = pafs_t.clone()
            stage_paf_masks = ignore_mask.clone()
            with torch.no_grad():
                mask_expanded = stage_paf_masks.repeat_interleave(4, dim=1)
                stage_pafs_t = stage_pafs_t.to(device=pafs_y.device, dtype=pafs_y.dtype)        
                stage_pafs_t[mask_expanded == 1] = pafs_y.detach()[mask_expanded == 1]

            pafs_off_y = torch.stack((
                pafs_y[:, 0],
                pafs_y[:, 1],
                pafs_y[:, 4],
                pafs_y[:, 5],
                pafs_y[:, 8],
                pafs_y[:, 9],
            ), dim=1)
            
            pafs_tgt_y = torch.stack((
                pafs_y[:, 2],
                pafs_y[:, 3],
                pafs_y[:, 6],
                pafs_y[:, 7],
                pafs_y[:, 10],
                pafs_y[:, 11],
            ), dim=1)
           
           
           
            loss_offset = mean_square_error(pafs_off_y, pafs_off_t)           
            loss_target = mean_square_error(pafs_tgt_y,pafs_tgt_t)
            loss_total += loss_offset + loss_target#+ loss_chamfer_xy + loss_chamfer_yz + loss_chamfer_zx
        torch.cuda.synchronize(); t1 = time.time()
        print(f"mesh loss processing time: {t1-t0:.4f}s")
        return loss_total, loss_offset,loss_target#,loss_chamfer_xy,loss_chamfer_yz,loss_chamfer_zx
    
class Stage_1(Module):
    def __init__(self):
        super(Stage_1, self).__init__()
        self.conv1_CPM_L1 = Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_CPM_L1 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_CPM_L1 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4_CPM_L1 = Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv5_CPM_L1 = Conv2d(in_channels=512, out_channels=12, kernel_size=1, stride=1, padding=0)
        self.relu = ReLU()
        
    def forward(self, x):
        h1 = self.relu(self.conv1_CPM_L1(x)) # branch1
        h1 = self.relu(self.conv2_CPM_L1(h1))
        h1 = self.relu(self.conv3_CPM_L1(h1))
        h1 = self.relu(self.conv4_CPM_L1(h1))
        h1 = self.conv5_CPM_L1(h1)
        return h1
    
class Stage_x(Module):
    def __init__(self):
        super(Stage_x, self).__init__()
        self.conv1_L1 = Conv2d(in_channels = 268, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.conv2_L1 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.conv3_L1 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.conv4_L1 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.conv5_L1 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.conv6_L1 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 1, stride = 1, padding = 0)
        self.conv7_L1 = Conv2d(in_channels = 128, out_channels = 12, kernel_size = 1, stride = 1, padding = 0)
        self.relu = ReLU()
        
    def forward(self, x):
        h1 = self.relu(self.conv1_L1(x)) # branch1
        h1 = self.relu(self.conv2_L1(h1))
        h1 = self.relu(self.conv3_L1(h1))
        h1 = self.relu(self.conv4_L1(h1))
        h1 = self.relu(self.conv5_L1(h1))
        h1 = self.relu(self.conv6_L1(h1))
        h1 = self.conv7_L1(h1)
        return h1
