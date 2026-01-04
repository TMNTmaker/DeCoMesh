import torch
import torch.nn as nn
import torch.nn.functional as F
from yolox.models.losses import *
import time
from typing import Tuple



def build_2d_sincos_pos_embed(h, w, embed_dim, device=None):
    """
    2D sin-cos 位置埋め込みを作る。
    embed_dim が 4 の倍数でない場合も、必ず (1, h*w, embed_dim) を返すようにする。
    return: (1, h*w, embed_dim)
    """
    if device is None:
        device = torch.device("cpu")
    grid_h = torch.arange(h, dtype=torch.float32, device=device)
    grid_w = torch.arange(w, dtype=torch.float32, device=device)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")  # (H, W)

    # each is (H, W)
    grid_h = grid[0].reshape(-1)  # (H*W,)
    grid_w = grid[1].reshape(-1)

    def _build_1d_sincos(pos_1d: torch.Tensor, dim: int) -> torch.Tensor:
        """
        pos_1d: (N,)
        return: (N, dim)  (dim が奇数なら最後を 0 でパディング)
        """
        if dim <= 0:
            return pos_1d.new_zeros((pos_1d.shape[0], 0))
        half = dim // 2
        if half == 0:
            # dim==1 のときは 0 のみ
            return pos_1d.new_zeros((pos_1d.shape[0], 1))
        omega = torch.arange(half, dtype=torch.float32, device=pos_1d.device) / float(half)
        omega = 1.0 / (10000 ** omega)  # (half,)
        out = torch.einsum("n,d->nd", pos_1d, omega)  # (N, half)
        emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # (N, 2*half)
        if emb.shape[1] < dim:
            emb = torch.cat([emb, emb.new_zeros((emb.shape[0], dim - emb.shape[1]))], dim=1)
        elif emb.shape[1] > dim:
            emb = emb[:, :dim]
        return emb

    # embed_dim を h と w に割り当て（奇数でもOK）
    dim_h = embed_dim // 2
    dim_w = embed_dim - dim_h

    pos_emb_h = _build_1d_sincos(grid_h, dim_h)  # (H*W, dim_h)
    pos_emb_w = _build_1d_sincos(grid_w, dim_w)  # (H*W, dim_w)
    pos_emb = torch.cat([pos_emb_h, pos_emb_w], dim=1)  # (H*W, embed_dim)
    pos_emb = pos_emb.unsqueeze(0)  # (1, H*W, C)
    return pos_emb


# =========================================================
# 2. Transformer Block (簡易版)
# =========================================================
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        # x: (B, N, C)
        # self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        # mlp
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        return x


# =========================================================
# 3. View-Conditioned Decoder (Conv) for PAF-like output
# =========================================================
class ViewDecoder(nn.Module):
    """
    spatial_feat: (B, C, H, W)
    view_latent: (B, C)
    output: (B, out_ch, H, W)  # out_ch = 5 (mask, off_x, off_y tgt_vx, tgt_vy)
    """
    def __init__(self, in_ch=256, out_ch=5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.head = nn.Conv2d(128, out_ch, kernel_size=1)

        # view_latent -> gamma/beta
        self.cond = nn.Linear(in_ch, 256 * 2)

    def forward(self, spatial_feat, view_latent):
        B, C, H, W = spatial_feat.shape
        gamma, beta = self.cond(view_latent).chunk(2, dim=1)  # (B,256), (B,256)
        gamma = gamma.view(B, 256, 1, 1)
        beta = beta.view(B, 256, 1, 1)

        x = self.conv1(spatial_feat)
        x = self.bn1(x)
        # FiLM
        x = x * (1 + gamma) + beta
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)

        out = self.head(x)  # (B, out_ch, H, W)
        return out



# =========================================================
# 4. Tri-View PAF Transformer本体
# =========================================================
class TriViewPAFTransformer(nn.Module):
    """
    出力は各ビューごとに (mask,off_vx, off_vy,tgt_vx, tgt_vy) の5/7chベクトル場
    front: (B, 5/7, H, W)
    side : (B, 5/7, H, W)
    top  : (B, 5/7, H, W)
    """
    def __init__(
        self,
        learn_uv=True,
        feat_dim: int = 126,#256+128*0,
        map_size: int = 80,
        num_layers: int = 3,
        num_heads: int = 3,
    ):
        super().__init__()
        self.learn_uv = learn_uv
        self.feat_dim = feat_dim
        self.map_h = map_size
        self.map_w = map_size

        self.token_emb = nn.Conv2d(self.feat_dim, self.feat_dim, kernel_size=3, padding=1)
        # 3つのviewトークン (front, side, top)
        self.view_tokens = nn.Parameter(torch.randn(3, feat_dim))

        
        # 2D位置埋め込み（今回は学習済パラメタにしてもいい）
        self.register_buffer(
            "pos_embed",
            build_2d_sincos_pos_embed(h=self.map_h, w=self.map_w, embed_dim=self.feat_dim),
            persistent=False,
        )

        # Transformerブロック
        self.blocks = nn.ModuleList(
            [TransformerBlock(feat_dim, num_heads=num_heads) for _ in range(num_layers)]
        )

        # viewごとのデコーダ
        if self.learn_uv:
            out_ch = 7
        else:
            out_ch = 5
        self.dec_front = ViewDecoder(in_ch=feat_dim, out_ch=out_ch)
        self.dec_side = ViewDecoder(in_ch=feat_dim, out_ch=out_ch)
        self.dec_top = ViewDecoder(in_ch=feat_dim, out_ch=out_ch)

        self.softsign = nn.Softsign()
        self.softplus = nn.Softplus()

        
    def forward(self,x,cls_feat):
        """
        feat: (B, 256+128, 80, 80)
        """
         # [B, 6*C, H, W] に畳み込んで NCHW に合わせる
        if cls_feat is not None and cls_feat.dim() == 5:
            # (B, V, C, H, W) に対応
            if cls_feat.shape[0] == x.shape[0]:
                #(B,V*C,H,W)にする
                cls_feat = cls_feat.reshape(cls_feat.shape[0],
                                            -1,
                                            cls_feat.shape[3],
                                            cls_feat.shape[4])
                #cls_feat = cls_feat[:,:,:20].sum(dim=2) #(B,V,H,W)
                # ソベルエッジ検出 (各チャネル独立に groups=V で適用)
                # 期待shape: cls_feat = (B, V, H, W)  ※Vは入力チャネルとして扱う
                c = cls_feat.shape[1]
                sobel_x = torch.tensor(
                    [[-1.0, 0.0, 1.0],
                     [-2.0, 0.0, 2.0],
                     [-1.0, 0.0, 1.0]],
                    device=cls_feat.device,
                    dtype=cls_feat.dtype,
                ).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
                sobel_y = torch.tensor(
                    [[-1.0, -2.0, -1.0],
                     [ 0.0,  0.0,  0.0],
                     [ 1.0,  2.0,  1.0]],
                    device=cls_feat.device,
                    dtype=cls_feat.dtype,
                ).view(1, 1, 3, 3).repeat(c, 1, 1, 1)

                gx = F.conv2d(cls_feat, sobel_x, bias=None, stride=1, padding=1, groups=c)
                gy = F.conv2d(cls_feat, sobel_y, bias=None, stride=1, padding=1, groups=c)
                cls_feat = torch.sqrt(gx * gx + gy * gy + 1e-6)        
        feat = cls_feat#torch.cat([feat,cls_feat], dim= 1)



        B, C, H, W = feat.shape

        # (B, C, H, W) -> (B, HW, C)
        tokens = self.token_emb(feat)
        tokens = tokens.flatten(2).transpose(1, 2)

        tokens = tokens + self.pos_embed  # (1, HW, C) broadcast

        # view tokens を先頭に付ける
        view_tokens = self.view_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, 3, C)
        x_tok = torch.cat([view_tokens, tokens], dim=1)  # (B, 3 + HW, C)

        # Transformer通過
        for blk in self.blocks:
            x_tok = blk(x_tok)

        # 出力を分割
        view_latent = x_tok[:, :3, :]  # (B, 3, C)
        spatial_tokens = x_tok[:, 3:, :]  # (B, HW, C)
        spatial_feat = spatial_tokens.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H, W)

        front = self.dec_front(spatial_feat, view_latent[:, 0, :])
        side = self.dec_side(spatial_feat, view_latent[:, 1, :])
        top = self.dec_top(spatial_feat, view_latent[:, 2, :])

        if self.learn_uv:
            
            
            m_pred_front = front[:, 0:1]
            m_pred_side = side[:, 0:1]
            m_pred_top = top[:, 0:1]
            dir_off_logit_front = front[:, 1:3]
            mag_off_logit_front = front[:, 3:4]
            dir_tgt_logit_front = front[:, 4:6]
            mag_tgt_logit_front = front[:, 6:7]
            dir_off_logit_side = side[:, 1:3]
            mag_off_logit_side = side[:, 3:4]
            dir_tgt_logit_side = side[:, 4:6]
            mag_tgt_logit_side = side[:, 6:7]
            dir_off_logit_top = top[:, 1:3]
            mag_off_logit_top = top[:, 3:4]
            dir_tgt_logit_top = top[:, 4:6]
            mag_tgt_logit_top = top[:, 6:7]
            
            
            v_off_dir_xy = dir_off_logit_front / (dir_off_logit_front.norm(dim=1, keepdim=True).clamp_min(1e-8))
            v_off_mag_xy = self.softplus(mag_off_logit_front)
            v_pred_off_xy = v_off_dir_xy * v_off_mag_xy
            
            v_tgt_dir_xy = dir_tgt_logit_front / (dir_tgt_logit_front.norm(dim=1, keepdim=True).clamp_min(1e-8))
            v_tgt_mag_xy = self.softplus(mag_tgt_logit_front)        
            v_pred_tgt_xy = v_tgt_dir_xy * v_tgt_mag_xy       
            
            v_off_dir_yz = dir_off_logit_side / (dir_off_logit_side.norm(dim=1, keepdim=True).clamp_min(1e-8))
            v_off_mag_yz = self.softplus(mag_off_logit_side)
            v_pred_off_yz = v_off_dir_yz * v_off_mag_yz
            
            v_tgt_dir_yz = dir_tgt_logit_side / (dir_tgt_logit_side.norm(dim=1, keepdim=True).clamp_min(1e-8))
            v_tgt_mag_yz = self.softplus(mag_tgt_logit_side)        
            v_pred_tgt_yz = v_tgt_dir_yz * v_tgt_mag_yz       
            
            v_off_dir_zx = dir_off_logit_top / (dir_off_logit_top.norm(dim=1, keepdim=True).clamp_min(1e-8))
            v_off_mag_zx = self.softplus(mag_off_logit_top)
            v_pred_off_zx = v_off_dir_zx * v_off_mag_zx
            
            v_tgt_dir_zx = dir_tgt_logit_top / (dir_tgt_logit_top.norm(dim=1, keepdim=True).clamp_min(1e-8))
            v_tgt_mag_zx = self.softplus(mag_tgt_logit_top)        
            v_pred_tgt_zx = v_tgt_dir_zx * v_tgt_mag_zx       
            # 結合
            h1 = torch.cat([v_pred_off_xy, v_pred_tgt_xy,
                            v_pred_off_yz, v_pred_tgt_yz,
                            v_pred_off_zx, v_pred_tgt_zx], dim= 1)  # B, 12, H, W

        else:
            m_pred_front = front[:, 0:1]
            m_pred_side = side[:, 0:1]
            m_pred_top = top[:, 0:1]
            front = self.softsign(front[:,1:5])
            side = self.softsign(side[:,1:5])
            top = self.softsign(top[:,1:5])
            v_pred_off_front = front[:, 0:2]
            v_pred_tgt_front = front[:, 2:4]
            v_pred_off_side = side[:, 0:2]
            v_pred_tgt_side = side[:, 2:4]
            v_pred_off_top = top[:, 0:2]
            v_pred_tgt_top = top[:, 2:4]
            
            # 結合
            h1 = torch.cat([v_pred_off_front, v_pred_tgt_front,
                            v_pred_off_side, v_pred_tgt_side,
                            v_pred_off_top, v_pred_tgt_top], dim= 1)
        m = torch.cat([m_pred_front,m_pred_side,m_pred_top],dim=1)
        return h1,spatial_feat,m


class transformer_threeviewNet(nn.Module):
    def __init__(self,learn_uv=True):
        super().__init__()
        self.learn_uv = learn_uv
        self.TriViewPAFTransformer = TriViewPAFTransformer(self.learn_uv)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 重みをHe(Kaiming)初期化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # バイアスがある場合だけ0で初期化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    
    
    
    def forward(self, x,cls_feat,labels=None,reduction_mag=None):
        pafs,features,masks_y = self.TriViewPAFTransformer(x,cls_feat)
        
        if self.training:
            assert labels is not None, "labels must be provided during training"
            assert reduction_mag is not None, "reduction_mag must be provided during training"
            
            joint_list = labels
            torch.cuda.synchronize(); t0 = time.time()
            pafs_t,masks_t= self.data_process_torch(x,joint_list,reduction_mag)
            torch.cuda.synchronize(); t1 = time.time()
            loss_total,loss_offset,loss_target=self.loss_all(pafs,pafs_t,
                                                 masks_y,masks_t)    
            loss_dict = {
                #"loss_total": loss_total,
                "loss_offset": loss_offset,
                "loss_target": loss_target,
            }
            torch.cuda.synchronize(); t2 = time.time()
            #print(f"Data processing time: {t1-t0:.4f}s, Loss calculation time: {t2-t1:.4f}s")
            return loss_total,loss_dict,pafs,features
        else:
            return pafs,features  
    
    
    @torch.no_grad()
    def data_process_torch(
        self,
        x: torch.Tensor,  
        joint_list: torch.Tensor,        # [B, N, F, 4, 3]  四角形(頂点×3D)
        reduction_mag: float,
        thickness: float = 0.5,
        out_dtype: torch.dtype | None = torch.float16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            pafs:         [B, 12, H, W]  (view: zx/yz/zx × 4ch = off_y, off_x, tgt_y, tgt_x)
            mask_t:  [B, 3,  H, W]  (0/1)
        """
        device = x.device
        B, _, H, W = x.shape
        rate = float(reduction_mag)
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
            A2_xy = (A3[:, [0, 1]] / rate).contiguous()
            B2_xy = (B3[:, [0, 1]] / rate).contiguous()
            draw_edges_view(pafs[b, 0:4], A2_xy, B2_xy)

            # yz
            A2_yz = (A3[:, [1, 2]] / rate).contiguous()
            B2_yz = (B3[:, [1, 2]] / rate).contiguous()
            draw_edges_view(pafs[b, 4:8], A2_yz, B2_yz)

            # zx
            A2_zx = (A3[:, [2, 0]] / rate).contiguous()
            B2_zx = (B3[:, [2, 0]] / rate).contiguous()
            draw_edges_view(pafs[b, 8:12], A2_zx, B2_zx)

        # --- ignore mask を作成（ch 合計が 0 の画素を ignore=1）---
        block_xy = pafs[:, 0:4].sum(dim=1).abs()
        block_yz = pafs[:, 4:8].sum(dim=1).abs()
        block_zx = pafs[:, 8:12].sum(dim=1).abs()
        mask_t = torch.stack([(block_xy > 1e-5),
                                (block_yz > 1e-5),
                                (block_zx > 1e-5)], dim=1).to(torch.uint8)

        if out_dtype is None:
            out_dtype = x.dtype
        return pafs.to(out_dtype), mask_t    
    
    
    
    def loss_all(self,
                     paf_y, 
                     paf_t, 
                     mask_y,
                     mask_t,
                     ):
        
        loss_total = 0
        torch.cuda.synchronize(); t0 = time.time()
    
                
        
        loss_offset_xy,*_ =loss_sparse_vector_field(
                            mask_y[:,[0]], paf_y[:, [0,1]] 
                                ,mask_t[:,[0]], paf_t[:, [0,1]])
        loss_offset_yz,*_ =loss_sparse_vector_field(
                            mask_y[:,[1]], paf_y[:, [4,5]] 
                                ,mask_t[:,[1]], paf_t[:, [4,5]])
        loss_offset_zx,*_ =loss_sparse_vector_field(
                            mask_y[:,[2]], paf_y[:, [8,9]] 
                                ,mask_t[:,[2]], paf_t[:, [8,9]])

        loss_target_xy,*_ =loss_sparse_vector_field(
                            mask_y[:,[0]], paf_y[:, [2,3]] 
                                ,mask_t[:,[0]], paf_t[:, [2,3]])
        loss_target_yz,*_ =loss_sparse_vector_field(
                            mask_y[:,[1]], paf_y[:, [6,7]] 
                                ,mask_t[:,[1]], paf_t[:, [6,7]])
        loss_target_zx,*_ =loss_sparse_vector_field(
                            mask_y[:,[2]], paf_y[:, [10,11]] 
                                ,mask_t[:,[2]], paf_t[:, [10,11]])
        
        loss_offset = loss_offset_xy + loss_offset_yz + loss_offset_zx 
        loss_target = loss_target_xy + loss_target_yz + loss_target_zx 
        loss_total += loss_offset + loss_target
        torch.cuda.synchronize(); t1 = time.time()
        #print(f"mesh loss processing time: {t1-t0:.4f}s")
        return loss_total, loss_offset,loss_target



 