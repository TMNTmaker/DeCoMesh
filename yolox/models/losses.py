#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
import torch
from typing import List, Tuple, Optional, Union
import torch.nn.functional as F

def mean_square_error(pred: torch.Tensor, 
                      target: torch.Tensor):
    assert pred.shape == target.shape, 'x and y should in same shape'
    return torch.sum((pred - target) ** 2) / target.nelement()


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

def asymmetric_loss(pred, target, gamma_pos=5.0, gamma_neg=1.0):
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

def masked_mse(pred, target, mask):
    # pred/target: [B,12,H,W], mask: [B,3,H,W] (1 = ignore)
    mask_exp = mask.repeat_interleave(4, dim=1).bool()  # -> [B,12,H,W]
    valid = ~mask_exp
    if valid.sum() == 0:
        return torch.tensor(0., device=pred.device, dtype=pred.dtype)
    diff = (pred - target) ** 2
    return diff[valid].mean()

def masked_mse_loss(pred, target, mask, eps=1e-12, anchor=None):
    # 1) 計算は少なくとも fp32 で
    pred   = pred.float()
    target = target.float()
    #mask を pred/target と同じ形状に拡張 
    if mask.dim() == pred.dim() - 1: 
        mask = mask.unsqueeze(1).expand_as(pred) 
    else: 
        mask = mask.expand_as(pred)
    
    
    m=mask
    # 3) 入力の有限性チェックで汚染を遮断
    m = m & torch.isfinite(pred) & torch.isfinite(target)

    m = m.to(dtype=pred.dtype)
    
    diff2 = (pred - target) ** 2
    num = (diff2 * m).sum()
    den = m.sum().clamp_min(1)
    loss = num / den
    if anchor is not None and not loss.requires_grad:
        loss = loss + 0.0 * anchor.sum()
    return loss

def masked_charbonnier(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None,
    eps: float = 1e-6, delta: float = 1e-3, anchor_param: torch.Tensor | None = None
):
    pred_f, target_f = pred.float(), target.float()
    finite = torch.isfinite(pred_f) & torch.isfinite(target_f)
    if mask is not None:
        #mask を pred/target と同じ形状に拡張 
        if mask.dim() == pred.dim() - 1: 
            mask = mask.unsqueeze(1).expand_as(pred) 
        else: 
            mask = mask.expand_as(pred)
        
        finite = finite & mask.to(torch.bool)
    m = finite.to(dtype=pred_f.dtype)

    diff = pred_f - target_f
    # ρ(x) = sqrt(x^2 + δ^2)  （δ は小さな定数）
    rho  = torch.sqrt(diff * diff + delta * delta)

    num = (rho * m).sum()
    den = m.sum().clamp_min(1)
    loss = num / den
    if anchor_param is not None and not loss.requires_grad:
        loss = loss + 0.0 * anchor_param.sum()
    return loss


TensorLike = Union[torch.Tensor, List, torch.Tensor]

#@torch.no_grad()
def chamfer_distance(
    predict: List[List[TensorLike]],
    ground:  List[List[TensorLike]],
    *,
    squared: bool = True,
    chunk_size: Optional[int] = None,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Symmetric Chamfer Distance between two *batched* point-cloud sets.

    前提:
      - predict[b][0] が  (Ni, 3) の点群 (float, 任意 Ni)
      - ground[b][0]  が  (Mi, 3) の点群 (float, 任意 Mi)
      - バッチ長は同じ (len(predict) == len(ground))
      - faces 等の他要素は無視（0番目の点群のみ使用）

    Args:
        predict: list length=B, each = [points: (Ni,3), faces: any, ...]
        ground : list length=B, each = [points: (Mi,3), faces: any, ...]
        squared: True の場合は L2^2 を最小化（一般的な設定）
        chunk_size: メモリ削減のための点群分割サイズ（None なら分割しない）
        eps: 数値安定用の極小値

    Returns:
        loss: scalar tensor = 平均Chamfer距離（対称, 全バッチ/全点で平均）
    """
    assert len(predict) == len(ground), "predict と ground のバッチ長が異なります"
    if len(predict) == 0:
        # 空バッチ
        device = torch.device("cpu")
        return torch.tensor(0.0, device=device)

    # device は最初の点群から推定
    first_pc: torch.Tensor = predict[0][0]
    device = first_pc.device

    total_sum = torch.tensor(0.0, device=device)
    total_cnt = torch.tensor(0,    device=device, dtype=torch.long)

    def _pairwise_min_dists_sum(
        A: torch.Tensor,  # (Na,3)
        B: torch.Tensor,  # (Nb,3)
    ) -> Tuple[torch.Tensor, int]:
        """
        sum_i min_j ||A_i - B_j||   （または二乗）と、A_i の個数を返す
        chunk_size を使ってメモリ節約可能
        """
        Na = A.shape[0]
        if Na == 0:
            return torch.tensor(0.0, device=A.device), 0

        if B.shape[0] == 0:
            # 相手が空なら、min距離は定義しにくいので 0 とし、カウント0扱いで返す
            # （= この片側寄与は無視。必要なら大罰則を入れるロジックに変更可）
            return torch.tensor(0.0, device=A.device), 0

        if chunk_size is None or Na <= chunk_size:
            # (Na, Nb) の距離行列
            D = torch.cdist(A, B, p=2)  # L2
            if squared:
                D = D * D
            mins = D.min(dim=1).values  # (Na,)
            return mins.sum(), Na

        # チャンク分割
        s = torch.tensor(0.0, device=A.device)
        cnt = 0
        for start in range(0, Na, chunk_size):
            end = min(start + chunk_size, Na)
            Ach = A[start:end]  # (k,3)
            D = torch.cdist(Ach, B, p=2)
            if squared:
                D = D * D
            mins = D.min(dim=1).values  # (k,)
            s += mins.sum()
            cnt += mins.numel()
        return s, cnt

    for b in range(len(predict)):
        pc_pred: torch.Tensor = predict[0][b].to(device).float()
        pc_gt:   torch.Tensor = ground[0][0].to(device).float()

        # 形状チェック
        if pc_pred.ndim != 2 or pc_pred.size(-1) != 3:
            raise ValueError(f"predict[{b}][0] の形状が不正です: {tuple(pc_pred.shape)} (期待: (N,3))")
        if pc_gt.ndim != 2 or pc_gt.size(-1) != 3:
            raise ValueError(f"ground[{b}][0] の形状が不正です: {tuple(pc_gt.shape)} (期待: (M,3))")

        # 片方向: pred -> gt
        s1, c1 = _pairwise_min_dists_sum(pc_pred, pc_gt)
        # 片方向: gt -> pred
        s2, c2 = _pairwise_min_dists_sum(pc_gt, pc_pred)

        # 和と点数で重み付き平均（全点での平均）
        total_sum = total_sum + s1 + s2
        total_cnt = total_cnt + (c1 + c2)

    if total_cnt.item() == 0:
        # すべてが空などの場合
        return torch.tensor(0.0, device=device)

    loss = total_sum / (total_cnt.to(total_sum.dtype) + eps)
    return loss
import torch
from typing import List, Tuple, Sequence

#@torch.no_grad()
def IoU3D_voxel(
    predict: List[List[TensorLike]],
    ground:  List[List[TensorLike]],
    grid_size: int = 128,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # ---------- 単一メッシュ用のボクセル化（バッチ非対応） ----------
    def voxelize_mesh_single(
        vertices: torch.Tensor,  # (N,3) in [0,1]
        faces: torch.Tensor,     # (M,3) long
        grid_size: int,
    ) -> torch.Tensor:
        """
        Returns:
            voxels: (grid, grid, grid) bool occupancy
        """
        device = vertices.device
        voxels = torch.zeros((grid_size, grid_size, grid_size),
                             dtype=torch.bool, device=device)

        if vertices.numel() == 0 or faces.numel() == 0:
            return voxels  # 空メッシュは空の占有

        # [0,1] -> [0, grid_size-1] へ
        verts_grid = (vertices.clamp(0.0, 1.0) * (grid_size - 1)).to(torch.long)  # (N,3)

        # 三角形ごとに粗いバウンディングボックスを True 埋め（近似）
        # ※ 正確な三角形充填ではなく高速化・安定化目的の近似
        for tri in faces:
            pts = verts_grid[tri]  # (3,3)
            min_xyz = pts.min(dim=0).values.clamp(0, grid_size - 1)
            max_xyz = pts.max(dim=0).values.clamp(0, grid_size - 1)

            # 範囲長が0以下の軸は arange が空になるので自然にスキップされる
            xs = torch.arange(min_xyz[0].item(), max_xyz[0].item() + 1, device=device)
            ys = torch.arange(min_xyz[1].item(), max_xyz[1].item() + 1, device=device)
            zs = torch.arange(min_xyz[2].item(), max_xyz[2].item() + 1, device=device)

            if xs.numel() == 0 or ys.numel() == 0 or zs.numel() == 0:
                continue

            xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
            voxels[xx, yy, zz] = True

        return voxels

    # ---------- 入力チェック ----------
    assert len(predict[0]) == len(ground[0]), "バッチ長が一致していません"
    B = len(predict[0])
    if B == 0:
        # 返り値のデバイスは不定なのでCPU返し
        return torch.empty(0), torch.empty(0)

    # デバイスは各要素の頂点デバイスを優先
    device = predict[0][0].device if len(predict[0]) > 0 else torch.device("cpu")

    ious: Sequence[torch.Tensor] = []
    losses: Sequence[torch.Tensor] = []

    for b in range(B):
        # それぞれ (N,3), (M,3)
        verts_pred = predict[0][b].to(device).float()
        faces_pred = predict[1][b].to(device).long()
        verts_gt   = ground[0][b].to(device).float()
        faces_gt   = ground[1][b].to(device).long()

        # --- 空メッシュの扱い（両方空なら IoU=1、片方空なら IoU=0） ---
        is_empty_pred = (verts_pred.numel() == 0) or (faces_pred.numel() == 0)
        is_empty_gt   = (verts_gt.numel() == 0) or (faces_gt.numel() == 0)
        if is_empty_pred and is_empty_gt:
            iou = torch.tensor(1.0, device=device)
            loss = 1.0 - iou
            ious.append(iou)
            losses.append(loss)
            continue
        elif is_empty_pred or is_empty_gt:
            iou = torch.tensor(0.0, device=device)
            loss = 1.0 - iou
            ious.append(iou)
            losses.append(loss)
            continue

        # --- 予測/GT を同じバウンディングボックスに正規化 ---
        #     合成BBoxで[0,1]に押し込める（同一体積に対してボクセル化）
        all_verts = torch.cat([verts_pred, verts_gt], dim=0)  # (N+O,3)
        mn = all_verts.min(dim=0).values
        mx = all_verts.max(dim=0).values
        span = (mx - mn).clamp_min(eps)  # 0除算回避

        verts_pred_n = (verts_pred - mn) / span
        verts_gt_n   = (verts_gt   - mn) / span

        # --- ボクセル化 ---
        vox_pred = voxelize_mesh_single(verts_pred_n, faces_pred, grid_size)
        vox_gt   = voxelize_mesh_single(verts_gt_n,   faces_gt,   grid_size)

        # --- IoU 計算 ---
        inter = torch.logical_and(vox_pred, vox_gt).sum(dtype=torch.float32)
        union = torch.logical_or(vox_pred,  vox_gt).sum(dtype=torch.float32)

        # 両方空は前段で処理済みだが、念のため union==0 なら IoU=1 とする
        iou = inter / (union + eps) if union > 0 else torch.tensor(1.0, device=device)
        loss = 1.0 - iou

        ious.append(iou)
        losses.append(loss)

    return torch.stack(list(ious)).mean(), torch.stack(list(losses)).mean()

from typing import List, Tuple, Union
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

# Kaolin
from kaolin.ops.mesh import check_sign  # NOTE: kaolin ソースビルド済みを想定


def IoU3D(
    predict: List[List[TensorLike]],
    ground:  List[List[TensorLike]],
    *,
    num_samples: int = 5000,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Kaolin の check_sign による 3D IoU（サンプリング近似）。
    - predict[b] = [verts:(Ni,3), faces:(Mi,3), ...]
    - ground[b]  = [verts:(Oi,3), faces:(Pi,3), ...]
    - PyTorch3D で各メッシュ表面から num_samples 点をサンプリング
    - Kaolin の check_sign で inside/outside を判定して IoU を計算
      （※ メッシュは概ね watertight 前提。非多様体や開いたメッシュでは誤判定の可能性）

    Returns:
        iou:  (B,)  各バッチの IoU
        loss: (B,)  1 - IoU  （学習用に返すが、関数自体は非微分的と考えてください）
    """
    assert len(predict) == len(ground), "predict と ground のバッチ長が異なります。"
    B = len(predict)
    if B == 0:
        return torch.tensor([]), torch.tensor([])

    device = predict[0][0].device
    ious, losses = [], []

    for b in range(B):
        v_pred: torch.Tensor = predict[b][0].to(device).float()  # (Ni,3)
        f_pred: torch.Tensor = predict[b][1].to(device).long()   # (Mi,3)
        v_gt:   torch.Tensor = ground[b][0].to(device).float()   # (Oi,3)
        f_gt:   torch.Tensor = ground[b][1].to(device).long()    # (Pi,3)
        if v_pred.numel() == 0 or f_pred.numel() == 0 or v_gt.numel() == 0 or f_gt.numel() == 0:
            ious.append(torch.tensor(0.0, device=device))
            losses.append(torch.tensor(1.0, device=device))
            continue
        # PyTorch3D Meshes へ（サンプリング用）
        mesh_pred = Meshes(verts=[v_pred], faces=[f_pred])
        mesh_gt   = Meshes(verts=[v_gt],   faces=[f_gt])

        # メッシュ表面から点をサンプリング（均等面積サンプリング）
        pts_pred = sample_points_from_meshes(mesh_pred, num_samples=num_samples)  # (1, S, 3)
        pts_gt   = sample_points_from_meshes(mesh_gt,   num_samples=num_samples)  # (1, S, 3)

        # 判定対象の点集合（両者のサンプルを連結）
        all_pts = torch.cat([pts_pred, pts_gt], dim=1).squeeze(0)  # (2S, 3)

        # Kaolin の check_sign は (B,V,3), (B,F,3), (B,P,3) を受ける想定
        v_pred_b = v_pred.unsqueeze(0)  # (1,Ni,3)
        f_pred_b = f_pred.unsqueeze(0)  # (1,Mi,3)
        v_gt_b   = v_gt.unsqueeze(0)    # (1,Oi,3)
        f_gt_b   = f_gt.unsqueeze(0)    # (1,Pi,3)
        pts_b    = all_pts.unsqueeze(0) # (1,2S,3)

        # inside 判定（bool）
        inside_pred: torch.Tensor = check_sign(v_pred_b, f_pred, pts_b)[0]  # (2S,)
        inside_gt:   torch.Tensor = check_sign(v_gt_b,   f_gt,   pts_b)[0]  # (2S,)

        inter = torch.sum(inside_pred & inside_gt).float()
        union = torch.sum(inside_pred | inside_gt).float()

        iou = inter / (union + eps)
        loss = 1.0 - iou
        ious.append(iou)
        losses.append(loss)

    return torch.stack(ious).mean(), torch.stack(losses).mean()



"""
# ---------------- Graph utils ----------------
def split_connected_components(V: int, edges: torch.Tensor):
    e = edges.long()
    adj = [[] for _ in range(V)]
    for u, v in e.tolist():
        adj[u].append(v); adj[v].append(u)
    seen = [False]*V
    comps = []
    for s in range(V):
        if seen[s]: continue
        stack = [s]; seen[s]=True; comp=[s]
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if not seen[w]:
                    seen[w]=True; stack.append(w); comp.append(w)
        comps.append(torch.tensor(comp, dtype=torch.long))
    return comps

def extract_rings_from_edges(verts: torch.Tensor, edges: torch.Tensor):
    V = verts.shape[0]
    adj = [[] for _ in range(V)]
    for u, v in edges.long().tolist():
        adj[u].append(v); adj[v].append(u)
    for i, nbr in enumerate(adj):
        if len(nbr) != 2:
            return []
            #raise ValueError(f"vertex {i} has degree {len(nbr)} (need 2).")
    visited = [False]*V
    rings = []
    for start in range(V):
        if visited[start]: continue
        ring_idx = [start]; visited[start]=True
        prev = -1; cur = start
        while True:
            a, b = adj[cur]
            nxt = a if a != prev else b
            if nxt == start: break
            ring_idx.append(nxt); visited[nxt]=True
            prev, cur = cur, nxt
        rings.append(verts[torch.tensor(ring_idx)])
    return rings

# ---------------- Plane fit & projection ----------------
def fit_plane_basis(pts: torch.Tensor, eps=1e-9):
    c = pts.mean(dim=0)
    X = pts - c
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    n = Vh[-1]; n = n / (n.norm() + eps)
    u = Vh[0];  u = u / (u.norm() + eps)
    v = torch.cross(n, u); v = v / (v.norm() + eps)
    return c, u, v, n

def project_to_plane(pts: torch.Tensor, c, u, v):
    X = pts - c
    x = (X * u).sum(dim=-1)
    y = (X * v).sum(dim=-1)
    return torch.stack([x, y], dim=-1)

def point_in_poly_2d(pt_xy: np.ndarray, poly_xy: np.ndarray) -> bool:
    x, y = pt_xy
    x1, y1 = poly_xy[:,0], poly_xy[:,1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    cond = ((y1 > y) != (y2 > y))
    xinters = (x2 - x1) / (y2 - y1 + 1e-18) * (y - y1) + x1
    hits = (x < xinters) & cond
    return bool(np.count_nonzero(hits) % 2)

def signed_area_2d(poly_xy: torch.Tensor) -> torch.Tensor:
    x, y = poly_xy[:,0], poly_xy[:,1]
    return 0.5 * torch.sum(x*torch.roll(y,-1,0) - y*torch.roll(x,-1,0))

# ---------------- Group rings into donuts per plane ----------------
def group_rings_outer_holes_3d(rings_3d: list, ang_tol_deg=5.0, dist_tol=1e-3):
    plane_data = []
    for r in rings_3d:
        c,u,v,n = fit_plane_basis(r); plane_data.append((r,c,u,v,n))
    ang_tol = np.deg2rad(ang_tol_deg)

    groups = []
    used = [False]*len(plane_data)
    for i,(ri,ci,ui,vi,ni) in enumerate(plane_data):
        if used[i]: continue
        idxs = [i]; used[i]=True
        for j,(rj,cj,uj,vj,nj) in enumerate(plane_data):
            if used[j]: continue
            cos = float(torch.clamp((ni*nj).sum(), -1, 1))
            ang = np.arccos(abs(cos))
            if ang > ang_tol: continue
            d = abs(float(((cj - ci) * ni).sum()))
            if d <= dist_tol:
                idxs.append(j); used[j]=True

        # reference basis: largest area ring on this plane
        def ring_area_like(k):
            r = plane_data[k][0]
            return float(torch.norm(torch.sum(torch.cross(r - r.mean(0), torch.roll(r - r.mean(0), -1, 0), dim=0))))
        ref = max(idxs, key=ring_area_like)
        _, c0,u0,v0,n0 = plane_data[ref]

        items = []
        for k in idxs:
            r3d, _,_,_,_ = plane_data[k]
            r2d = project_to_plane(r3d, c0,u0,v0)
            A = signed_area_2d(r2d).abs()
            items.append((r3d, r2d, float(A)))
        items.sort(key=lambda t: -t[2])  # big first

        outers, holes_of_outer = [], []
        for r3d, r2d, _A in items:
            c2 = r2d.mean(dim=0).cpu().numpy()
            parent = None
            for oi,(o3d,o2d) in enumerate(outers):
                if point_in_poly_2d(c2, o2d.cpu().numpy()):
                    parent = oi; break
            if parent is None:
                outers.append((r3d, r2d))
                holes_of_outer.append([])
            else:
                holes_of_outer[parent].append((r3d, r2d))
        groups.append({"plane":(c0,u0,v0,n0), "outers":outers, "holes_of_outer":holes_of_outer})
    return groups

# ---------------- Sampling & areas ----------------
def polygon_area_3d(ring3d: torch.Tensor):
    a = ring3d; b = torch.roll(ring3d, -1, dims=0)
    return 0.5 * torch.norm(torch.sum(torch.cross(a, b, dim=-1), dim=0))

def sample_edges_along_ring_3d(ring3d: torch.Tensor, m_per_ring=256):
    a = ring3d; b = torch.roll(ring3d, -1, dims=0)
    seg = b - a
    L = torch.sqrt((seg**2).sum(dim=-1))
    w = (L / (L.sum() + 1e-12)).clamp_min(1e-9)
    n = torch.clamp((w * m_per_ring).round().long(), min=1)
    pts=[]
    for i in range(ring3d.shape[0]):
        t = torch.rand(n[i], 1, device=ring3d.device)
        pts.append(a[i:i+1] + t * seg[i:i+1])
    return torch.cat(pts, dim=0)

def sample_groups_3d(groups: list, m_per_ring=256):
    pts=[]; areas=[]
    device = None
    for g in groups:
        if len(g["outers"])==0 and len(g["holes_of_outer"])==0:
            continue
        device = g["outers"][0][0].device if g["outers"] else g["holes_of_outer"][0][0][0].device
        A = torch.tensor(0.0, device=device)
        for (o3d,_o2d) in g["outers"]:
            pts.append(sample_edges_along_ring_3d(o3d, m_per_ring))
            A = A + polygon_area_3d(o3d)
        for holes in g["holes_of_outer"]:
            for (h3d,_h2d) in holes:
                pts.append(sample_edges_along_ring_3d(h3d, m_per_ring))
                A = A - polygon_area_3d(h3d)
        areas.append(A.abs())
    if device is None:
        return torch.empty(0,3), torch.tensor([])
    P = torch.cat(pts, dim=0) if len(pts)>0 else torch.empty(0,3, device=device)
    return P, (torch.stack(areas) if areas else torch.tensor([], device=device))

# ---------------- Chamfer (point->point) ----------------
def chamfer_pp(P: torch.Tensor, T: torch.Tensor, chunk: int|None=None):
    if P.numel()==0 or T.numel()==0:
        return torch.tensor(0.0, device=(P.device if P.numel()>0 else T.device))
    if chunk is None:
        d_PT = torch.cdist(P, T, p=2).min(dim=1).values
        d_TP = torch.cdist(T, P, p=2).min(dim=1).values
        return 0.5 * (d_PT.mean() + d_TP.mean())
    def min_d(A,B):
        outs=[]
        for i in range(0, A.shape[0], chunk):
            outs.append(torch.cdist(A[i:i+chunk], B, p=2).min(dim=1).values)
        return torch.cat(outs, dim=0) if outs else torch.empty(0, device=A.device)
    return 0.5 * (min_d(P,T).mean() + min_d(T,P).mean())

# ---------------- Greedy matching of components ----------------
def greedy_match(cost_matrix: torch.Tensor):
    C = cost_matrix.clone()
    pairs=[]; used_p=set(); used_g=set()
    while True:
        mask = torch.ones_like(C, dtype=torch.bool)
        if used_p: mask[list(used_p)] = False
        if used_g: mask[:, list(used_g)] = False
        if not mask.any(): break
        Cm = C.clone(); Cm[~mask] = float("inf")
        val, idx = torch.min(Cm.view(-1), dim=0)
        if not torch.isfinite(val): break
        i = (idx // C.shape[1]).item(); j = (idx % C.shape[1]).item()
        pairs.append((i,j)); used_p.add(i); used_g.add(j)
    un_p = [i for i in range(C.shape[0]) if i not in used_p]
    un_g = [j for j in range(C.shape[1]) if j not in used_g]
    return pairs, un_p, un_g

# ---------------- Single-sample multi-shape loss ----------------
def multi_shape_boundary_plus_area_loss_3d_single(
    pred_verts: torch.Tensor, pred_edges: torch.Tensor,
    gt_verts: torch.Tensor,   gt_edges: torch.Tensor,
    m_per_ring=256, w_area=0.1, cdist_chunk: int|None=None,
    match_cost_samples=128, unmatched_penalty=0.1,
    ang_tol_deg=5.0, dist_tol=1e-3
):
    # split connected components
    pred_comps = split_connected_components(pred_verts.shape[0], pred_edges)
    gt_comps   = split_connected_components(gt_verts.shape[0],   gt_edges)

    # comp -> groups (rings -> plane -> outer/holes)
    def comp_to_groups(verts, edges, comp_idx):
        sub_idx = comp_idx
        idmap = {int(old): i for i, old in enumerate(sub_idx.tolist())}
        # sub-edges
        keeplist = [(int(u) in idmap and int(v) in idmap) for u,v in edges.long().tolist()]
        sub_edges = []
        for (u,v),keep in zip(edges.long().tolist(), keeplist):
            if keep: sub_edges.append([idmap[int(u)], idmap[int(v)]])
        sub_edges = torch.tensor(sub_edges, dtype=torch.long, device=verts.device)
        sub_verts = verts[sub_idx]
        rings = extract_rings_from_edges(sub_verts, sub_edges)
        groups = group_rings_outer_holes_3d(rings, ang_tol_deg=ang_tol_deg, dist_tol=dist_tol)
        
        return groups

    pred_groups_list = [comp_to_groups(pred_verts, pred_edges, ci) for ci in pred_comps]
    gt_groups_list   = [comp_to_groups(gt_verts,   gt_edges,   ci) for ci in gt_comps]

    # representative points (light) for matching
    def comp_repr_points(groups, k):
        pts=[]
        for g in groups:
            for (o3d,_o2d) in g["outers"]:
                pts.append(sample_edges_along_ring_3d(o3d, m_per_ring=k))
            for hlist in g["holes_of_outer"]:
                for (h3d,_h2d) in hlist:
                    pts.append(sample_edges_along_ring_3d(h3d, m_per_ring=k))
        if len(pts)==0: 
            return torch.empty(0,3, device=(groups[0]["outers"][0][0].device if len(groups) else pred_verts.device))
        return torch.cat(pts, dim=0)

    P_reprs = [comp_repr_points(g, match_cost_samples) for g in pred_groups_list]
    T_reprs = [comp_repr_points(g, match_cost_samples) for g in gt_groups_list]

    # edge cases
    if len(P_reprs)==0 and len(T_reprs)==0:
        return torch.tensor(0.0, device=pred_verts.device)
    def area_total(groups_list):
        if len(groups_list)==0: return torch.tensor(0.0, device=pred_verts.device)
        areas=[]
        for g in groups_list:
            A = torch.tensor(0.0, device=pred_verts.device)
            if g!=[]:
                for (o3d,_o2d) in g["outers"]: A = A + polygon_area_3d(o3d)
                for holes in g["holes_of_outer"]:
                    for (h3d,_h2d) in holes: A = A - polygon_area_3d(h3d)
            areas.append(A.abs())
        return torch.stack(areas).sum()

    if len(P_reprs)==0:
        total = area_total(gt_groups_list)
        return unmatched_penalty * (total / (total.abs()+1e-6))
    if len(T_reprs)==0:
        total = area_total(pred_groups_list)
        return unmatched_penalty * (total / (total.abs()+1e-6))

    # cost matrix for matching (Chamfer on light samples)
    C = torch.empty((len(P_reprs), len(T_reprs)), device=pred_verts.device)
    for i,Pi in enumerate(P_reprs):
        for j,Tj in enumerate(T_reprs):
            if Pi.numel()==0 or Tj.numel()==0:
                C[i,j] = 1e6
            else:
                C[i,j] = chamfer_pp(Pi, Tj, chunk=None)
    pairs, un_p, un_g = greedy_match(C)

    # accumulate matched pairs loss
    total_loss = torch.tensor(0.0, device=pred_verts.device)
    for (ip, jg) in pairs:
        P_full, Ap_vec = sample_groups_3d(pred_groups_list[ip], m_per_ring)
        T_full, At_vec = sample_groups_3d(gt_groups_list[jg],   m_per_ring)
        Lb = chamfer_pp(P_full, T_full, chunk=cdist_chunk)
        Ap = Ap_vec.sum() if Ap_vec.numel()>0 else torch.tensor(0.0, device=pred_verts.device)
        At = At_vec.sum() if At_vec.numel()>0 else torch.tensor(0.0, device=pred_verts.device)
        La = (Ap - At).abs() / (At.abs() + 1e-6)
        total_loss = total_loss + (Lb + w_area * La)

    # unmatched penalties (area only)
    if un_p:
        Ap_un = area_total([pred_groups_list[i] for i in un_p])
        total_loss = total_loss + unmatched_penalty * (Ap_un / (Ap_un.abs()+1e-6))
    if un_g:
        At_un = area_total([gt_groups_list[j] for j in un_g])
        total_loss = total_loss + unmatched_penalty * (At_un / (At_un.abs()+1e-6))

    return total_loss

# ---------------- Batched wrapper ----------------
def multi_shape_boundary_plus_area_loss_3d_batched(
    batch_pred: list,  # [[(Vp,3), (Ep,2)], ...], len=N
    batch_gt:   list,  # [[(Vg,3), (Eg,2)], ...], len=N
    m_per_ring=256, w_area=0.1, cdist_chunk: int|None=None,
    match_cost_samples=128, unmatched_penalty=0.1,
    ang_tol_deg=5.0, dist_tol=1e-3,
    reduction: str = "mean"  # "mean" | "sum" | "none"
):
    assert len(batch_pred) == len(batch_gt), "batch size mismatch"
    per_sample = []
    for i in range(len(batch_pred)):
        pv, pe = batch_pred[i][0], batch_pred[i][1]
        gv, ge = batch_gt[i][0],   batch_gt[i][1]
        loss_i = multi_shape_boundary_plus_area_loss_3d_single(
            pv, pe, gv, ge,
            m_per_ring=m_per_ring, w_area=w_area, cdist_chunk=cdist_chunk,
            match_cost_samples=match_cost_samples, unmatched_penalty=unmatched_penalty,
            ang_tol_deg=ang_tol_deg, dist_tol=dist_tol
        )
        per_sample.append(loss_i)
    per_sample = torch.stack(per_sample) if len(per_sample)>0 else torch.tensor(0.0)
    if reduction == "mean":
        return per_sample.mean(), per_sample
    elif reduction == "sum":
        return per_sample.sum(), per_sample
    elif reduction == "none":
        return per_sample, per_sample
    else:
        raise ValueError("reduction must be 'mean' | 'sum' | 'none'")

def group_rings_outer_holes_2d(rings: list):
    
    #外枠/穴を自動仕分け。
    #規約: outer=CCW(面積>0), hole=CW(面積<0) に正規化しつつ、包含関係で決定。
    #return: List[ [outer, hole1, ...] ]   複数“ドーナツ”に分割
    areas = torch.tensor([signed_area_2d(r) for r in rings])
    order = torch.argsort(-areas.abs())  # 大きい順
    rings = list(rings)
    outer_indices = []
    holes_belong = {i:[] for i in range(len(rings))}
    for idx in order.tolist():
        r = rings[idx]
        a = areas[idx].item()
        c = r.mean(dim=0).cpu().numpy()
        parent = None
        for oi in outer_indices:
            if point_in_poly_2d(c, rings[oi].cpu().numpy()):
                parent = oi; break
        is_outer_by_orientation = (a > 0)
        if parent is None:
            if not is_outer_by_orientation:
                rings[idx] = torch.flip(r, dims=[0]); areas[idx] = -areas[idx]
            outer_indices.append(idx)
        else:
            if is_outer_by_orientation:
                rings[idx] = torch.flip(r, dims=[0]); areas[idx] = -areas[idx]
            holes_belong[parent].append(idx)
    groups = [[rings[oi]] + [rings[hi] for hi in holes_belong[oi]] for oi in outer_indices]
    return groups

def polygon_area_with_holes_2d(group: list) -> torch.Tensor:
    #outer(CLW正規化済)=正, holes=負 として合算の絶対値
    A = torch.tensor(0.0, device=group[0].device)
    for k,r in enumerate(group):
        s = signed_area_2d(r)
        A = A + (s if k==0 else -s.abs()) if s>=0 else A + (s if k==0 else +s.abs())  # どちらでも最終的に outer - holes
    # 上の書き方が気になるなら素直に:
    A = torch.tensor(0.0, device=group[0].device)
    for i,r in enumerate(group):
        s = signed_area_2d(r)
        A = A + (s if i==0 else -s)
    return A.abs()

def sample_edges_along_ring_2d(ring: torch.Tensor, m_per_ring=256):
    a = ring; b = torch.roll(ring, -1, dims=0)
    seg = b - a
    L = torch.sqrt((seg**2).sum(dim=-1))  # (K,)
    w = (L / (L.sum() + 1e-12)).clamp_min(1e-9)
    n = torch.clamp((w * m_per_ring).round().long(), min=1)
    pts=[]
    for i in range(ring.shape[0]):
        t = torch.rand(n[i], 1, device=ring.device)
        pts.append(a[i:i+1] + t * seg[i:i+1])
    return torch.cat(pts, dim=0) 

def sample_groups_2d(groups: list, m_per_ring=256):
    pts=[]
    areas=[]
    device = None
    for g in groups:
        if len(g)==0: continue
        device = g[0].device
        A = torch.tensor(0.0, device=device)
        for i,ring in enumerate(g):
            pts.append(sample_edges_along_ring_2d(ring, m_per_ring))
            s = signed_area_2d(ring)
            A = A + (s if i==0 else -s)
        areas.append(A.abs())
    if device is None:
        return torch.empty(0,2), torch.tensor([])
    P = torch.cat(pts, dim=0) if pts else torch.empty(0,2, device=device)
    return P, (torch.stack(areas) if areas else torch.tensor([], device=device))

def multi_shape_boundary_plus_area_loss_2d_single(
    pred_verts: torch.Tensor, pred_edges: torch.Tensor,
    gt_verts: torch.Tensor,   gt_edges: torch.Tensor,
    m_per_ring=256, w_area=0.1, cdist_chunk: int|None=None,
    match_cost_samples=128, unmatched_penalty=0.1
):
    # 1) 連結成分に分割
    pred_comps = split_connected_components(pred_verts.shape[0], pred_edges)
    gt_comps   = split_connected_components(gt_verts.shape[0],   gt_edges)

    # 2) 各成分 → rings → outer/holes グループ
    def comp_to_groups(verts, edges, comp_idx):
        sub_idx = comp_idx
        idmap = {int(old): i for i, old in enumerate(sub_idx.tolist())}
        # 辺をこの成分に限定
        keep = [(int(u) in idmap and int(v) in idmap) for u,v in edges.long().tolist()]
        sub_edges = []
        for (u,v),k in zip(edges.long().tolist(), keep):
            if k: sub_edges.append([idmap[int(u)], idmap[int(v)]])
        sub_edges = torch.tensor(sub_edges, dtype=torch.long, device=verts.device)
        sub_verts = verts[sub_idx]
        rings = extract_rings_from_edges(sub_verts, sub_edges)
        groups = group_rings_outer_holes_2d(rings)  # [[outer, hole...], ...]
        return groups

    pred_groups_list = [comp_to_groups(pred_verts, pred_edges, ci) for ci in pred_comps]
    gt_groups_list   = [comp_to_groups(gt_verts,   gt_edges,   ci) for ci in gt_comps]

    # 3) マッチング用の軽量サンプル
    def comp_repr_points(groups, k):
        pts=[]
        for g in groups:
            for ring in g:
                pts.append(sample_edges_along_ring_2d(ring, m_per_ring=k))
        if len(pts)==0: 
            dev = (groups[0][0].device if groups and groups[0] else pred_verts.device)
            return torch.empty(0,2, device=dev)
        return torch.cat(pts, dim=0)

    P_reprs = [comp_repr_points(g, match_cost_samples) for g in pred_groups_list]
    T_reprs = [comp_repr_points(g, match_cost_samples) for g in gt_groups_list]

    # 例外処理（どちらかが空）
    def area_total(groups_list):
        if len(groups_list)==0: return torch.tensor(0.0, device=pred_verts.device)
        areas=[]
        for g in groups_list:
            A = torch.tensor(0.0, device=pred_verts.device)
            for i,ring in enumerate(g):
                s = signed_area_2d(ring)
                A = A + (s if i==0 else -s)
            areas.append(A.abs())
        return torch.stack(areas).sum()

    if len(P_reprs)==0 and len(T_reprs)==0:
        return torch.tensor(0.0, device=pred_verts.device)
    if len(P_reprs)==0:
        total = area_total(gt_groups_list)
        return unmatched_penalty * (total / (total.abs()+1e-6))
    if len(T_reprs)==0:
        total = area_total(pred_groups_list)
        return unmatched_penalty * (total / (total.abs()+1e-6))

    # 4) 成分マッチング（Chamfer on light samples）
    C = torch.empty((len(P_reprs), len(T_reprs)), device=pred_verts.device)
    for i,Pi in enumerate(P_reprs):
        for j,Tj in enumerate(T_reprs):
            if Pi.numel()==0 or Tj.numel()==0:
                C[i,j] = 1e6
            else:
                C[i,j] = chamfer_pp(Pi, Tj, chunk=None)
    pairs, un_p, un_g = greedy_match(C)

    # 5) マッチしたペアでフル Chamfer＋面積
    total_loss = torch.tensor(0.0, device=pred_verts.device)
    for (ip, jg) in pairs:
        P_full, Ap_vec = sample_groups_2d(pred_groups_list[ip], m_per_ring)
        T_full, At_vec = sample_groups_2d(gt_groups_list[jg],   m_per_ring)
        Lb = chamfer_pp(P_full, T_full, chunk=cdist_chunk)
        Ap = Ap_vec.sum() if Ap_vec.numel()>0 else torch.tensor(0.0, device=pred_verts.device)
        At = At_vec.sum() if At_vec.numel()>0 else torch.tensor(0.0, device=pred_verts.device)
        La = (Ap - At).abs() / (At.abs() + 1e-6)
        total_loss = total_loss + (Lb + w_area * La)

    # 6) 未マッチ成分のペナルティ（面積のみ）
    if un_p:
        Ap_un = area_total([pred_groups_list[i] for i in un_p])
        total_loss = total_loss + unmatched_penalty * (Ap_un / (Ap_un.abs()+1e-6))
    if un_g:
        At_un = area_total([gt_groups_list[j] for j in un_g])
        total_loss = total_loss + unmatched_penalty * (At_un / (At_un.abs()+1e-6))

    return total_loss

# ---------------- Batched wrapper (2D) ----------------
def multi_shape_boundary_plus_area_loss_2d_batched(
    batch_pred: list,  # [[(Vp,2), (Ep,2)], ...]
    batch_gt:   list,  # [[(Vg,2), (Eg,2)], ...]
    m_per_ring=256, w_area=0.1, cdist_chunk: int|None=None,
    match_cost_samples=128, unmatched_penalty=0.1,
    reduction: str = "mean"  # "mean" | "sum" | "none"
):
    assert len(batch_pred) == len(batch_gt), "batch size mismatch"
    per_sample = []
    for i in range(len(batch_pred)):
        pv, pe = batch_pred[i][0], batch_pred[i][1]
        gv, ge = batch_gt[i][0],   batch_gt[i][1]
        loss_i = multi_shape_boundary_plus_area_loss_2d_single(
            pv, pe, gv, ge,
            m_per_ring=m_per_ring, w_area=w_area, cdist_chunk=cdist_chunk,
            match_cost_samples=match_cost_samples, unmatched_penalty=unmatched_penalty
        )
        per_sample.append(loss_i)
    per_sample = torch.stack(per_sample) if len(per_sample)>0 else torch.tensor(0.0)
    if reduction == "mean":
        return per_sample.mean(), per_sample
    elif reduction == "sum":
        return per_sample.sum(), per_sample
    elif reduction == "none":
        return per_sample, per_sample
    else:
        raise ValueError("reduction must be 'mean' | 'sum' | 'none'")

def loss_polygon_iou(
        pred: torch.Tensor,   # (N, K, 2) 頂点は順序付き（凸/凹OK）
        target: torch.Tensor, # (N, M, 2)
        grid_size: int = 640, # ラスタ解像度（↑で精度↑、計算量も↑）
        tau: float = 0.02,    # ソフト化温度（小さいほど硬い境界）
        reduction: str = "mean",
        eps: float = 1e-7
    ):
        def _segment_distance(p, a, b, eps=1e-12):
            #p: (..., 2) 点
            #a,b: (..., E, 2) 辺の端点
            #返り値: (..., E) 各辺までの最近接距離
            
            ab = b - a                             # (..., E, 2)
            ap = p.unsqueeze(-2) - a               # (..., E, 2)
            ab2 = (ab ** 2).sum(-1).clamp_min(eps) # (..., E)
            t = (ap * ab).sum(-1) / ab2            # (..., E)
            t = t.clamp(0.0, 1.0)
            proj = a + t.unsqueeze(-1) * ab        # (..., E, 2)
            d = ((p.unsqueeze(-2) - proj) ** 2).sum(-1).sqrt()
            return d

        def _winding_number(p, poly, eps=1e-12):
            
            #p: (..., 2)
            #poly: (..., K, 2) 連続順（CW/CCWどちらでも）
            #返り値: (...,) 実数の巻き数（insideなら ≈ ±2π、outsideなら≈0）
            
            v1 = poly
            v2 = torch.roll(poly, shifts=-1, dims=-2)
            a = v1 - p.unsqueeze(-2)   # (..., K, 2)
            b = v2 - p.unsqueeze(-2)   # (..., K, 2)
            # 角度の差の総和
            cross = a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]
            dot   = (a*b).sum(-1)
            ang = torch.atan2(cross, dot.clamp(min=-1e30, max=1e30))  # (..., K)
            return ang.sum(dim=-1)

        def _signed_distance(points, poly):
            
            #points: (G*G, 2)  [0,1] 正規化座標を想定
            #poly:   (K, 2)
            #返り値: (G*G,) SDF（内側を負、外側を正）
            
            K = poly.shape[-2]
            a = poly
            b = torch.roll(poly, shifts=-1, dims=-2)
            # 辺までの距離の最小
            d = _segment_distance(points, a.unsqueeze(0), b.unsqueeze(0)).min(dim=-1).values
            # 符号（巻き数の符号で決定）
            wn = _winding_number(points, poly)
            inside = (wn.abs() > 1e-3)  # True=内側
            sdf = torch.where(inside, -d, d)
            return sdf

        device = pred.device
        N = pred.shape[0]
        pred /= grid_size
        target /= grid_size
        # サンプル格子（[0,1]×[0,1]）
        ys, xs = torch.meshgrid(
            torch.linspace(0, 1, grid_size, device=device),
            torch.linspace(0, 1, grid_size, device=device),
            indexing="ij"
        )
        pts = torch.stack([xs, ys], dim=-1).view(-1, 2)  # (G*G, 2)
        pixel_area = 1.0 / (grid_size * grid_size)

        losses = []
        for i in range(N):
            P = pred[i]   # (K,2)
            T = target[i] # (M,2)

            sdf_p = _signed_distance(pts, P)     # (G*G,)
            sdf_t = _signed_distance(pts, T)     # (G*G,)

            # ソフト占有（sigmoidで連続化）
            occ_p = torch.sigmoid(-sdf_p / tau)
            occ_t = torch.sigmoid(-sdf_t / tau)

            inter = (occ_p * occ_t).sum() * pixel_area
            area_p = occ_p.sum() * pixel_area
            area_t = occ_t.sum() * pixel_area
            union = (area_p + area_t - inter).clamp_min(eps)

            iou = inter / union
            losses.append(1.0 - iou)

        loss = torch.stack(losses)
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss
        else:
            raise ValueError("reduction must be 'none' | 'mean' | 'sum'")




class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
"""