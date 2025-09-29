#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.

import numpy as np
from scipy.spatial import cKDTree
import torch
import torchvision

__all__ = [
    "filter_box",
    "postprocess",
    "postprocess2D",
    "postprocess3D",
    "bboxes_iou",
    "matrix_3diou",
    "compute_map",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
    "cxcywh2xyxy",
    "Box3D",
    "Det",
    "GT",
    "evaluate_sunrgbd_3dIoU_mAP"
]


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]




def prepare_valid_vectors(vector_field, min_norm=0.1):
    # ノルムの一括計算
    norms = np.linalg.norm(vector_field, axis=-1)  # shape: (W, H, D)
    valid_mask = norms >= min_norm                # boolean mask

    # 対象となる点のインデックス（flat配列）
    coords = np.argwhere(valid_mask)              # shape: (N, 3)
    vectors = vector_field[valid_mask]            # shape: (N, 3)

    return coords, vectors


import torch
from typing import Tuple

@torch.jit.script
def _angle_lt_deg(u: torch.Tensor, v: torch.Tensor):
    # u, v: (3,) を想定。半精度で来ても安全にするためfloat32へ。
    u = u.to(torch.float32)
    v = v.to(torch.float32)

    # ノルムの二乗でゼロ判定（sqrtを避けて精度/速度↑）
    du2 = torch.dot(u, u)           # 0-dim tensor
    dv2 = torch.dot(v, v)
    if float(du2.item()) < 1e-16 or float(dv2.item()) < 1e-16:
        return 1000.0

    # cosθ 計算（分母はsqrt(du2)*sqrt(dv2)）
    denom = torch.sqrt(du2) * torch.sqrt(dv2) + 1e-8
    cosv = torch.clamp(torch.dot(u, v) / denom, -1.0, 1.0)

    # 角度[deg] を Python float にしてから比較 → bool を返せる
    ang_deg = float(torch.rad2deg(torch.acos(cosv)).item())
    return ang_deg 




@torch.no_grad()
def cluster_vectors3D_torch(
    vector_field: torch.Tensor,
    mag: float,
    min_norm: float = 1e-4,
    merge_radius: float = 1.0,
    max_sample: int = 1000
):
    """
    Args:
        vector_field: (9, D, H, W)  [offset(z,y,x),..., target(z,y,x)]
    Returns:
        vertices: (N, 3)  float (x,y,z) in coords
        faces:    (M, 3)  long  [vertices indicesx 3] triangular faces
        index_map:(D, H, W) long  頂点インデックス（未割当は-1）
    """
    from collections import defaultdict
    assert vector_field.dim() == 4 and vector_field.size(0) ==6,\
        "vector_field must be (6,D,H,W)"
    device = vector_field.device
    _, D, H, W = vector_field.shape

    off = vector_field[:3]   # (3,D,H,W): z,y,x
    tgt = vector_field[3:]# (3,D,H,W): z,y,x

    # 有効ボクセル
    mag_tgt = torch.linalg.norm(tgt, dim=0)  # (D,H,W)
    mask_tgt = mag_tgt > min_norm
    mag_off = torch.linalg.norm(off, dim=0)  # (D,H,W)
    mask_off = mag_off > min_norm
    
    
    index_map_tgt = torch.full((D, H, W), -1, dtype=torch.long, device=device)
    length_map_step  = defaultdict(list)
    vertices_xyz: list[torch.Tensor] = []
    edges_ij: list[torch.Tensor] = [] 

    # start_idx(チェーンID) -> そのチェーンで生成した頂点(旧インデックス)集合
    group2verts = defaultdict(set)
    
    
    # 26近傍（GPUテンソル→.item()でPython intに）
    neigh = torch.tensor(
        [[dz, dy, dx] for dz in (-1, 0, 1)
                      for dy in (-1, 0, 1)
                      for dx in (-1, 0, 1)
                      if not (dz == 0 and dy == 0 and dx == 0)],
        device=device, dtype=torch.long
    )

    current_vertex = 0
    seeds = mask_tgt.nonzero(as_tuple=False)  # (N,3) [z,y,x]
    blk=0
    for zyx in seeds[:max_sample]:
        z0, y0, x0 = int(zyx[0].item()), int(zyx[1].item()), int(zyx[2].item())
        if index_map_tgt[z0, y0, x0] != -1:
            continue
        
        # スタート頂点 中点
        m0 = tgt[:, z0, y0, x0]
        m0_grid = torch.tensor([z0, y0, x0], device=device, dtype=off.dtype)
        
        start_idx = current_vertex

        # チェーン（貪欲ウォーク）
        cz_p, cy_p, cx_p = z0, y0, x0
        cz_m, cy_m, cx_m = z0, y0, x0
        index_map_tgt[cz_p, cy_p, cx_p] = start_idx

        v_start = m0   # 進行方向の基準
        step = v_start / (torch.linalg.norm(v_start) + 1e-12)
        v_here_p = m0_grid.clone()
        v_here_m = m0_grid.clone()-step

        # 安全装置（無限ループ回避）        
        max_steps = math.sqrt(D**2+H**2+W**2)
        steps = 0
        over_times=0
        n_p=0
        n_m=0

        found_end_p = False
        found_end_m = False

        while True:
            steps += 1
            if steps > max_steps:
                break
            # 進行（連続座標）
            v_here_p += step
            v_here_m -= step
            n_p+=1
            n_m+=1

            # 近傍のうち p_next に最も近い有効セルを選ぶ
            min_dist_p = float('inf')
            min_dist_m = float('inf')
            best_k_p = -1
            best_k_m = -1
            
            #プラス方向
            for k in range(neigh.size(0)):
                nz = cz_p + int(neigh[k, 0].item())
                ny = cy_p + int(neigh[k, 1].item())
                nx = cx_p + int(neigh[k, 2].item())
                if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                    if index_map_tgt[nz, ny, nx] == start_idx:
                        continue
                    p_nb = torch.tensor([nz, ny, nx], device=device, dtype=off.dtype)
                    d_p = float(torch.linalg.norm(v_here_p - p_nb).item())  # ← .item() で比較OKに
                    if d_p < min_dist_p:
                        min_dist_p = d_p
                        best_k_p = k
                else:
                    over_times += 1
            #マイナス方向
            for k in range(neigh.size(0)):
                nz = cz_m + int(neigh[k, 0].item())
                ny = cy_m + int(neigh[k, 1].item())
                nx = cx_m + int(neigh[k, 2].item())
                if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                    if index_map_tgt[nz, ny, nx] == start_idx:
                        continue
                    p_nb = torch.tensor([nz, ny, nx], device=device, dtype=off.dtype)
                    d_m = float(torch.linalg.norm(v_here_m - p_nb).item())  # ← .item() で比較OKに
                    if d_m < min_dist_m:
                        min_dist_m = d_m
                        best_k_m = k
                else:
                    over_times += 1
            
            if best_k_p == -1 or best_k_m == -1 or over_times > 26:
                break

            # 最良近傍に一歩進む
            if found_end_p==False:
                cz_p = cz_p + int(neigh[best_k_p, 0].item())
                cy_p = cy_p + int(neigh[best_k_p, 1].item())
                cx_p = cx_p + int(neigh[best_k_p, 2].item())
                index_map_tgt[cz_p, cy_p, cx_p] = start_idx
                n_p+=1

            if found_end_m==False:
                cz_m = cz_m + int(neigh[best_k_m, 0].item())
                cy_m = cy_m + int(neigh[best_k_m, 1].item())
                cx_m = cx_m + int(neigh[best_k_m, 2].item())
                index_map_tgt[cz_m, cy_m, cx_m] = start_idx
                n_m+=1
                
            # 進行方向に近いベクトルが見つかったら“終点/始点”とする
            for k in range(neigh.size(0)):
                nz = cz_p + int(neigh[k, 0].item())
                ny = cy_p + int(neigh[k, 1].item())
                nx = cx_p + int(neigh[k, 2].item())
                if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                    if mask_off[nz, ny, nx]: #and abs(_angle_lt_deg(np0, normal[:,nz, ny, nx])<90.0):
                        found_end_p = True
                        break
            for k in range(neigh.size(0)):
                nz = cz_m + int(neigh[k, 0].item())
                ny = cy_m + int(neigh[k, 1].item())
                nx = cx_m + int(neigh[k, 2].item())
                if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                    if mask_off[nz, ny, nx]: #and abs(_angle_lt_deg(np0, normal[:,nz, ny, nx])<90.0):
                        found_end_m = True
                        break
            if abs(n_m-n_p)>1:
                if n_m>n_p:
                    found_end_p=False
                else:
                    found_end_m=False
            
            for lms in length_map_step[z0, y0, x0]:
                if abs(lms-(n_m+n_p))<2:
                    found_end_p=False
                    found_end_m=False
                    
            if found_end_p and found_end_m and abs(n_m-n_p)<2:
                length_map_step[z0, y0, x0].append(n_m+n_p)
                break
        
        #始点頂点
        o0 = off[:, cz_p, cy_p, cx_p] + 0.5  # (oz,oy,ox)
        p0_grid = torch.tensor([cz_p, cy_p, cx_p], device=device, dtype=off.dtype)  # (x,y,z)
        p0_off  = torch.tensor([o0[2], o0[1], o0[0]], device=device, dtype=off.dtype)
        p0 = (p0_grid + p0_off) * mag
        vertices_xyz.append(p0)
        group2verts[start_idx].add(current_vertex)  # 紐づけ
        v0_old = current_vertex
        current_vertex += 1
        
        # 終点頂点
        o1 = off[:, cz_m, cy_m, cx_m] + 0.5
        p1_grid = torch.tensor([cx_m, cy_m, cz_m], device=device, dtype=off.dtype)
        p1_off  = torch.tensor([o1[2], o1[1], o1[0]], device=device, dtype=off.dtype)
        p1 = (p1_grid + p1_off ) * mag
        vertices_xyz.append(p1)
        group2verts[start_idx].add(current_vertex)
        v1_old = current_vertex
        current_vertex += 1
        

        if torch.linalg.norm(p1 - p0) > 1e-6:
            edges_ij.append(torch.tensor([v0_old, v1_old], device=device, dtype=torch.long))

    # === stack
    vertices = torch.stack(vertices_xyz, dim=0) if len(vertices_xyz) > 0 \
        else torch.empty((0, 3), device=device, dtype=off.dtype)    
    edges = torch.stack(edges_ij, dim=0) if len(edges_ij) > 0 \
        else torch.empty((0, 2), device=device, dtype=torch.long)
    
    # === 双方向化
    def make_bidir_pairs(pairs: torch.Tensor, dedup: bool = True) -> torch.Tensor:
        """
        pairs: (M, 2) の long/ints テンソル。例 [[1,2],[3,5],...]
        戻り値: (M' ,2) 双方向化したペア。例 [[1,2],[2,1],[3,5],[5,3],...]
        """
        assert pairs.dim() == 2 and pairs.size(1) == 2
        rev = pairs[:, [1, 0]]
        both = torch.cat([pairs, rev], dim=0)
        if dedup:
            # 行を辞書式で一意化（GPU可）
            both = torch.unique(both, dim=0)
        return both
    
    edges_each = make_bidir_pairs(edges)
    
    # 近接マージ（簡易）
    keep = torch.ones(vertices.size(0), dtype=torch.bool, device=device)
    keep_map = {} # 代表 -> 吸収された旧頂点一覧
    if vertices.size(0) > 1 and merge_radius > 0:
        for i in range(vertices.size(0)):
            if not keep[i]:
                continue
            if i + 1 >= vertices.size(0):
                break
            di = torch.linalg.norm(vertices[i+1:] - vertices[i], dim=1)
            dup = (di < merge_radius).nonzero(as_tuple=False).squeeze(-1)
            if dup.numel() > 0:
                dup_idxs = dup + (i + 1)
                keep[dup_idxs] = False
                keep_map[i]=dup_idxs        
        # 旧→新インデックス
        old2new = torch.full((vertices.size(0),), -1, dtype=torch.long, device=device)
        new_idx = 0
        if vertices.numel() > 0:
            new_idx = 0
            for i in range(vertices.size(0)):
                if keep[i]:
                    old2new[i] = new_idx
                    new_idx += 1
    # マージ後頂点
    new_vertices = vertices[keep] if vertices.numel() > 0 else vertices

    # === エッジを新インデックスへリマップ & クリーンアップ
    if edges_each.numel() > 0:
        remap = old2new[edges_each]                           # (E,2)
        valid = (remap[:,0] >= 0) & (remap[:,1] >= 0)
        remap = remap[valid]
        remap = remap[remap[:,0] != remap[:,1]]              # 自己ループ除去
        # ソートして無向エッジ重複を潰す
        remap_sorted = torch.sort(remap, dim=1).values
        new_edges = torch.unique(remap_sorted, dim=0)
    else:
        new_edges = torch.empty((0,2), dtype=torch.long, device=device)

    # === group2verts をマージ後インデックスへ変換（index_map_tgt の活用）
    new_group2verts = {}
    for gid, olds in group2verts.items():
        mapped = [old2new[o].item() for o in olds if old2new[o] >= 0]
        if len(mapped) > 0:
            new_group2verts[gid] = torch.tensor(sorted(set(mapped)), device=device, dtype=torch.long)

    # === 隣接リスト（neighbors_of_i = unique(dst[src==i])）
    if new_edges.numel() > 0:
        src = new_edges[:,0]
        dst = new_edges[:,1]
        # 双方向化（無向グラフ扱い）
        src_ud = torch.cat([src, dst], dim=0)
        dst_ud = torch.cat([dst, src], dim=0)
    else:
        src_ud = torch.empty((0,), dtype=torch.long, device=device)
        dst_ud = torch.empty((0,), dtype=torch.long, device=device)

    def neighbors_of(i: int) -> torch.Tensor:
        if src_ud.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=device)
        nbs = dst_ud[src_ud == i]
        return torch.unique(nbs)
    def tensor_intersect1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return unique sorted intersection of 1D long tensors a, b (device返し)."""
        if a.numel() == 0 or b.numel() == 0:
            return torch.empty(0, dtype=a.dtype, device=a.device)
        # set で高速 & 簡潔（要素数が巨大でなければ十分速い）
        s = set(a.tolist())
        s.intersection_update(b.tolist())
        if not s:
            return torch.empty(0, dtype=a.dtype, device=a.device)
        return torch.tensor(sorted(s), dtype=a.dtype, device=a.device)

    # === 三角面検出（3-cycle）
    faces_list = []
    Nv = new_vertices.size(0)

    # チェーングループ単位で三角形候補を探索（偽陽性削減）
    groups = list(new_group2verts.values()) if len(new_group2verts) > 0 else [torch.arange(Nv, device=device)]

    for group in groups:
        if group.numel() < 3:
            continue
        # グループ内のみで探索
        in_group = torch.zeros(Nv, dtype=torch.bool, device=device)
        in_group[group] = True

        # i < j < k の順で重複回避
        for i in group.tolist():
            Ni = neighbors_of(i)
            Ni = Ni[in_group[Ni]]           # グループ内に限定
            Ni = Ni[Ni > i]                 # i より大きい頂点のみ
            if Ni.numel() < 2:
                continue
            # j を走査
            for j in Ni.tolist():
                Nj = neighbors_of(j)
                Nj = Nj[in_group[Nj]]
                # 共通近傍 k（かつ k > j）を面とする
                common = Ni[torch.isin(Ni, Nj)]#tensor_intersect1d(Ni, Nj)
                common = common[common > j]
                for k in common.tolist():
                    # 三角形 (i,j,k) を追加
                    faces_list.append([i, j, k])
    if len(faces_list) > 0:
        faces = torch.tensor(faces_list, dtype=torch.long, device=device)
        # 同一三角形の重複除去（頂点をソートして一意化）
        faces_sorted = torch.sort(faces, dim=1).values
        faces = torch.unique(faces_sorted, dim=0)
    else:
        faces = torch.empty((0,3), dtype=torch.long, device=device)

    # 返却（必要なら new_edges も返せるが、仕様に合わせて省略可）
    return new_vertices, faces, index_map_tgt


def cluster_vectors2D_torch(
    vector_field: torch.Tensor,
    mag: float,
    threshold_deg: float = 10.0,
    min_norm: float = 1e-4,
    merge_radius: float = 1.0,
) :
    """
    Torch版クラスタリング（2D）。簡易な“貪欲ウォーク”で、近傍(3x3)に
    角度が近いベクトルがあれば辿って一本のエッジにまとめる。

    Args:
        vector_field: (4, H, W)  [offset(y,x), target(y,x)]
        mag:          1ボクセルの物理長
        threshold_deg:角度閾値（度）
        min_norm:     無視する最小ベクトル長（targetのノルム）
        merge_radius: 近接頂点のマージ半径（物理長）

    Returns:
        vertices: (N, 2)  float (x,y,z) in world coords
        edges:    (M, 2)  long  [start_idx, end_idx]
        index_map:(H, W) long  頂点インデックス（未割当は-1）
    """
    assert vector_field.dim() == 3 and vector_field.size(0) == 4, \
        "vector_field must be (4,H,W)"
    device = vector_field.device
    _,H, W = vector_field.shape

    # 成分取り出し
    off = vector_field[:2]   # (2,H,W): y,x
    tgt = vector_field[2:]   # (2,H,W): y,x

    # 小さいベクトルを除外（targetのノルム基準）
    mag_tgt = torch.norm(tgt, dim=0)  # (H,W)
    mask = mag_tgt > min_norm

    # -1で初期化（未割当）
    index_map = torch.full(( H, W), -1, dtype=torch.long, device=device)

    vertices_xy: list[torch.Tensor] = []
    edges_ij: list[torch.Tensor] = []

    # 8近傍
    neigh = torch.tensor(
        [[dy, dx]     for dy in (-1, 0, 1)
                      for dx in (-1, 0, 1)
                      if not (dy == 0 and dx == 0)],
        #+ [[-2, 0], [2, 0], [0, -2], [0, 2]],
        device=device, dtype=torch.long
    )

    current_vertex = 0

    # 走査（Trueのボクセルから未割当のものをスタートにする）
    seeds = mask.nonzero(as_tuple=False)  # (N,2) [y,x]
    for yx in seeds:
        y0, x0 = int(yx[0]), int(yx[1])
        if index_map[y0, x0] != -1:
            continue

        # スタート頂点（始点のworld座標）
        # world = (grid + offset) * mag
        o0 = off[:,y0, x0]  # (oy,ox)
        p0_grid = torch.tensor([x0, y0], device=device, dtype=off.dtype)  # (x,y,z)
        p0_off  = torch.tensor([o0[1], o0[0]], device=device, dtype=off.dtype)
        
        p0 = (p0_grid + p0_off) * mag

        vertices_xy.append(p0)
        start_idx = current_vertex
        current_vertex += 1

        # チェーンを貪欲に延ばす
        cy, cx = y0, x0
        index_map[cy, cx] = start_idx
        moved_once = False

        while True:
            v_here = tgt[:, cy, cx]  # (z,y,x)
            found = False

            # 近傍探索（はじめに見つかった似た向きのボクセルへ進む）
            for k in range(neigh.size(0)):
                ny = cy + int(neigh[k, 0])
                nx = cx + int(neigh[k, 1])
                if  (0 <= ny < H) and (0 <= nx < W):
                    if not mask[ny, nx]:
                        continue
                    if index_map[ny, nx] != -1:
                        continue
                    v_nb = tgt[:, ny, nx]
                    if _angle_lt_deg(v_here, v_nb, threshold_deg):
                        index_map[ ny, nx] = start_idx  # 同じチェーン扱い
                        cy, cx = ny, nx
                        moved_once = True
                        found = True
                        break
            if not found:
                break

        # 終点頂点（終端のworld座標 = (grid + offset + target) * mag）
        o1 = off[:, cy, cx]
        t1 = tgt[:, cy, cx]
        p1_grid = torch.tensor([cy,cx], device=device, dtype=off.dtype)
        p1_off  = torch.tensor([ o1[1], o1[0]], device=device, dtype=off.dtype)
        p1_tgt  = torch.tensor([ t1[1], t1[0]], device=device, dtype=off.dtype)
        p1 = (p1_grid + p1_off + p1_tgt) * mag

        vertices_xy.append(p1)
        end_idx = current_vertex
        current_vertex += 1

        # エッジ（スタート→エンド）。動けなかった点は長さゼロの自己エッジでもよいが、
        # ここでは start != end のみ登録
        if moved_once or torch.norm(p1 - p0) > 1e-6:
            edges_ij.append(torch.tensor([start_idx, end_idx], device=device, dtype=torch.long))

    if len(vertices_xy) == 0:
        return (torch.empty((0, 2), device=device),
                torch.empty((0, 2), dtype=torch.long, device=device))

    vertices = torch.stack(vertices_xy, dim=0)  # (N,2)
    edges = torch.stack(edges_ij, dim=0) if len(edges_ij) > 0 else torch.empty((0, 2), dtype=torch.long, device=device)

    # 近接頂点のマージ（超簡易：貪欲クラスタ）
    if vertices.size(0) > 1 and merge_radius > 0:
        keep = torch.ones(vertices.size(0), dtype=torch.bool, device=device)
        for i in range(vertices.size(0)):
            if not keep[i]:
                continue
            di = torch.norm(vertices[i+1:] - vertices[i], dim=1)
            dup = (di < merge_radius).nonzero(as_tuple=False).squeeze(-1)
            if dup.numel() > 0:
                # マージは代表を i とし、エッジ参照を書き換え、重複は捨てる
                dup_idxs = dup + (i + 1)
                # エッジのインデックス置換
                if edges.numel() > 0:
                    for j in dup_idxs:
                        edges[edges == j] = i
                keep[dup_idxs] = False
        # 実際に削る
        old2new = torch.full((vertices.size(0),), -1, dtype=torch.long, device=device)
        new_idx = 0
        for i in range(vertices.size(0)):
            if keep[i]:
                old2new[i] = new_idx
                new_idx += 1
        vertices = vertices[keep]
        if edges.numel() > 0:
            edges = torch.stack([old2new[edges[:, 0]], old2new[edges[:, 1]]], dim=1)
            # 自己エッジや重複を落とす
            mask_valid = edges[:, 0] != edges[:, 1]
            edges = edges[mask_valid]
            if edges.numel() > 0:
                edges = torch.unique(edges, dim=0)
    faces=[]
    return vertices, faces, index_map

import math
import torch.nn.functional as F
def cluster_vectors2D_torch_fast(
    vector_field: torch.Tensor,   # (4,H,W) [off(y,x), tgt(y,x)]
    mag: float,
    threshold_deg: float = 10.0,
    min_norm: float = 0.1,
    merge_radius: float = 0.0,
):
    assert vector_field.dim() == 3 and vector_field.size(0) == 4, "vector_field must be (4,H,W)"
    device = vector_field.device
    dtype  = vector_field.dtype
    _, H, W = vector_field.shape

    off = vector_field[:2]              # (2,H,W) oy, ox
    tgt = vector_field[2:]              # (2,H,W) ty, tx

    # 有効画素マスク & 正規化
    mag_tgt = torch.linalg.norm(tgt, dim=0,dtype=torch.float16)                  # (H,W)
    mask    = mag_tgt > min_norm
    v = torch.zeros_like(tgt)
    v[:, mask] = tgt[:, mask] / mag_tgt[mask]                # 正規化済み target

    # 8近傍方向（dy,dx）
    dirs = [(-1,-1), (-1,0), (-1,1),
            ( 0,-1),         ( 0,1),
            ( 1,-1), ( 1,0), ( 1,1)]
    # 中心を padding してから切り出すと速い
    v_pad   = F.pad(v, (1,1,1,1))                            # (2,H+2,W+2)
    mask_pad= F.pad(mask, (1,1,1,1), value=False)            # (H+2,W+2)

    # 8方向の近傍正規化ベクトルを積んだテンソルを作る: (8,2,H,W)
    nb = []
    nbm= []
    for dy, dx in dirs:
        y0, y1 = 1+dy, 1+dy+H
        x0, x1 = 1+dx, 1+dx+W
        nb.append(v_pad[:, y0:y1, x0:x1])
        nbm.append(mask_pad[y0:y1, x0:x1])
    nb  = torch.stack(nb,  dim=0)                            # (8,2,H,W)
    nbm = torch.stack(nbm, dim=0)                            # (8,H,W)

    # コサイン類似度：中心 v と 8近傍 nb の内積（すでに正規化済み）
    # sim[k,y,x] = dot(v[:,y,x], nb[k,:,y,x])
    sim = (v.unsqueeze(0) * nb).sum(dim=1)                   # (8,H,W)

    # 無効画素/無効近傍は -inf にして選ばれないように
    sim = sim.masked_fill(~mask.unsqueeze(0), float('-inf'))     # 中心が無効なら全部 -inf
    sim = sim.masked_fill(~nbm,               float('-inf'))     # 近傍が無効なら -inf

    # 閾値
    cos_thr = math.cos(math.radians(threshold_deg))
    sim_thr = torch.tensor(cos_thr, device=device, dtype=sim.dtype)

    # 各画素に対して最良近傍方向（argmax）。しきい値未満なら「次なし」
    best_dir = sim.argmax(dim=0)                               # (H,W) in [0..7]
    best_val = sim.gather(0, best_dir.unsqueeze(0)).squeeze(0) # (H,W)
    has_next = best_val >= sim_thr

    # フラット化した id と “次id”（なければ self か -1）を作る
    yy, xx   = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    id_flat  = (yy*W + xx).flatten()                           # (HW,)
    # 8方向のシフト先座標を事前に持つ
    dyy = torch.tensor([d[0] for d in dirs], device=device)
    dxx = torch.tensor([d[1] for d in dirs], device=device)
    ny = (yy + dyy[best_dir])
    nx = (xx + dxx[best_dir])
    in_bounds = (ny>=0)&(ny<H)&(nx>=0)&(nx<W)
    has_next  = has_next & in_bounds & mask                   # 自身が有効

    next_id = torch.full((H,W), -1, dtype=torch.long, device=device)
    next_id[has_next] = (ny[has_next] * W + nx[has_next]).to(torch.long)
    next_id = next_id.flatten()                                # (HW,)

    # 入次数を数える（始点=入次数0のノード）
    valid_next = next_id >= 0
    indeg = torch.zeros(H*W, device=device, dtype=torch.int32)
    if valid_next.any():
        indeg.scatter_add_(0, next_id[valid_next], torch.ones_like(next_id[valid_next], dtype=indeg.dtype))
    starts_mask = (mask.flatten()) & valid_next & (indeg == 0)

    if not starts_mask.any():
        # 何も鎖が作れない
        return (torch.empty((0,2), device=device, dtype=dtype),
                torch.empty((0,2), device=device, dtype=torch.long))

    start_ids = id_flat[starts_mask]                            # (Ns,)

    # ポインタジャンピングで終点へ（log N 回くらい）
    # -1 は自己に置換しておくと収束計算が簡単
    to = next_id.clone()
    self_id = torch.arange(H*W, device=device, dtype=torch.long)
    to = torch.where(to < 0, self_id, to)
    # 32回も回せば十分（~2^32 > 4e9）
    for _ in range(32):
        to = to[to]

    end_ids = to[start_ids]                                     # (Ns,)

    # 画素座標→ワールド座標
    # 始点 p0 = (grid + off) * mag, 終点 p1 = (grid + off + tgt) * mag
    def id_to_xy(ids: torch.Tensor):
        y = (ids // W)
        x = (ids %  W)
        return x, y

    sx, sy = id_to_xy(start_ids)
    ex, ey = id_to_xy(end_ids)

    # 取り出し（gather で一括）
    # off, tgt は (2,H,W)。順序は (ox, oy) に並べ替える
    # p = ( [x,y] + [ox,oy] [+ [tx,ty]] ) * mag
    ox = off[1]; oy = off[0]
    tx = tgt[1]; ty = tgt[0]

    p0x = (sx.to(dtype) + ox[sy, sx]) * mag
    p0y = (sy.to(dtype) + oy[sy, sx]) * mag
    p1x = (ex.to(dtype) + ox[ey, ex] + tx[ey, ex]) * mag
    p1y = (ey.to(dtype) + oy[ey, ex] + ty[ey, ex]) * mag

    # 頂点: [p0s, p1s] を交互に並べると、エッジ作成が O(1)
    Ns = start_ids.numel()
    vertices = torch.empty((Ns*2, 2), device=device, dtype=dtype)
    vertices[0::2, 0] = p0x; vertices[0::2, 1] = p0y
    vertices[1::2, 0] = p1x; vertices[1::2, 1] = p1y

    edges = torch.stack([
        torch.arange(0, Ns*2, 2, device=device, dtype=torch.long),
        torch.arange(1, Ns*2, 2, device=device, dtype=torch.long)
    ], dim=1)

    # 近接頂点マージ（量子化ハッシュで O(N)）
    if merge_radius and merge_radius > 0 and vertices.numel() > 0:
        # 量子化サイズ q ≈ merge_radius
        q = torch.tensor(merge_radius, device=device, dtype=dtype)
        keys = torch.round(vertices / q).to(torch.int64)              # (M,2)
        # ハッシュ（2D → 1D）
        # 衝突低減のために大きめ係数をかける
        key_hash = keys[:, 0] * 73856093 + keys[:, 1] * 19349663
        # Unique & inverse map
        uniq, inv = torch.unique(key_hash, return_inverse=True)
        # 代表点：同一バケツ内の平均にしても良いが、ここは最初の頂点を代表に
        # 新しい頂点はバケツ順で 0..K-1
        new_vertices = torch.zeros((uniq.size(0), 2), device=device, dtype=dtype)
        # 代表を「最初の頂点」にしたいなら scatter_reduce の 'amax' で idx 集約なども可。
        # ここでは単純に平均にしておく（見た目が安定）
        # 平均 = sum / count
        counts = torch.bincount(inv, minlength=uniq.size(0)).to(vertices.dtype).unsqueeze(1)
        sums = torch.zeros_like(new_vertices)
        sums.index_add_(0, inv, vertices)
        new_vertices = sums / counts.clamp_min(1)

        # エッジ端点のインデックスも置換して、自己ループと重複を除去
        new_edges = torch.stack([inv[edges[:,0]], inv[edges[:,1]]], dim=1)
        # 自己ループ除去
        new_edges = new_edges[new_edges[:,0] != new_edges[:,1]]
        if new_edges.numel() > 0:
            new_edges = torch.unique(new_edges, dim=0)

        vertices = new_vertices
        edges    = new_edges

    return vertices, edges


from math import cos, radians
def cluster_vectors3D_torch_fast(
    vector_field: torch.Tensor,
    mag: float,
    threshold_deg: float = 10.0,
    min_norm: float = 0.1,
    merge_radius: float = 0.0,
    max_steps: int | None = None,
):
    """
    高速クラスタリング（3D, 全面ベクトル化）。
    vector_field: (6, D, H, W) = [offset(z,y,x), target(z,y,x)]
    戻り値: vertices (N,3: x,y,z), edges (M,2: long)
    """
    
    assert vector_field.dim() == 4 and vector_field.size(0) == 6, "vector_field must be (6,D,H,W)"
    dev = vector_field.device
    dtype_in = vector_field.dtype
    _, D, H, W = vector_field.shape

    # 計算安定性のため内部は float32（半精度入力でも混在を避ける）
    vf = vector_field.to(torch.float32)
    off = vf[:3]  # (3,D,H,W) z,y,x
    tgt = vf[3:]  # (3,D,H,W) z,y,x

    # 有効ボクセル（targetノルム）
    tgt_norm = torch.linalg.norm(tgt, dim=0)  # (D,H,W)
    src_valid = tgt_norm > float(min_norm)

    # unit ベクトル
    eps = 1e-8
    unit = tgt / (tgt_norm.clamp_min(eps))  # (3,D,H,W)

    # 26近傍をまとめて評価: パディングしてスライス
    # pad: (W,H,D) の順で [left,right, top,bottom, front,back] = 1 ずつ
    pad_unit = F.pad(unit, (1, 1, 1, 1, 1, 1))  # (3, D+2, H+2, W+2)
    pad_valid = F.pad(src_valid[None].float(), (1,1,1,1,1,1)).to(unit.dtype)  # (1,D+2,H+2,W+2)

    # 26 neighbor shifts
    shifts = [(dz,dy,dx) for dz in (-1,0,1) for dy in (-1,0,1) for dx in (-1,0,1)
              if not (dz==0 and dy==0 and dx==0)]

    neigh_units = []
    neigh_valids = []
    for dz,dy,dx in shifts:
        zs = slice(1+dz, 1+dz+D)
        ys = slice(1+dy, 1+dy+H)
        xs = slice(1+dx, 1+dx+W)
        neigh_units.append(pad_unit[:, zs, ys, xs])     # (3,D,H,W)
        neigh_valids.append(pad_valid[:, zs, ys, xs])   # (1,D,H,W)

    # (26,3,D,H,W) にしてコサイン一括計算
    neigh_units = torch.stack(neigh_units, dim=0)   # (26,3,D,H,W)
    neigh_valids = torch.stack(neigh_valids, dim=0) # (26,1,D,H,W)
    # cos = <u, v_nb>
    cos_sim = (neigh_units * unit[None, ...]).sum(dim=1)  # (26,D,H,W)

    # 閾値 & 隣も有効
    cos_thr = float(cos(radians(threshold_deg)))
    valid_pair = (cos_sim >= cos_thr) & (neigh_valids.squeeze(1) > 0.5) & src_valid  # ブロードキャストで (26,D,H,W)

    # 進行先: cos 最大の近傍を選ぶ（無い所は next=-1）
    # 無効な候補を -inf にして argmax
    cos_masked = torch.where(valid_pair, cos_sim, torch.tensor(float("-inf"), device=dev))
    best_idx = cos_masked.argmax(dim=0)                          # (D,H,W) 0..25
    has_next = (cos_masked.max(dim=0).values > -1e30) & src_valid  # (D,H,W)

    # best shift → next のインデックス（線形）を作成
    # まず現在の線形index
    z = torch.arange(D, device=dev)[:, None, None]
    y = torch.arange(H, device=dev)[None, :, None]
    x = torch.arange(W, device=dev)[None, None, :]

    # best shift（dz,dy,dx）をテンソル化して引く
    shift_map = torch.tensor(shifts, device=dev, dtype=torch.int64)  # (26,3)
    dz = shift_map[best_idx, 0]  # (D,H,W)
    dy = shift_map[best_idx, 1]
    dx = shift_map[best_idx, 2]

    nz = (z + dz).clamp(0, D-1)
    ny = (y + dy).clamp(0, H-1)
    nx = (x + dx).clamp(0, W-1)
    next_lin = (nz * (H*W) + ny * W + nx)  # (D,H,W)
    # 端の clamping で“その場”になる場合を無効化（本来は has_next がFalseになっている想定だが二重保険）
    same_pos = (nz==z) & (ny==y) & (nx==x)
    has_next = has_next & (~same_pos)

    # next table: (-1) or next_lin
    next_table = torch.where(has_next, next_lin, torch.full_like(next_lin, -1))

    # 入次数（誰かに指されているか）
    flat_next = next_table.view(-1)
    flat_mask = flat_next >= 0
    in_deg = torch.bincount(flat_next[flat_mask], minlength=D*H*W)  # (DHW,)

    # スタート候補 = src_valid & has_next & (in_deg==0)
    flat_src_valid = src_valid.view(-1)
    flat_has_next = has_next.view(-1)
    starts_mask = (flat_src_valid & flat_has_next & (in_deg == 0))

    start_lin = starts_mask.nonzero(as_tuple=False).squeeze(1)  # (Ns,)
    if start_lin.numel() == 0:
        # すべて閉路などでスタートが無い場合、便宜的に has_next の中から代表サンプルを開始点にする
        start_lin = (flat_src_valid & flat_has_next).nonzero(as_tuple=False).squeeze(1)

    if start_lin.numel() == 0:
        # 何もつながらない
        empty_v = torch.empty((0,3), device=dev, dtype=dtype_in)
        empty_e = torch.empty((0,2), device=dev, dtype=torch.long)
        return empty_v, empty_e

    # ポインタジャンプ（並列トレース）
    curr = start_lin.clone()
    last = curr.clone()
    step = 0
    if max_steps is None:
        max_steps = D + H + W  # 十分大きければOK（局所経路なので多くは早期収束）

    flat_next_table = next_table.view(-1)
    active = torch.ones_like(curr, dtype=torch.bool)

    seen = torch.full((D*H*W,), -1, device=dev, dtype=torch.int64)  # サイクル検出（訪問済み世代）
    gen = 1
    # 訪問印を初期化
    seen[curr] = gen

    while active.any() and step < max_steps:
        nxt = flat_next_table[curr]                      # 次インデックス（-1 あり）
        alive = nxt >= 0
        # 更新
        curr = torch.where(alive, nxt, curr)
        active = alive
        last = torch.where(alive, curr, last)

        # サイクル検出: すでに同世代で訪問済みなら停止
        collided = (seen[curr] == gen) & active
        # 衝突したものは停止扱い（last は直前位置になっている）
        active = active & (~collided)

        # 訪問マーキング
        seen[curr[active]] = gen
        step += 1

    # 始点/終点の (z,y,x)
    sz = (start_lin // (H*W))
    sy = (start_lin % (H*W)) // W
    sx = (start_lin % (W))

    ez = (last // (H*W))
    ey = (last % (H*W)) // W
    ex = (last % (W))

    # 始点ワールド座標 p0 = (grid + off) * mag
    # 注意: off は (z,y,x) 順、出力は (x,y,z)
    p0x = (sx.to(torch.float32) + off[2, sz, sy, sx])
    p0y = (sy.to(torch.float32) + off[1, sz, sy, sx])
    p0z = (sz.to(torch.float32) + off[0, sz, sy, sx])
    p0 = torch.stack([p0x, p0y, p0z], dim=1) * float(mag)  # (Ns,3)

    # 終点ワールド座標 p1 = (grid + off + tgt) * mag
    p1x = (ex.to(torch.float32) + off[2, ez, ey, ex] + tgt[2, ez, ey, ex])
    p1y = (ey.to(torch.float32) + off[1, ez, ey, ex] + tgt[1, ez, ey, ex])
    p1z = (ez.to(torch.float32) + off[0, ez, ey, ex] + tgt[0, ez, ey, ex])
    p1 = torch.stack([p1x, p1y, p1z], dim=1) * float(mag)  # (Ns,3)

    # 0長エッジ除去
    nonzero = (torch.linalg.norm(p1 - p0, dim=1) > 1e-6)
    p0 = p0[nonzero]
    p1 = p1[nonzero]
    Ns = p0.size(0)
    if Ns == 0:
        empty_v = torch.empty((0,3), device=dev, dtype=dtype_in)
        empty_e = torch.empty((0,2), device=dev, dtype=torch.long)
        return empty_v, empty_e

    # --- 頂点配列とエッジ（[start_idx, end_idx]） ---
    # （始点,終点）を連結して 2*Ns 個の頂点を作り、後でマージ
    verts = torch.cat([p0, p1], dim=0)  # (2Ns,3)
    edges = torch.stack([torch.arange(Ns, device=dev), torch.arange(Ns, device=dev)+Ns], dim=1)  # (Ns,2)

    # --- 近接頂点マージ（グリッドハッシュ近似、O(N)） ---
    if merge_radius and merge_radius > 0.0 and verts.size(0) > 1:
        cell = float(merge_radius)
        keys = torch.floor(verts / cell).to(torch.int64)  # (2Ns,3)
        # 3D キーを 1D にハッシュ（原点からの相対なので衝突は稀）
        # キーの範囲が広い場合は 64bit ハッシュ
        primes = torch.tensor([73856093, 19349663, 83492791], device=dev, dtype=torch.int64)
        h = (keys * primes).sum(dim=1)

        # ソートして同一セルを隣接化
        order = torch.argsort(h)
        h_sorted = h[order]

        # 代表頂点のインデックス（セル先頭を代表に）
        rep = torch.empty_like(order)
        rep[0] = order[0]
        for i in range(1, order.numel()):
            if h_sorted[i] == h_sorted[i-1]:
                rep[i] = rep[i-1]
            else:
                rep[i] = order[i]

        # 元の順序に戻すための old->rep マップ
        old2rep = torch.empty_like(rep)
        old2rep[order] = rep

        # 代表座標で頂点を置換（代表点に吸着）
        verts = verts[rep.unique(sorted=True)]
        # old index -> new compact index
        uniq_rep = rep.unique(sorted=True)
        new_id = torch.full((2*Ns,), -1, device=dev, dtype=torch.int64)
        new_id[uniq_rep] = torch.arange(uniq_rep.numel(), device=dev, dtype=torch.int64)

        # エッジの張替え
        e = edges.clone()
        e = torch.stack([new_id[old2rep[e[:,0]]], new_id[old2rep[e[:,1]]]], dim=1)
        # 自己ループ/重複除去
        valid = e[:,0] != e[:,1]
        e = e[valid]
        if e.numel() > 0:
            e = torch.unique(e, dim=0)
        edges = e

    # dtype を入力に合わせる
    verts = verts.to(dtype_in)

    return verts, edges


def postprocess3D(
    prediction: torch.Tensor,  # (B, D, H, W, 6)
    mag: float,
    threshold_deg: float = 10.0,
    min_norm: float = 1e-4,
    merge_radius: float = 10.0,
):    
    """
    バッチ版ポストプロセス。各サンプルに対して (vertices, edges, index_map) を返す。
    """  
    
    assert prediction.dim() == 5 and prediction.size(1) == 6#9
    outs = []
    for b in range(prediction.size(0)):
        v, e, im = cluster_vectors3D_torch(
            prediction[b],
            mag=mag,
            min_norm=min_norm,
            merge_radius=merge_radius,
        )
        outs.append([v, e, im])
    return outs

def postprocess2D(
    prediction: torch.Tensor,  # (B, 12,H, W)
    mag: float,
    threshold_deg: float = 5.0,
    min_norm: float = 1e-4,
    merge_radius: float = 3.0,
):
    """
    バッチ版ポストプロセス。各サンプルに対して (vertices, edges, index_map) を返す。
    """
    assert prediction.dim() == 4 and prediction.size(1) == 4
    outs = []
    for b in range(prediction.size(0)):
        v, e,im = cluster_vectors2D_torch(
            prediction[b],
            mag=mag,
            threshold_deg=threshold_deg,
            min_norm=min_norm,
            merge_radius=merge_radius,
        )
        outs.append((v, e,im))
    return outs




def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output




def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)

def compute_ap(iou_matrix, iou_threshold=0.5, pred_scores=None):
    """
    Compute Average Precision given IoU matrix and IoU threshold.

    Args:
        iou_matrix: (N_gt, N_pred) matrix of IoU
        iou_threshold: float, threshold above which a match is valid
        pred_scores: (N_pred,) array of prediction confidence scores (optional)

    Returns:
        ap: float, average precision
    """
    N_gt, N_pred = iou_matrix.shape

    # スコア順に予測を並び替える（スコアがなければそのまま）
    if pred_scores is not None:
        sorted_idx = np.argsort(-pred_scores)
        iou_matrix = iou_matrix[:, sorted_idx]
    else:
        sorted_idx = np.arange(N_pred)

    tp = np.zeros(N_pred, dtype=np.bool_)
    fp = np.zeros(N_pred, dtype=np.bool_)
    matched_gt = set()

    for j in range(N_pred):
        pred_col = iou_matrix[:, j]
        gt_idx = np.argmax(pred_col)
        iou_val = pred_col[gt_idx]

        if iou_val >= iou_threshold and gt_idx not in matched_gt:
            tp[j] = True
            matched_gt.add(gt_idx)
        else:
            fp[j] = True

    acc_tp = np.cumsum(tp)
    acc_fp = np.cumsum(fp)
    recalls = acc_tp / (N_gt + 1e-8)
    precisions = acc_tp / (acc_tp + acc_fp + 1e-8)

    # AP = PR曲線の下側の面積（数値積分）
    ap = np.trapz(precisions, recalls)
    return ap

def compute_map(iou_matrix, iou_thresholds=[0.25, 0.5, 0.75], pred_scores=None):
    aps = [compute_ap(iou_matrix, thr, pred_scores) for thr in iou_thresholds]
    return np.mean(aps)



def matrix_3diou(gt_meshes, pred_meshes):
    """
    Compute 3D IoU between GT (ground truth) and predicted meshes using AABB.

    Args:
        gt_meshes: (N_gt, P_gt, 3) numpy array - Ground truth meshes
        pred_meshes: (N_pred, P_pred, 3) numpy array - Predicted meshes

    Returns:
        iou: (N_gt, N_pred) numpy array with IoU values between each GT and prediction
    """
    # Get AABB (min/max corners) for GT and predictions
    gt_min = np.min(gt_meshes, axis=1)    # (N_gt, 3)
    gt_max = np.max(gt_meshes, axis=1)    # (N_gt, 3)
    pred_min = np.min(pred_meshes, axis=1)  # (N_pred, 3)
    pred_max = np.max(pred_meshes, axis=1)  # (N_pred, 3)

    # Compute intersection AABB
    inter_min = np.maximum(gt_min[:, np.newaxis, :], pred_min[np.newaxis, :, :])  # (N_gt, N_pred, 3)
    inter_max = np.minimum(gt_max[:, np.newaxis, :], pred_max[np.newaxis, :, :])  # (N_gt, N_pred, 3)
    inter_dims = np.clip(inter_max - inter_min, a_min=0, a_max=None)              # (N_gt, N_pred, 3)
    inter_vol = np.prod(inter_dims, axis=2)                                       # (N_gt, N_pred)

    # Compute volume for each GT and predicted box
    vol_gt = np.prod(gt_max - gt_min, axis=1)           # (N_gt,)
    vol_pred = np.prod(pred_max - pred_min, axis=1)     # (N_pred,)
    union_vol = vol_gt[:, np.newaxis] + vol_pred[np.newaxis, :] - inter_vol  # (N_gt, N_pred)

    # IoU
    iou = inter_vol / (union_vol + 1e-12)
    return iou

def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes


def cxcywh2xyxy(bboxes):
    bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] * 0.5
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    return bboxes



from dataclasses import dataclass
from typing import List, Dict, Tuple, Literal
import math
import numpy as np

@dataclass
class Box3D:
    # center (x,y,z) in meters (camera/worldどちらでもGT/Predで一致していればOK)
    cx: float; cy: float; cz: float
    # size along axes (w: x幅, h: y高さ, l: z奥行) [>0]
    w: float; h: float; l: float
    # yaw (rad), +y を上とし、y軸回りの回転（右手系）
    yaw: float

@dataclass
class Det:
    box: Box3D
    score: float
    label: int           # 0..C-1

@dataclass
class GT:
    box: Box3D
    label: int           # 0..C-1
    difficult: bool=False  # 使わないなら常にFalse









# ---------- 幾何ユーティリティ（回転矩形の多角形交差 × 高さ重なり） ----------
def _rot_rect_xy(cx, cz, w, l, yaw):
    # XZ 平面の回転矩形の4頂点（反時計回り）
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dz = w/2.0, l/2.0
    corners = np.array([[-dx,-dz],[ dx,-dz],[ dx, dz],[-dx, dz]], dtype=np.float64)
    R = np.array([[c,-s],[s, c]], dtype=np.float64)
    rot = (corners @ R.T) + np.array([cx, cz])
    return rot  # (4,2) [x,z]

def _poly_area(poly: np.ndarray) -> float:
    if len(poly) < 3: return 0.0
    x = poly[:,0]; y = poly[:,1]
    return 0.5*abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))

def _clip_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    # Sutherland–Hodgman convex clipper（clipは凸を想定：矩形）
    def inside(p, a, b):
        return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) >= 0
    def intersect(a1, a2, b1, b2):
        # 2線分交点
        x1,y1 = a1; x2,y2 = a2; x3,y3 = b1; x4,y4 = b2
        den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(den) < 1e-12: return a2  # parallel → そのまま
        px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / den
        py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / den
        return np.array([px,py], dtype=np.float64)

    output = subject.copy()
    for i in range(len(clip)):
        input_list = output
        if len(input_list)==0: break
        output = []
        A = clip[i]; B = clip[(i+1)%len(clip)]
        S = input_list[-1]
        for E in input_list:
            if inside(E, A, B):
                if not inside(S, A, B):
                    output.append(intersect(S, E, A, B))
                output.append(E)
            elif inside(S, A, B):
                output.append(intersect(S, E, A, B))
            S = E
        output = np.array(output, dtype=np.float64)
    return output

def iou3d_gravity_aligned(a: Box3D, b: Box3D) -> float:
    # XZで多角形IoU、Y（高さ）は1D重なり
    A = _rot_rect_xy(a.cx, a.cz, a.w, a.l, a.yaw)
    B = _rot_rect_xy(b.cx, b.cz, b.w, b.l, b.yaw)
    inter_poly = _clip_polygon(A, B)
    inter_area = _poly_area(inter_poly)
    area_a = a.w * a.l
    area_b = b.w * b.l
    union_area = area_a + area_b - inter_area
    if union_area <= 0: return 0.0

    # y の重なり（中心と高さから上下端を出す）
    ay1, ay2 = a.cy - a.h/2.0, a.cy + a.h/2.0
    by1, by2 = b.cy - b.h/2.0, b.cy + b.h/2.0
    h_inter = max(0.0, min(ay2, by2) - max(ay1, by1))
    if h_inter <= 0: return 0.0

    inter_vol = inter_area * h_inter
    vol_a = area_a * a.h
    vol_b = area_b * b.h
    union_vol = vol_a + vol_b - inter_vol
    if union_vol <= 0: return 0.0
    return float(inter_vol / union_vol)

# ---------- AP / mAP ----------
def _compute_ap(rec: np.ndarray, prec: np.ndarray, method: Literal["11point","all"]= "11point") -> float:
    if method == "11point":
        ap = 0.0
        for t in np.linspace(0.0,1.0,11):
            p = np.max(prec[rec>=t]) if np.any(rec>=t) else 0.0
            ap += p
        return ap/11.0
    # all-points（単調化して台形積分）
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size-1,0,-1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx+1]-mrec[idx]) * mpre[idx+1]))

def evaluate_sunrgbd_3dIoU_mAP(
    preds: Dict[str, List[Det]],
    gts:   Dict[str, List[GT]],
    num_classes: int,
    iou_thresh_list=(0.25, 0.5),
    ap_method: Literal["11point","all"]="11point",
) -> Dict[float, Dict[str, float]]:
    """
    preds/gts は 画像ID -> [Det] / [GT]
    戻り値: {thr: {"mAP":..., "AP_per_class[i]":...}}
    """
    results = {}
    img_ids = sorted(set(list(preds.keys()) + list(gts.keys())))
    for thr in iou_thresh_list:
        aps = []
        for c in range(num_classes):
            # すべての検出（このクラスのみ）を集約
            dets = []
            npos = 0
            gt_used = {}  # (img, idx) -> matched?
            for img in img_ids:
                gtc = [i for i in gts.get(img, []) if i.label==c and not i.difficult]
                npos += len(gtc)
                for j,_ in enumerate(gtc):
                    gt_used[(img,j)] = False
                for d in preds.get(img, []):
                    if d.label==c:
                        dets.append((img, d))
            # スコア降順でマッチング
            dets.sort(key=lambda x: x[1].score, reverse=True)
            tp = np.zeros(len(dets), dtype=np.float64)
            fp = np.zeros(len(dets), dtype=np.float64)
            for k,(img,d) in enumerate(dets):
                gtc = [i for i in gts.get(img, []) if i.label==c and not i.difficult]
                if len(gtc)==0:
                    fp[k]=1; continue
                ious = np.array([iou3d_gravity_aligned(d.box, g.box) for g in gtc], dtype=np.float64)
                m = int(np.argmax(ious)) if ious.size>0 else -1
                if ious.size>0 and ious[m] >= thr and not gt_used[(img,m)]:
                    tp[k]=1; gt_used[(img,m)]=True
                else:
                    fp[k]=1
            if npos==0:
                aps.append(0.0)
                continue
            fp = np.cumsum(fp); tp = np.cumsum(tp)
            rec = tp / npos
            prec = tp / np.maximum(tp + fp, 1e-12)
            ap = _compute_ap(rec, prec, ap_method)
            aps.append(ap)
        results[thr] = {"mAP": float(np.mean(aps))}
        for i,ap in enumerate(aps):
            results[thr][f"AP_per_class{i}"] = float(ap)
    return results
