
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def normalize_paf(paf):
    norm = np.linalg.norm(paf, axis=0, keepdims=True) + 1e-6
    return paf / norm


def paf_to_lines(paf, axis_order, length=5.0, threshold=0.5):
    H, W = paf.shape[1:]
    coords = np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1).astype(np.float32)  # (H, W, 2)
    coords += 0.5  # center of pixel
    vecs = normalize_paf(paf).transpose(1, 2, 0)  # (H, W, 2)

    mag = np.linalg.norm(paf.transpose(1, 2, 0), axis=-1)
    mask = mag > threshold

    starts = coords[mask]
    ends = (starts + vecs[mask] * length)

    def embed(p2d):
        if axis_order == 'xy':
            return np.stack([p2d[:, 0], p2d[:, 1], np.zeros_like(p2d[:, 0])], axis=1)
        elif axis_order == 'yz':
            return np.stack([np.zeros_like(p2d[:, 0]), p2d[:, 0], p2d[:, 1]], axis=1)
        elif axis_order == 'zx':
            return np.stack([p2d[:, 1], np.zeros_like(p2d[:, 0]), p2d[:, 0]], axis=1)
        else:
            raise ValueError("Invalid axis")

    return embed(starts), embed(ends)


def rasterize_lines_3d(starts, ends, grid_shape):
    D, H, W = grid_shape
    grid = np.zeros(grid_shape, dtype=np.int32)
    for s, e in zip(starts, ends):
        num_points = int(np.linalg.norm(e - s) * 2) + 1
        points = np.linspace(s, e, num=num_points)
        idxs = np.clip(points.round().astype(int), [0, 0, 0], [W - 1, H - 1, D - 1])
        for x, y, z in idxs:
            grid[z, y, x] += 1
    return grid


def pafs_to_voxel(paf_xy, paf_yz, paf_zx, grid_shape=(64, 64, 64), threshold=0.5):
    starts_xy, ends_xy = paf_to_lines(paf_xy, 'xy', threshold=threshold)
    starts_yz, ends_yz = paf_to_lines(paf_yz, 'yz', threshold=threshold)
    starts_zx, ends_zx = paf_to_lines(paf_zx, 'zx', threshold=threshold)

    grid_xy = rasterize_lines_3d(starts_xy, ends_xy, grid_shape)
    grid_yz = rasterize_lines_3d(starts_yz, ends_yz, grid_shape)
    grid_zx = rasterize_lines_3d(starts_zx, ends_zx, grid_shape)

    occ_grid = ((grid_xy > 0).astype(int) + (grid_yz > 0).astype(int) + (grid_zx > 0).astype(int)) >= 2
    return occ_grid.astype(np.uint8)


def voxel_to_open3d(voxel):
    coords = np.argwhere(voxel == 1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords[:, [2, 1, 0]] * 1.0)
    return pcd


def chamfer_distance(pcd1, pcd2):
    pcd_tree1 = o3d.geometry.KDTreeFlann(pcd1)
    pcd_tree2 = o3d.geometry.KDTreeFlann(pcd2)

    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)

    def avg_nn_distance(src_points, target_tree):
        dist_sum = 0
        for p in src_points:
            [_, _, dists] = target_tree.search_knn_vector_3d(p, 1)
            dist_sum += dists[0]
        return dist_sum / len(src_points)

    return avg_nn_distance(points1, pcd_tree2) + avg_nn_distance(points2, pcd_tree1)


# ==== テストデータ生成 ====
H, W = 64, 64
np.random.seed(42)
paf_xy = normalize_paf(np.random.randn(2, H, W))
paf_yz = normalize_paf(np.random.randn(2, H, W))
paf_zx = normalize_paf(np.random.randn(2, H, W))

voxel = pafs_to_voxel(paf_xy, paf_yz, paf_zx)

# 可視化と変換
pcd = voxel_to_open3d(voxel)

# 仮のGroundTruth（球体）と比較
xx, yy, zz = np.meshgrid(np.arange(W), np.arange(H), np.arange(H))
sphere = ((xx - W // 2) ** 2 + (yy - H // 2) ** 2 + (zz - H // 2) ** 2) < 10**2
gt_pcd = voxel_to_open3d(sphere.astype(np.uint8))

cd = chamfer_distance(pcd, gt_pcd)
print("Chamfer Distance:", cd)

def plot_point_cloud(pcd, color='blue', name='point cloud'):
    pts = np.asarray(pcd.points)
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode='markers',
        marker=dict(size=2, color=color),
        name=name
    )

fig = go.Figure(data=[
    plot_point_cloud(pcd, 'green', 'Reconstructed'),
    plot_point_cloud(gt_pcd, 'red', 'Ground Truth')
])
fig.update_layout(scene=dict(aspectmode='data'))
fig.show()