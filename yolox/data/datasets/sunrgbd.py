import copy
import os
import json
import cv2
import numpy as np
from .datasets_wrapper import CacheDataset, cache_read_img
from .sunrgbd_classes import SUNRGBD_CLASSES_38, SUNRGBD_CLASSES_21

class SUNRGBDDataset(CacheDataset):
    def __init__(
        self,
        data_dir,
        json_file,
        name="SUNRGBD",
        img_size=(640, 640),
        preproc=None,
        cache=False,
        cache_type="ram",
    ):
        assert data_dir is not None, "data_dir must be specified"
        self.data_dir = data_dir
        self.json_file = json_file
        with open(os.path.join(self.data_dir, name, self.json_file)) as f: self.sunrgbd = json.load(f)
        self.ids = list(self.sunrgbd.keys())
        self.num_imgs = len(self.ids)
        self.preproc = preproc
        self.name = name
        self.img_size = img_size
        self.annotations = self._load_sun_annotations()
        self.class_ids = list(range(len(SUNRGBD_CLASSES_21)))#list(range(len(SUNRGBD_CLASSES_38)))
        path_filename = [os.path.join(name, anno[3]) for anno in self.annotations]
        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=data_dir,
            cache_dir_name=f"cache_{name}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type
        )
        
    def __len__(self):
        return self.num_imgs

    def _load_sun_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    
    def load_camera_params(self,K, Rt):
        R, t = Rt[:,:3], Rt[:,3]
        Rt_4x4 = np.eye(4)
        Rt_4x4[:3,:3] = R
        Rt_4x4[:3,3] = t
        Rt_inv = np.linalg.inv(Rt_4x4)
        return K, Rt_inv

    
    
    import numpy as np

    # ========= ユーティリティ =========

    def _is_inside(self,val, bound, keep_below, eps=1e-9):
        return (val <= bound + eps) if keep_below else (val >= bound - eps)

    def _line_plane_intersection(self,p1, p2, axis, value, eps=1e-12):
        """線分[p1,p2] と平面 axis=value の交点を返す（無い/平行はNone）。"""
        d1, d2 = p1[axis] - value, p2[axis] - value
        denom = (d2 - d1)
        if abs(denom) < eps:  # 平行 or 同一点
            return None
        t = -d1 / denom
        if t < -eps or t > 1.0 + eps:
            return None
        return p1 + t * (p2 - p1)

    def _dedup_consecutive(self,poly, atol=1e-7):
        """連続重複頂点の除去（順序は保持）。閉路の重複も落とす。"""
        if len(poly) == 0:
            return poly
        out = [poly[0]]
        for p in poly[1:]:
            if not np.allclose(p, out[-1], atol=atol):
                out.append(p)
        if len(out) >= 2 and np.allclose(out[0], out[-1], atol=atol):
            out = out[:-1]
        return np.asarray(out)

    def _order_points_on_plane(self,points, axis, atol=1e-9):
        """
        平面 axis=const 上の点群を、同平面内でCCWに並べ替え。
        axis=0→(y,z)平面, axis=1→(x,z), axis=2→(x,y) を使って角度ソート。
        """
        if len(points) == 0:
            return points
        pts = np.asarray(points)
        # プロジェクション
        if axis == 0:
            P = pts[:, [1, 2]]  # (y,z)
        elif axis == 1:
            P = pts[:, [0, 2]]  # (x,z)
        else:
            P = pts[:, [0, 1]]  # (x,y)

        c = P.mean(0)
        ang = np.arctan2(P[:,1] - c[1], P[:,0] - c[0])
        order = np.argsort(ang)
        return pts[order]

    def _fan_triangulation(self,n):
        """頂点数nの単純多角形を扇形分割。返り値は (K,3) int。n<3なら空。"""
        if n < 3:
            return np.empty((0,3), dtype=int)
        faces = [[0, i, i+1] for i in range(1, n-1)]
        return np.asarray(faces, dtype=int)

    # ========= 平面ごとのクリッピング（交点収集つき） =========

    def _clip_against_plane_collect(self,polygon, axis, value, keep_below, eps=1e-9):
        """
        3D多角形（閉ループ）を平面 axis=value でクリップ。
        ・ポリゴンの新しい頂点列（順序保持）
        ・その過程で生じた この平面上の交点群（順序は後で整える）
        を返す。
        """
        poly = np.asarray(polygon)
        if len(poly) == 0:
            return poly, []

        new_poly = []
        intersections = []
        n = len(poly)

        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            v1, v2 = p1[axis], p2[axis]
            in1 = self._is_inside(v1, value, keep_below, eps)
            in2 = self._is_inside(v2, value, keep_below, eps)

            if in1:
                new_poly.append(p1)

            if in1 != in2:
                ipt = self._line_plane_intersection(p1, p2, axis, value)
                if ipt is not None:
                    new_poly.append(ipt)
                    intersections.append(ipt)

        new_poly = np.asarray(new_poly)
        return new_poly, intersections

    # ========= 立方体クリップ（断面ポリゴンも復元） =========

    def clip_polygon_to_cube_with_caps(self,polygon, cube_min=0.0, cube_max=1.0):
        """
        入力: polygon … (N,3) 任意の3D多角形（閉ループ順序）
        出力:
        main_loop: (M,3) クリップ後の主ループ頂点列（順序保持）
        caps:      List[(K_i,3)] 各境界面で生じた“断面ポリゴン”の頂点列
        """
        poly = np.asarray(polygon).copy()

        # 6面それぞれで: 主ループを更新しつつ、その面で生じた交点を収集
        planes = [
            (0, cube_min, False),  # x >= min
            (0, cube_max, True ),  # x <= max
            (1, cube_min, False),  # y >= min
            (1, cube_max, True ),  # y <= max
            (2, cube_min, False),  # z >= min
            (2, cube_max, True ),  # z <= max
        ]

        # 面ごとの交点バケツ: axis->min/maxインデックス 0/1
        cap_points = { (0,0):[], (0,1):[], (1,0):[], (1,1):[], (2,0):[], (2,1):[] }

        for axis, value, keep_below in planes:
            poly, ipts = self._clip_against_plane_collect(poly, axis, value, keep_below)
            if len(poly) == 0:
                break
            key = (axis, 1 if keep_below else 0)  # keep_below=True→上限面
            cap_points[key].extend(ipts)

            # 連続重複除去で数値安定化
            poly = self._dedup_consecutive(poly)

        main_loop = poly

        # 断面ポリゴン復元：各面で交点が3点以上あれば、同一平面内でCCW整列して一つの輪郭にする
        caps = []
        for (axis, hi), pts in cap_points.items():
            if len(pts) >= 3:
                pts = np.asarray(pts)
                # 数値的重複の除去
                pts = np.unique(np.round(pts, 9), axis=0)
                if len(pts) >= 3:
                    ring = self._order_points_on_plane(pts, axis)
                    caps.append(ring)

        return main_loop, caps

    # ========= 三角形分割（主ループ & 各断面） =========

    def triangulate_loop_and_caps(self,main_loop, caps):
        """
        main_loop: (M,3)
        caps: list of (K_i,3)
        返り値:
        V: (T,3) 連結頂点配列（main→cap1→cap2→… の順に並べる）
        F_main: (F1,3) main_loop 用のfaces（Vに対するインデックス）
        F_caps: list of (F_i,3) 各cap用faces（Vに対するインデックス）
        """
        # 連結頂点配列を作りつつ、各ブロックの開始オフセットを控える
        blocks = []
        V_list = []
        # main
        off = 0
        if len(main_loop)>0:
            
            V_list.append(main_loop)
            blocks.append(('main', off, len(main_loop)))
            off += len(main_loop)
        # caps
        for i, c in enumerate(caps):
            
            V_list.append(c)
            blocks.append((f'cap{i}', off, len(c)))
            off += len(c)
        V = np.vstack(V_list) if len(V_list) else np.empty((0,3))

        # faces
        if len(blocks) == 0:
            return V, np.empty((0,3), dtype=int), []
        F_main = self._fan_triangulation(blocks[0][2]) + blocks[0][1] if blocks[0][2] > 0 else np.empty((0,3), dtype=int)
        F_caps = []
        for name, start, cnt in blocks[1:]:
            F = self._fan_triangulation(cnt)
            if len(F):
                F = F + start
            F_caps.append(F.astype(int))

        return V, F_main.astype(int), F_caps

    
    
    def load_anno_from_ids(self, id_):
        im_ann = self.sunrgbd[id_]
        width = im_ann["width"]
        height = im_ann["height"]
        K, Rt_inv = self.load_camera_params(np.array(im_ann["intrinsics"]),
                                            np.array(im_ann["extrinsics"]))
        cam_info ={"intrinsics":K,
                   "extrinsics":Rt_inv}
        
        annotations = im_ann["objects"]
        objs = []
        m3Dboxes = []
        categories = []
        
        for obj in annotations:
            o={"clean_point":[]}
            for poly in obj["polygon"][0]:
                o["clean_point"].append([poly[0],poly[1],poly[2]])
            objs.append(o)
            m3Dboxes.append(obj["3Dbox"])
            if obj["name"] in SUNRGBD_CLASSES_21:#SUNRGBD_CLASSES_38:
                #categories.append(SUNRGBD_CLASSES_38.index(obj["name"]))
                categories.append(SUNRGBD_CLASSES_21.index(obj["name"]))
            else:
                #categories.append(SUNRGBD_CLASSES_38.index("others"))
                categories.append(SUNRGBD_CLASSES_21.index("others"))
        
        r = min(self.img_size[0] / height, self.img_size[1] / width)
        faces = np.array([[0,1,2,3],
                    [1,5,6,2],
                    [1,0,4,5],
                    [0,3,7,4],
                    [2,6,7,3],
                    [4,7,6,5]])


        res=[]
        for obj in objs:
            o3b = np.array(obj["clean_point"])*r

            # クリップ + 断面復元
            # 境界端の処理
            # x, y, z の値域は [0, self.img_size[0]]
            # 各点が範囲外なら、各面でクリッピングし、必要なら新しい点を追加

            #main_loop, caps = self.clip_polygon_to_cube_with_caps(o3b, 0, self.img_size[0])

            # 三角形化（扇形分割）
            #o3b_new, F_main, F_caps = self.triangulate_loop_and_caps(main_loop, caps)
            
            #faces_all = np.vstack([F_main, *F_caps])


            #res.append(np.array(o3b_new)[faces_all])
            cube_min = 0 +10
            cube_max = self.img_size[0]-10
            points = o3b.tolist() 
            clipped_points = [] 
            for pt in points: 
                x, y, z = pt 
                clipped = [x, y, z] 
                for i, v in enumerate([x, y, z]): 
                    if v < cube_min: 
                        clipped[i] = cube_min 
                    elif v > cube_max: 
                        clipped[i] = cube_max 
                clipped_points.append(clipped)
            res.append(np.array(clipped_points)[faces])
            #debug
            #res.append(np.array(o3b)[faces]) 
                    
        res = np.array(res, dtype=object)
        img_info = (height, width)
        resize_coef = r
        
        file_name = im_ann["img_path"]

        return (res, img_info, resize_coef, file_name,
                categories,
                cam_info,
                m3Dboxes)


    def load_anno(self, index):
        return self.annotations[index][0]
    
    
    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]

        img_file = os.path.join(self.data_dir, file_name)

        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"

        return img

    @cache_read_img(use_cache=True)
    def read_img(self, index):
        return self.load_resized_img(index)

    def pull_item(self, index):
        id_ = self.ids[index]
        label, origin_image_size, resize_coef,file_name,categories,cam_info,m3Dboxes = self.annotations[index]
        img = self.read_img(index)

        return img, copy.deepcopy(label), origin_image_size, id_,categories,resize_coef,cam_info,m3Dboxes

    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id,categories,resize_coef,cam_info,m3Dboxes = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        targets = {"mesh": target,
                   "categories": categories,
                   "resize_coef": resize_coef,
                   "cam_info": cam_info,
                   "m3Dboxes": m3Dboxes}
        return img, targets, img_info, img_id,categories,resize_coef,cam_info,m3Dboxes
