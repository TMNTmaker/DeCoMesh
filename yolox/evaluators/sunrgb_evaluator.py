#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import time
from collections import  defaultdict
from loguru import logger
from typing import List, Dict
from tqdm import tqdm

import numpy as np

import torch


from yolox.utils import (
    is_main_process,
    synchronize,
    time_synchronized,
    Box3D,Det,GT
)




class SUNRGBDEvaluator:
    """
    SUNRGB AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by SUNRGB.
    """

    def __init__(
        self,
        dataloader,
        img_size: int,
        confthre: float,
        testdev: bool = False,
        per_class_AP: bool = False,
        per_class_AR: bool = False,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre: confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre: IoU threshold of non-max supression ranging from 0 to 1.
            per_class_AP: Show per class AP during evalution or not. Default to True.
            per_class_AR: Show per class AR during evalution or not. Default to True.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.testdev = testdev
        self.per_class_AP = per_class_AP
        self.per_class_AR = per_class_AR

    def image_to_world(self,vertices_img, K, Rt_inv):
        """
        vertices_img: (N,3) 投影後の画像座標 (u,v,z)
            u,v: pixel, z: depth (同じ座標系で)
        return: (N,3) world座標
        """
        N = vertices_img.shape[0]
        homo = np.concatenate([vertices_img[:,:2], np.ones((N,1))], axis=1)  # (u,v,1)
        z = vertices_img[:,2:3]  # depth
        K_inv = np.linalg.inv(K)
        cam = (K_inv @ homo.T) * z.T  # (3,N)
        cam_h = np.vstack([cam, np.ones((1,N))])
        world = (Rt_inv @ cam_h).T[:,:3]
        return world




    def connected_components(self,vertices: np.ndarray, faces: np.ndarray) -> List[np.ndarray]:
        """
        vertices: (N,3)
        faces: (M,3) 頂点インデックス
        return: list of index arrays (それぞれの成分に属する頂点インデックス)
        """
        # 隣接リストを構築
        adj = defaultdict(set)
        for f in faces:
            i, j, k = f
            adj[i].update([j,k])
            adj[j].update([i,k])
            adj[k].update([i,j])

        visited = np.zeros(len(vertices), dtype=bool)
        components = []

        for v in range(len(vertices)):
            if visited[v]:
                continue
            # BFS/DFSで1成分を収集
            stack = [v]
            comp = []
            while stack:
                cur = stack.pop()
                if visited[cur]:
                    continue
                visited[cur] = True
                comp.append(cur)
                for nei in adj[cur]:
                    if not visited[nei]:
                        stack.append(nei)
            components.append(np.array(comp, dtype=np.int64))
        return components

    def mesh_to_3dbox_multi(self,vertices_world: np.ndarray, faces: np.ndarray) -> List[Box3D]:
        """
        vertices_world: (N,3) world座標に変換済み頂点群（複数オブジェクト混在可）
        faces: (M,3) 各三角形の頂点インデックス
        return: list of Box3D
        """
        comps = self.connected_components(vertices_world, faces)
        boxes = []
        for comp in comps:
            verts_sub = vertices_world[comp]

            # AABB
            min_xyz = verts_sub.min(axis=0)
            max_xyz = verts_sub.max(axis=0)
            center = (min_xyz + max_xyz) / 2
            size = (max_xyz - min_xyz)

            # yaw を PCA (XZ 平面)
            pts_xz = verts_sub[:,[0,2]] - verts_sub[:,[0,2]].mean(0)
            cov = np.cov(pts_xz.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            main_axis = eigvecs[:,np.argmax(eigvals)]
            yaw = np.arctan2(main_axis[1], main_axis[0])  # rad

            boxes.append(Box3D(
                float(center[0]), float(center[1]), float(center[2]),
                float(size[0]), float(size[1]), float(size[2]),
                float(yaw)
            ))
        return boxes
    
    
    def evaluate(
        self, model,  half=False,  return_outputs=False
    ):
        """
        SUNRGBD average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by SUNRGBD API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            mAp25 (float) : SUNRGBD AP of 3DIoU=25
            mAp50 (float) : SUNRGBD AP of 3DIoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        sunrgbdDt = defaultdict(Dict[str, List[Det]])
        sunrgbdGt = defaultdict(Dict[str, List[GT]])
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        for cur_iter, (imgs, targets, info_imgs, ids,
                       categories,resize_coefes,cam_infoes,m3Dboxes) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                # skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            data_list_elem, sunrgbdDt_elem, sunrgbdGt_elem = self.convert_to_sunrgb_format(
                outputs, info_imgs, ids,
                categories,resize_coefes,cam_infoes,m3Dboxes, 
                return_outputs=True)
            data_list.extend(data_list_elem)
            sunrgbdDt.update(sunrgbdDt_elem)
            sunrgbdGt.update(sunrgbdGt_elem)
        statistics = torch.cuda.FloatTensor([inference_time, n_samples])

        eval_results = self.evaluate_prediction(sunrgbdGt, sunrgbdDt, statistics)
        synchronize()

        if return_outputs:
            return eval_results, data_list
        return eval_results

    def convert_to_sunrgb_format(self, outputs, info_imgs, ids,
                                 categories,resize_coefes,cam_infoes,m3Dboxes):
        data_list = []
        sunrgbdDt = defaultdict(Dict[str, List[Det]])
        sunrgbdGt = defaultdict(Dict[str, List[GT]])
        for (output, info_img, img_id,gt_category,resize_coef,cam_info,gt_m3Dboxes) in zip(
            outputs, info_imgs, ids,
            categories,resize_coefes,cam_infoes,m3Dboxes
        ):
            if output is None:
                continue
            output = output.cpu()
            
            virtex_world = self.image_to_world(self,output[:,0]/resize_coef, 
                                cam_info["intrinsics"], 
                                cam_info["extrinsics"])
            pred_3dboxes = self.mesh_to_3dbox(virtex_world)
            

            # preprocessing: resize
            labels = [37]*len(pred_3dboxes) # dummy 37=others
            scores = [1.0]*len(pred_3dboxes)#dummy


            for ind in range(pred_3dboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(labels[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "object": pred_3dboxes[ind],
                    "score": float(scores[ind]),
                }  # for SUNRGBD json format
                data_list.append(pred_data)

            sunrgbdDt.update({
                str(img_id): {Det: [
                    Det(label=int(self.dataloader.dataset.class_ids[int(labels[ind])]),
                        box=pred_3dboxes[ind],
                        score=float(scores[ind])) for ind in range(pred_3dboxes.shape[0])
                ]}})
            
            sunrgbdGt.update({
                str(img_id): {GT: [
                    GT(label=int(self.dataloader.dataset.class_ids[int(gt_category[ind])]),
                       box=gt_m3Dboxes[ind]) for ind in range(gt_m3Dboxes.shape[0])
                ]}})
            
        return data_list, sunrgbdDt,sunrgbdGt

    def evaluate_prediction(self, sunrgbdGt, sunrgbdDt, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["mesh","3DIoU"]

        inference_time = statistics[0].item()
        n_samples = statistics[1].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "inference"],
                    [a_infer_time, (a_infer_time )],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        from yolox.layers import SUNRGBDeval_opt as SUNRGBDeval

        sunrgbdEval = SUNRGBDeval(sunrgbdGt, sunrgbdDt, annType[1])
        sunrgbdEval.accumulate()
        return sunrgbdEval.stats[0], sunrgbdEval.stats[1], info
