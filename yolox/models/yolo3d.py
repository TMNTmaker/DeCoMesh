#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
import torch
import torch.nn as nn
from yolox.utils import freeze_module,unfreeze_module


class YOLOx3D(nn.Module):
    def __init__(self, backbone, classnet, meshnet, coordinate3d):
        super().__init__()
        self.backbone = backbone
        self.classnet = classnet
        self.meshnet = meshnet
        self.coordinate3d = coordinate3d

        self.stage = 1  # 1, 2, 3 で制御

    @torch.no_grad()
    def _monitor_losses(self, fpn0, targets):
        """
        学習に使わずにモニタ用の値だけ計算（勾配なし）。
        """
        loss_class_m, _ = self.classnet(fpn0, targets["category"], targets["mesh"], 8)
        loss_2D_m, loss_offset2D_m, loss_target2D_m, pafs_m,features_m = self.meshnet(fpn0, targets["mesh"], 8)
        loss_3D_m, loss_offset3D_m, loss_target3D_m, loss_chamfer_m, loss_3DIoU_m = \
            self.coordinate3d(features_m[-1], targets["mesh"], 8)
        return dict(
            loss_class=loss_class_m,
            loss_2D=loss_2D_m,
            loss_offset2D=loss_offset2D_m,
            loss_target2D=loss_target2D_m,
            loss_3D=loss_3D_m,
            loss_offset3D=loss_offset3D_m,
            loss_target3D=loss_target3D_m,
            loss_chamfer=loss_chamfer_m,
            loss_3DIoU=loss_3DIoU_m,
        )

    def set_stage(self, stage: int):
        """
        1段階: backbone + classnet だけ学習
        2段階: meshnet だけ学習（TriView2CoordGridへの勾配はMeshNetへ返さない）
        3段階: coordinate3d だけ学習
        """
        assert stage in (1, 2, 3)
        self.stage = stage

        if stage == 1:
            # 学習: backbone, meshnet,classnet
            unfreeze_module(self.backbone, True)
            unfreeze_module(self.classnet, True)
            # 凍結: coordinate3d
            freeze_module(self.meshnet, True)
            freeze_module(self.coordinate3d, True)

        elif stage == 2:
            # 学習: backbone, meshnet,classnet
            unfreeze_module(self.backbone, True)
            unfreeze_module(self.meshnet, True)
            unfreeze_module(self.classnet, True)
            # 凍結: coordinate3d
            freeze_module(self.coordinate3d, True)
            # 凍結: coordinate3d（モニタ用に使っても更新しない）

        elif stage == 3:
            # 凍結: backbone, classnet
            freeze_module(self.backbone, True)
            freeze_module(self.classnet, True)
            # 学習: meshnet,coordinate3d
            unfreeze_module(self.meshnet, True)
            unfreeze_module(self.coordinate3d, True)

    def forward(self, x, targets=None):
        fpn_outs = self.backbone(x)
        fpn0 = fpn_outs[0]
        loss_class=loss_2D=loss_offset2D=loss_target2D=loss_3D=total_loss=torch.nan
        
        outputs = {
                "total_loss": total_loss,
                "loss_class": loss_class,
                "loss_2D": loss_2D,
                "loss_offset2D": loss_offset2D,
                "loss_target2D": loss_target2D,
                "loss_3D": loss_3D,
            }
        
        if self.training:
            assert targets is not None

            if self.stage == 1:
                # === Stage 1: classnet のみ学習                
                loss_class, cls_prob = self.classnet(fpn0, targets["category"], targets["mesh"], 8)
                total_loss = loss_class  # 学習に使うのは2D/PAF系 classだけ

            elif self.stage == 2:
                # === Stage 2: meshnet classnet  両方で学習（backboneにも勾配は流れる）
                loss_class, cls_prob = self.classnet(fpn0, targets["category"], targets["mesh"], 8)
                loss_2D, loss_offset2D, loss_target2D, _,features = self.meshnet(fpn0, targets["mesh"], 8)
                total_loss = loss_2D+loss_class  

            else:  # self.stage == 3
                # === Stage 3: 
                
                loss_2D, loss_offset2D, loss_target2D, _,features = self.meshnet(fpn0, targets["mesh"], 8)
                loss_3D, loss_dict = \
                    self.coordinate3d(features[-1], targets["mesh"], 8)
                
                total_loss = loss_2D+loss_3D  # 学習に使うのは3D系だけ
                outputs |= loss_dict
            outputs |= {
                "total_loss": total_loss,
                "loss_class": loss_class,
                "loss_2D": loss_2D,
                "loss_offset2D": loss_offset2D,
                "loss_target2D": loss_target2D,
                "loss_3D": loss_3D,
            }
            return outputs

        else:
            # 推論時はそのまま
            cls_prob = self.classnet(fpn0)
            meshout,features = self.meshnet(fpn0)
            vertices, faces = self.coordinate3d(features[-1])
            return {"cls_prob": cls_prob, "vertices": vertices, "faces": faces}


    def visualize(self, x, targets, save_prefix="assign_vis_"):
        fpn_outs = self.backbone(x)
        self.head.visualize_assign_result(fpn_outs, targets, x, save_prefix)
