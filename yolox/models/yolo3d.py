#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch.nn as nn

#from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN


class YOLOx3D(nn.Module):
    def __init__(self, backbone,meshnet,coordinate3d):
        super().__init__()
        self.backbone = backbone
        self.meshnet = meshnet
        self.coordinate3d = coordinate3d

    def forward(self, x, targets_link=None):
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets_link is not None
            
            
            loss_2D,loss_offset2D,loss_target_mask2D,pafs= self.meshnet(
                fpn_outs[0], targets_link,8
            )
            loss_3D,loss_offset3D,loss_chamfer,loss_3DIoU = self.coordinate3d(
                pafs[-1], targets_link,8
            )
            total_loss = loss_2D+ loss_3D 
            outputs = {
                "total_loss": total_loss,
                "loss_2D":   loss_2D,
                "loss_offset2D":loss_offset2D,
                "loss_target_mask2D":loss_target_mask2D,
                #"loss_loss_chamfer_xy":loss_chamfer_xy,
                #"loss_loss_chamfer_yz":loss_chamfer_yz,
                #"loss_loss_chamfer_zx":loss_chamfer_zx,
                "loss_3D" : loss_3D,
                "loss_offset3D": loss_offset3D,
                "loss_chamfer": loss_chamfer,
                "loss_3DIoU": loss_3DIoU,
            }
            
        else:
            meshout = self.meshnet(fpn_outs[0])
            outputs = self.coordinate3d(meshout[-1])

        return outputs

    def visualize(self, x, targets, save_prefix="assign_vis_"):
        fpn_outs = self.backbone(x)
        self.head.visualize_assign_result(fpn_outs, targets, x, save_prefix)
