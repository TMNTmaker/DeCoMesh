#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch.nn as nn

from yolox.exp import Exp3D as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_model(self,):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOx3D, YOLOPAFPN, MeshNet, TriView2CoordGrid
            backbone = YOLOPAFPN()
            meshnet = MeshNet()
            coordinate3d = TriView2CoordGrid()
            self.model = YOLOx3D(backbone,meshnet,coordinate3d)
        self.model.apply(init_yolo)

        return self.model
