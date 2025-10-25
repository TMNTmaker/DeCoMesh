#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.

import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn
from yolox.data import SUNRGBD_CLASSES_20, SUNRGBD_CLASSES_38
from .base_exp import BaseExp

__all__ = ["Exp", "check_exp_value"]



from typing import Any, List
import torch
import numpy as np

def _can_stack(ts: List[torch.Tensor]) -> bool:
    return all(torch.is_tensor(t) and t.shape == ts[0].shape and t.dtype == ts[0].dtype for t in ts)

def _to_tensor(x: Any):
    if torch.is_tensor(x): return x
    if isinstance(x, np.ndarray): return torch.from_numpy(x)
    if isinstance(x, (int, float, np.integer, np.floating)): return torch.tensor(x)
    return x

def collate_auto(batch: List[Any]):
    """
    - テンソルは“形が完全一致”なら stack、そうでなければ list のまま
    - dict/tuple/list は再帰的に処理
    - 文字列などは list のまま
    """
    elem = batch[0]

    # Tensor
    if torch.is_tensor(elem):
        return torch.stack(batch, 0) if _can_stack(batch) else batch

    # numpy or number
    if isinstance(elem, (np.ndarray, np.number, int, float)):
        ts = [_to_tensor(x) for x in batch]
        return torch.stack(ts, 0) if _can_stack(ts) else ts

    # dict
    if isinstance(elem, dict):
        out = {}
        keys = elem.keys()
        for k in keys:
            out[k] = collate_auto([b[k] for b in batch])
        return out

    # list / tuple
    if isinstance(elem, (list, tuple)):
        # すべて list/tuple かつ 長さが揃っているなら "列方向に" まとめる
        if all(isinstance(b, (list, tuple)) for b in batch):
            lens = [len(b) for b in batch]
            if len(set(lens)) == 1:
                transposed = list(zip(*batch))
                return type(elem)(collate_auto(list(items)) for items in transposed)
        # それ以外（= 可変長）はそのまま返す（例: categories）
        return batch
    return batch

class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        # ---------------- model config ---------------- #
        # detect classes number of model
        self.num_classes = len(SUNRGBD_CLASSES_20)
        # factor of model depth
        self.depth = 1.00
        # factor of model width
        self.width = 1.00
        # activation name. For example, if using "relu", then "silu" will be replaced to "relu".
        self.act = "silu"

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        # If your training process cost many memory, reduce this value.
        self.data_num_workers = 4
        self.input_size = (640, 640)  # (height, width)
        # Actual multiscale ranges: [640 - 5 * 32, 640 + 5 * 32].
        # To disable multiscale training, set the value to 0.
        #self.multiscale_range = 5
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        # dir of dataset images, if data_dir is None, this project will use `datasets` dir
        self.data_dir = "datasets"
        # name of annotation file for training
        self.train_ann = "train_anno.json"
        # name of annotation file for evaluation
        self.val_ann = "test_anno.json"
        # name of annotation file for testing
        #self.test_ann = "instances_test2017.json"

        # --------------- transform config ----------------- #
        # prob of applying mosaic aug
        self.mosaic_prob = 0#1.0
        # prob of applying mixup aug
        self.mixup_prob =0# 1.0
        # prob of applying hsv aug
        self.hsv_prob = 0#1.0
        # prob of applying flip aug
        self.flip_prob = 0#0.5
        # rotation angle range, for example, if set to 2, the true range is (-2, 2)
        self.degrees = 0#10.0
        # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        # apply mixup aug or not
        self.enable_mixup = True
        self.mixup_scale = (0.5, 1.5)
        # shear angle range, for example, if set to 2, the true range is (-2, 2)
        self.shear = 2.0

        # --------------  training config --------------------- #
        # epoch number used for warmup
        self.warmup_epochs = 5
        # max training epoch
        self.max_epoch = 60#300
        # minimum learning rate during warmup
        self.warmup_lr = 0
        self.min_lr_ratio = 0.05
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = 1e-4
        # name of LRScheduler
        self.scheduler = "yoloxwarmcos"
        # last #epoch to close augmention like mosaic
        self.no_aug_epochs = 15
        # apply EMA during training
        self.ema = True

        # weight decay of optimizer
        self.weight_decay = 5e-4
        # momentum of optimizer
        self.momentum = 0.9
        # log period in iter, for example,
        # if set to 1, user could see log every iteration.
        self.print_interval = 10
        # eval period in epoch, for example,
        # if set to 1, model will be evaluate after every epoch.
        self.eval_interval = 5
        # save history checkpoint or not.
        # If set to False, yolox will only save latest and best ckpt.
        self.save_history_ckpt = True
        # name of experiment
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        # output image size during evaluation/test
        self.test_size = (640, 640)
        # confidence threshold during evaluation/test,
        # boxes whose scores are less than test_conf will be filtered
        self.test_conf = 0.01
        # nms threshold
        self.nmsthre = 0.65

    def get_model(self):
        from yolox.models import YOLOx3D, YOLOPAFPN, ClassNet,MeshNet, TriView2CoordGrid
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            #in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width,in_channels = [256, 512, 1024], depthwise=True)
            classnet = ClassNet(in_channels=256, hidden=256, num_classes=len(SUNRGBD_CLASSES_20), dropout_p=0.1)
            meshnet = MeshNet()
            coordinate3d = TriView2CoordGrid()
            self.model = YOLOx3D(backbone,classnet,meshnet,coordinate3d)

        self.model.apply(init_yolo)
        #self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        """
        Get dataset according to cache and cache_type parameters.
        Args:
            cache (bool): Whether to cache imgs to ram or disk.
            cache_type (str, optional): Defaults to "ram".
                "ram" : Caching imgs to ram for fast training.
                "disk": Caching imgs to disk for fast training.
        """
        from yolox.data import SUNRGBDDataset, TrainTransform3D

        return SUNRGBDDataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            img_size=self.input_size,
            preproc=TrainTransform3D(
                max_objects=20,
                max_faces=6,
                max_vertex=4,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_data_loader(self, batch_size, is_distributed, no_aug=True, cache_img: str = None):
        """
        Get dataloader according to cache_img parameter.
        Args:
            no_aug (bool, optional): Whether to turn off mosaic data enhancement. Defaults to False.
            cache_img (str, optional): cache_img is equivalent to cache_type. Defaults to None.
                "ram" : Caching imgs to ram for fast training.
                "disk": Caching imgs to disk for fast training.
                None: Do not use cache, in this case cache_data is also None.
        """
        from yolox.data import (
            TrainTransform3D,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master

        # if cache is True, we will create self.dataset before launch
        # else we will create self.dataset after launch
        if self.dataset is None:
            with wait_for_the_master():
                assert cache_img is None, \
                    "cache_img must be None if you didn't create self.dataset before launch"
                self.dataset = self.get_dataset(cache=False, cache_type=cache_img)

        #self.dataset = MosaicDetection(
        #    dataset=self.dataset,
        #    mosaic=False,#not no_aug,
        #    img_size=self.input_size,
        #    preproc=TrainTransform3D(
        #        max_objects=20,
        #        max_faces=6,
        #        max_vertex=4,
        #        flip_prob=self.flip_prob,
        #        hsv_prob=self.hsv_prob),
        #    degrees=self.degrees,
        #    translate=self.translate,
        #    mosaic_scale=self.mosaic_scale,
        #    mixup_scale=self.mixup_scale,
        #    shear=self.shear,
        #    enable_mixup=self.enable_mixup,
        #    mosaic_prob=self.mosaic_prob,
        #    mixup_prob=self.mixup_prob,
        #)

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        
        
        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=no_aug#not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset,collate_fn=collate_auto, **dataloader_kwargs)

        return train_loader

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if not hasattr(self, 'random_size'):
                min_size = int(self.input_size[0] / 32) - self.multiscale_range
                max_size = int(self.input_size[0] / 32) + self.multiscale_range
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        scale_z = tsize[1] / 1024#self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            targets["mesh"][..., 0] = targets["mesh"][..., 0] * scale_x
            targets["mesh"][..., 1] = targets["mesh"][..., 1] * scale_y
            targets["mesh"][..., 2] = targets["mesh"][..., 2] * scale_x
        return inputs, targets

    def get_optimizer(self, batch_size, force_rebuild: bool = False):
        need_build = force_rebuild or ("optimizer" not in self.__dict__)
        if need_build:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # no decay (BN), with decay (weights), biases

            # ★ named_modules()から拾う方式はそのままでもOKだが、BNは全種類を判定する
            import torch.nn as nn
            for k, v in self.model.named_modules():
                # bias
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)
                # weights
                if isinstance(v, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)) or ("bn" in k):
                    if hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                        pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # decay

            optimizer = torch.optim.SGD(pg0, lr=lr, momentum=self.momentum, nesterov=True)
            optimizer.add_param_group({"params": pg1, "weight_decay": self.weight_decay})
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from yolox.utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler

    def get_eval_dataset(self, **kwargs):
        from yolox.data import SUNRGBDDataset, ValTransform3D
        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)

        return SUNRGBDDataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="SUNRGBD",
            img_size=self.test_size,
            preproc=ValTransform3D(legacy=legacy),
        )

    def get_eval_loader(self, batch_size, is_distributed, **kwargs):
        valdataset = self.get_eval_dataset(**kwargs)

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset,collate_fn=collate_auto, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import SUNRGBDEvaluator

        return SUNRGBDEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed,
                                            testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            testdev=testdev,
        )

    def get_trainer(self, args):
        from yolox.core import Trainer
        trainer = Trainer(self, args)
        # NOTE: trainer shouldn't be an attribute of exp object
        return trainer

    def eval(self, model, evaluator, half=False, return_outputs=False):
        return evaluator.evaluate(model, half, return_outputs=return_outputs)


def check_exp_value(exp: Exp):
    h, w = exp.input_size
    assert h % 32 == 0 and w % 32 == 0, "input size must be multiples of 32"
