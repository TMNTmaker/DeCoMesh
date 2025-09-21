#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
import os
import shutil
from loguru import logger
import re
from collections import OrderedDict
import torch
    
def load_ckpt(model, ckpt):
    # ckpt は torch.load 済みの dict を想定
    sd_ckpt_all = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    sd_model = model.state_dict()

    # どの ckpt キーを採用するか（ckpt 側に対してフィルタを掛ける）
    ALLOW_PREFIXES = ("backbone.", "backbone.backbone.", "head.", "backbone.out")
    DROP_PREFIXES  = tuple()  # 必要に応じて調整

    def allowed_ckpt_key(k: str) -> bool:
        if DROP_PREFIXES and any(k.startswith(dp) for dp in DROP_PREFIXES):
            return False
        return any(k.startswith(ap) for ap in ALLOW_PREFIXES)

    # darkブロック (例: backbone.backbone.dark3.1.) を拾う
    DARK_RE = re.compile(r"(?:^|\.)(dark\d+\.\d+)\.")

    # ckptキー → モデルキー候補を列挙（形状一致で採用）
    def candidate_model_keys_from_ckpt_key(k: str):
        cands = set()
        # そのまま
        cands.add(k)
        # layer1/2 ↔ conv1/2
        cands.add(k.replace(".layer1.", ".conv1.").replace(".layer2.", ".conv2."))
        cands.add(k.replace(".conv1.", ".layer1.").replace(".conv2.", ".layer2."))
        # prefix 相互
        if k.startswith("backbone.backbone."):
            k2 = k.replace("backbone.backbone.", "backbone.", 1)
            cands.add(k2)
            cands.add(k2.replace(".layer1.", ".conv1.").replace(".layer2.", ".conv2."))
        if k.startswith("backbone."):
            k2 = k.replace("backbone.", "backbone.backbone.", 1)
            cands.add(k2)
            cands.add(k2.replace(".layer1.", ".conv1.").replace(".layer2.", ".conv2."))

        # stem の揺れ（必要な分だけ。過剰にしない）
        if ".stem." in k:
            cands.add(k.replace(".stem.0.", ".stem.conv."))
            cands.add(k.replace(".stem.1.", ".stem.conv2."))
            cands.add(k.replace(".stem.2.", ".stem.conv3."))

        # ★ C3系の揺れ対策：layer{1,2} → m.{i}.conv{1,2} を複数候補化
        m = DARK_RE.search(k)
        if m:
            block = m.group(1)  # e.g., "dark3.1"
            for L in (1, 2):
                if f".layer{L}." in k or f".conv{L}." in k:
                    base = k.replace(f".layer{L}.", f".conv{L}.")  # まず convL に寄せる
                    for i in range(10):  # m.0〜m.9 を探索（必要なら上限を増やす）
                        cands.add(base.replace(f".{block}.", f".{block}.m.{i}."))
                        # prefix の相互置換も合わせ技
                        if base.startswith("backbone.backbone."):
                            bb = base.replace("backbone.backbone.", "backbone.", 1)
                            cands.add(bb.replace(f".{block}.", f".{block}.m.{i}."))
                        if base.startswith("backbone."):
                            bbb = base.replace("backbone.", "backbone.backbone.", 1)
                            cands.add(bbb.replace(f".{block}.", f".{block}.m.{i}."))
        return list(cands)

    remapped = OrderedDict()
    hit = miss = filt = 0

    for k_ckpt, v_ckpt in sd_ckpt_all.items():
        if not allowed_ckpt_key(k_ckpt):
            filt += 1
            continue
        matched = False
        for k_model in candidate_model_keys_from_ckpt_key(k_ckpt):
            if k_model in sd_model and sd_model[k_model].shape == v_ckpt.shape:
                remapped[k_model] = v_ckpt  # ckptの値を入れる
                matched = True
                break
        hit += int(matched)
        miss += int(not matched)

    print(f"Filtered by prefix: {filt}")
    print(f"Matched (name+shape): {hit},  Unmatched after remap: {miss}")
    blk=0    
    #model_state_dict = model.state_dict()
    #load_dict = {}
    #for key_model, v in model_state_dict.items():
    #    if key_model not in ckpt:
    #        logger.warning(
    #            "{} is not in the ckpt. Please double check and see if this is desired.".format(
    #                key_model
    #            )
    #        )
    #        continue
    #    v_ckpt = ckpt[key_model]
    #    if v.shape != v_ckpt.shape:
    #        logger.warning(
    #            "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
    #                key_model, v_ckpt.shape, key_model, v.shape
    #            )
    #        )
    #        continue
    #    load_dict[key_model] = v_ckpt
#
    #model.load_state_dict(load_dict, strict=False)
    return model


def save_checkpoint(state, is_best, save_dir, model_name=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, model_name + "_ckpt.pth")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_dir, "best_ckpt.pth")
        shutil.copyfile(filename, best_filename)
