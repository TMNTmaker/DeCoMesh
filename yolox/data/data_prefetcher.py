import torch
import numpy as np

def _pick(d, keys):
    # dict から最初に見つかったキーを返す（なければ None）
    if isinstance(d, dict):
        for k in keys:
            if k in d:
                return d[k]
    return None

def _to_device(obj, device="cuda", non_blocking=True):
    # 再帰的に Tensor をデバイス移動（numpy も Tensor 化）
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).to(device, non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {k: _to_device(v, device, non_blocking) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [_to_device(v, device, non_blocking) for v in obj]
        return type(obj)(t) if not isinstance(obj, list) else t
    return obj

def _record_stream(obj, stream):
    # 再帰的に Tensor にだけ record_stream を張る
    if torch.is_tensor(obj):
        obj.record_stream(stream); return
    if isinstance(obj, dict):
        for v in obj.values():
            _record_stream(v, stream)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _record_stream(v, stream)
        return

class DataPrefetcher:
    """
    loader は (img, tgts, _, _, _, _, _, _) を返す想定
    tgts から 'categories' と 'mesh' を抜き出し、GPUへ非同期転送
    """
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.next_input = None
        self.next_target = None
        self.preload()

    def preload(self):
        try:
            self.next_input, tgts, *_ = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        cat  = _pick(tgts, ["categories"])  # 例: (B,) or list[Tensor] etc.
        mesh = _pick(tgts, ["mesh"])        # 例: list[Tensor(Ni, …)] など可変長でもOK
        self.next_target = {"category": cat, "mesh": mesh}

        with torch.cuda.stream(self.stream):
            self.input_cuda()  # 画像を非同期で GPU へ
            self.next_target = _to_device(self.next_target, device="cuda", non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        inp = self.next_input
        tgt = self.next_target

        if inp is not None:
            self.record_stream(inp)  # 画像（Tensor）に record_stream
        if tgt is not None:
            _record_stream(tgt, torch.cuda.current_stream())  # ← dict/list を再帰処理

        self.preload()
        return inp, tgt

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input_tensor):
        input_tensor.record_stream(torch.cuda.current_stream())
