"""Reading a corpus as fixed windows.

The batch-shaped view of a corpus, used by the batch trainer and by the
benchmarks. The streaming trainer uses stream.Reader instead, which walks a
corpus in order behind a cache rather than drawing independent windows.
"""

import json
import os

import numpy as np
import torch

from . import device as _device


class Corpus:
    def __init__(self, data_dir, device, block, batch):
        self.meta = json.load(open(os.path.join(data_dir, "meta.json")))
        self.train = np.memmap(os.path.join(data_dir, "train.bin"),
                               dtype=np.uint16, mode="r")
        self.val = np.memmap(os.path.join(data_dir, "val.bin"),
                             dtype=np.uint16, mode="r")
        self.device = device
        self.block = block
        self.batch = batch
        self.self_tokens = None      # filled in by the trainer

    def _draw(self, data, n, rng):
        ix = rng.integers(0, len(data) - self.block - 1, size=n)
        x = np.stack([data[i:i + self.block] for i in ix]).astype(np.int64)
        y = np.stack([data[i + 1:i + self.block + 1] for i in ix]).astype(np.int64)
        return x, y

    def batch_for(self, split, rng, self_frac=0.0):
        data = self.train if split == "train" else self.val
        n_self = 0
        if (split == "train" and self_frac > 0 and self.self_tokens is not None
                and len(self.self_tokens) > self.block + 2):
            n_self = int(round(self.batch * self_frac))
        xs, ys = self._draw(data, self.batch - n_self, rng)
        if n_self:
            xa, ya = self._draw(self.self_tokens, n_self, rng)
            xs = np.concatenate([xs, xa])
            ys = np.concatenate([ys, ya])
        # Pinning needs a backend that implements it, which is asked rather
        # than inferred from the device name - torch grew MPS pinning during
        # the 2.x series. `non_blocking` travels with the same answer, because
        # asking for an async copy out of unpinned memory is a synchronous one
        # that merely looks asynchronous.
        pin = _device.supports_pinning(self.device)
        x = torch.from_numpy(xs)
        y = torch.from_numpy(ys)
        if pin:
            x, y = x.pin_memory(), y.pin_memory()
        x = x.to(self.device, non_blocking=pin)
        y = y.to(self.device, non_blocking=pin)
        return x, y


