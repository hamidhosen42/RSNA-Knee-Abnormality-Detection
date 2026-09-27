#!/usr/bin/env python3
"""Diagnostics for train_ours_leg.py on Kaggle, round 2.

Round 1 showed: NFS page cache from corpus reads is reclaimable and fadvise-droppable, yet training
runs lost ~5 MB of *available* RAM per study read; and the training step took ~5 s regardless of
resolution, checkpointing or memory format.

C. Loader memory: the real training DataLoader (pread rows -> uint8 windows -> collate), no model,
   with/without pin_memory and fadvise; RSS of main and workers, shmem, cgroup anon/file.
D. GPU scaling: raw fp16 matmul TFLOPS, a ResNet-50 reference, and EfficientNetV2-S step time vs
   batch size, resolution and single GPU vs DataParallel, to see whether time tracks compute.
"""
import glob, os, random, time

import numpy as np
import psutil
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def mem(tag):
    vm = psutil.virtual_memory()
    info = {l.split(":")[0]: int(l.split()[1]) / 2**20 for l in open("/proc/meminfo")}
    st = {}
    if os.path.exists("/sys/fs/cgroup/memory.stat"):
        st = {k: int(v) / 2**30 for k, v in (l.split() for l in open("/sys/fs/cgroup/memory.stat"))}
    me = psutil.Process()
    kids = me.children(recursive=True)
    rss_k = sum(k.memory_info().rss for k in kids if k.is_running()) / 2**30
    print(f"  {tag}: used {(vm.total - vm.available) / 2**30:.1f}G shmem {info.get('Shmem', 0):.2f}G | "
          f"cgroup anon {st.get('anon', 0):.1f}G file {st.get('file', 0):.1f}G shmem {st.get('shmem', 0):.2f}G | "
          f"rss main {me.memory_info().rss / 2**30:.2f}G workers({len(kids)}) {rss_k:.2f}G", flush=True)


class NpyRows:
    def __init__(self, path, fadvise):
        with open(path, "rb") as f:
            major, _ = np.lib.format.read_magic(f)
            hdr = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
            self.shape, _, dtype = hdr(f)
            self.offset = f.tell()
        self.path, self.dtype, self.fd, self.fadvise = path, np.dtype(dtype), None, fadvise
        self.row_bytes = int(np.prod(self.shape[1:])) * self.dtype.itemsize

    def __getitem__(self, row):
        if self.fd is None:
            self.fd = os.open(self.path, os.O_RDONLY)
        pos = self.offset + row * self.row_bytes
        buf = os.pread(self.fd, self.row_bytes, pos)
        if self.fadvise:
            os.posix_fadvise(self.fd, pos, self.row_bytes, os.POSIX_FADV_DONTNEED)
        return np.frombuffer(buf, dtype=self.dtype).reshape(self.shape[1:])


class Windows(Dataset):                                   # same work as StudyWindows(train=True)
    def __init__(self, path, masks, n, fadvise, k=12):
        self.path, self.masks, self.n, self.fadvise, self.k, self.vols = path, masks, n, fadvise, k, None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.vols is None:
            self.vols = NpyRows(self.path, self.fadvise)
        vol = self.vols[i]
        valid = np.where(self.masks[i] > 0)[0]
        lo, hi = (int(valid.min()), int(valid.max())) if len(valid) >= 3 else (0, 2)
        cs = [c for c in range(lo + 1, hi)] or [1]
        cs = (cs * (self.k // len(cs) + 1))[:self.k]
        random.shuffle(cs)
        wins = np.stack([np.stack([vol[c - 1], vol[c], vol[c + 1]], 0) for c in cs], 0)
        return torch.from_numpy(wins), torch.zeros(12)


def part_c():
    print("=== C. training DataLoader memory (no model) ===", flush=True)
    path = sorted(glob.glob("/kaggle/input/**/all_vols.npy", recursive=True))[0]
    masks = np.load(path.replace("all_vols", "all_masks"))
    mem("start")
    for name, pin, fad, iters, start in [("pin_memory=True  fadvise=0", True, 0, 250, 0),
                                         ("pin_memory=False fadvise=0", False, 0, 150, 2000),
                                         ("pin_memory=True  fadvise=1", True, 1, 150, 2000)]:
        ds = Windows(path, masks, 3200, fad)
        order = list(range(start, 3200)) + list(range(start))
        dl = DataLoader(torch.utils.data.Subset(ds, order), batch_size=8, shuffle=False, num_workers=4,
                        pin_memory=pin, collate_fn=lambda b: (torch.stack([x for x, _ in b]), torch.stack([y for _, y in b])))
        print(f" -- {name}", flush=True)
        t0 = time.time()
        for it, (xb, yb) in enumerate(dl):
            if pin:
                xb = xb.cuda(non_blocking=True)
            if it % 50 == 0:
                mem(f"it{it} ({(it + 1) * 8 / (time.time() - t0):.1f} studies/s)")
            if it >= iters:
                break
        del dl
        time.sleep(3)
        mem("after loader exit")


def timeit(fn, n=8, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t = time.time(); fn(); torch.cuda.synchronize(); ts.append(time.time() - t)
    return float(np.median(ts))


class Head(nn.Module):
    def __init__(self, bb, n=12):
        super().__init__()
        self.backbone, F = bb, bb.num_features
        self.norm = nn.LayerNorm(F)
        self.att = nn.Sequential(nn.Linear(F, 256), nn.Tanh(), nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F))

    def forward(self, x):
        B, K = x.shape[:2]
        h = self.norm(self.backbone(x.flatten(0, 1)).view(B, K, -1))
        a = torch.softmax(self.att(h), dim=1)
        return (torch.einsum("bkn,bkf->bnf", a, h) * self.clsW).sum(-1)


def part_d():
    print("=== D. GPU scaling ===", flush=True)
    torch.backends.cudnn.benchmark = True
    print(f"  torch {torch.__version__} cuda {torch.version.cuda} cudnn {torch.backends.cudnn.version()} "
          f"timm {timm.__version__} cpus {os.cpu_count()} | {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}", flush=True)
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    s = timeit(lambda: a @ a, n=20)
    print(f"  fp16 matmul 4096^3: {2 * 4096**3 / s / 1e12:.1f} TFLOPS", flush=True)
    r50 = timm.create_model("resnet50", pretrained=False).cuda()
    opt = torch.optim.SGD(r50.parameters(), lr=0.01)
    x = torch.randn(64, 3, 224, 224, device="cuda")

    def r50_step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            r50(x).float().mean().backward()
        opt.step()
    s = timeit(r50_step)
    print(f"  resnet50 fp16 train bs64@224 on 1 GPU: {64 / s:.0f} img/s (a healthy T4 does ~250-300)", flush=True)
    del r50, opt, x
    for bs, res, dp, ck in [(8, 384, True, True), (16, 384, True, True), (8, 384, False, True), (4, 384, False, True),
                            (8, 256, True, True), (16, 256, True, True), (8, 256, True, False)]:
        torch.cuda.empty_cache()
        try:
            bb = timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k", pretrained=False, num_classes=0, global_pool="avg")
            if ck:
                bb.set_grad_checkpointing(True)
            core = Head(bb).cuda()
            model = nn.DataParallel(core) if dp else core
            opt = torch.optim.AdamW(core.parameters(), lr=1e-4)
            scaler = torch.amp.GradScaler("cuda")
            xx = torch.randn(bs, 12, 3, res, res, device="cuda")
            yy = torch.rand(bs, 12, device="cuda")

            def step():
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = nn.functional.binary_cross_entropy_with_logits(model(xx).float(), yy)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            s = timeit(step, n=5, warm=2)

            def fwd():
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    model(xx)
            f = timeit(fwd, n=5, warm=1)
            print(f"  effv2s bs{bs:>2} res{res} {'DP x2' if dp else '1 GPU'} ckpt={ck!s:5}: train {s:.2f} s/step "
                  f"= {bs / s:.2f} studies/s ({bs * 12 / s:.0f} img/s) | fwd-only {bs * 12 / f:.0f} img/s", flush=True)
            del model, core, bb, opt, xx
        except RuntimeError as e:
            print(f"  effv2s bs{bs} res{res} dp={dp} ckpt={ck}: FAILED {str(e)[:100]}", flush=True)


if __name__ == "__main__":
    part_d()
    part_c()
