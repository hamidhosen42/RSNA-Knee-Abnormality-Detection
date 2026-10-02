"""Generate e13/train_e13.py from ../train_ours_leg.py (E11, verified on Kaggle).

Everything that worked in E11 is kept byte-for-byte: frozen folds + checksum, soft-label OOF metric
with the uncertain band masked, gold-58 as diagnostic only, fp16 on T4, one fold per GPU process,
SWA + BatchNorm recalibration, glibc malloc fix, wall-clock budget. Only the data and the model
change, so E13 vs E11 differs in exactly those two places.

    python make_train_e13.py      # rewrites train_e13.py next to this file
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE.parent / "train_ours_leg.py").read_text()


def cut(text, start, end, new):
    i, j = text.index(start), text.index(end)
    assert i < j, (start, end)
    return text[:i] + new + text[j:]


def sub(text, old, new, count=1):
    assert text.count(old) == count, (old[:60], text.count(old))
    return text.replace(old, new)


HEADER = '''#!/usr/bin/env python3
"""E13: multi-view 2.5D leg trained on our own cache (Kaggle GPU script).

Same scaffold as E11 (train_ours_leg.py); changes, and why:
  1. Data: the E13 cache (kernels hosen42/rsna-knee-e13-cache-a/-b), built from the official DICOMs
     by e13/knee_prep.py, the SAME file the submission leg will import. E11 trained on a third-party
     corpus and rebuilt test inputs with separate code, then lost 0.008 on the LB.
  2. Input: 4 views (Axial fluid, Sagittal non-fluid, Coronal fluid, Sagittal fluid) x K three-slice
     windows; training shifts every view's window centres by up to half a slice.
  3. Model (KneeNet): timm backbone per window, then per-finding attention over slices inside each
     view, then per-finding attention over the views that are present.
  4. Defaults sized for a T4: ResNet34 at 224 px, 4 studies per step.
Outputs as in E11 under arms/<arm>/weights/, plus OOF/gold npz and train_<tag>.json.
"""
'''

CACHE = '''def cache_files(root, max_depth=4):
    """{StudyInstanceUID: npz path} from every attached E13 cache shard (a dir named 'cache')."""
    files, base = {}, root.rstrip(os.sep).count(os.sep)
    for d, dirs, fs in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in _DICOM_TREES and x != "competitions"
                         and d.count(os.sep) - base < max_depth)
        if os.path.basename(d) == "cache":
            for f in fs:
                if f.endswith(".npz"):
                    files.setdefault(f[:-4], os.path.join(d, f))
            dirs[:] = []
    return files


def cache_spec(files):
    """spec.json of the shard(s); all shards must agree."""
    specs = set()
    for d in {os.path.dirname(os.path.dirname(p)) for p in files.values()}:
        p = os.path.join(d, "spec.json")
        if os.path.exists(p):
            s = json.load(open(p))
            specs.add(json.dumps({k: s[k] for k in ("views", "n_slices", "res", "crop_mm", "clip_pct", "version")},
                                 sort_keys=True))
    if len(specs) != 1:
        raise SystemExit(f"cache shards disagree or lack spec.json: {specs}")
    return json.loads(specs.pop())


'''

DATA = '''class E13Studies(Dataset):
    """One study -> (uint8 windows (V, K, 3, H, W), view mask (V,)), label (12,)."""

    def __init__(self, files, ids, labels, k, train):
        self.files, self.ids, self.labels, self.k, self.train = files, ids, labels, k, train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        uid = self.ids[i]
        with np.load(self.files[uid]) as z:
            vol, mask = z["vol"], z["mask"]
        V, S = vol.shape[:2]
        base = np.linspace(0, S - 1, self.k)
        out = np.empty((V, self.k, 3) + vol.shape[2:], np.uint8)
        for v in range(V):
            c = base + (random.uniform(-0.5, 0.5) if self.train else 0.0)
            c = np.clip(np.round(c).astype(int), 0, S - 1)
            for ch, d in enumerate((-1, 0, 1)):
                out[v, :, ch] = vol[v, np.clip(c + d, 0, S - 1)]
        return (torch.from_numpy(out), torch.from_numpy(mask.astype(np.float32))), \\
            torch.from_numpy(self.labels[uid])


def collate(batch):
    return ((torch.stack([b[0][0] for b in batch]), torch.stack([b[0][1] for b in batch])),
            torch.stack([b[1] for b in batch]))


def to_input(xb, res, mean, std, dev, train):
    """((B,V,K,3,H,W) uint8, (B,V) mask) -> normalized float on `dev`, plus the mask."""
    wins, mask = xb
    x = wins.to(dev, non_blocking=True).float().div_(255.0)
    shp = x.shape
    x = x.flatten(0, 2)
    if x.shape[-1] != res or x.shape[-2] != res:
        x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
    x = x.view(*shp[:3], 3, res, res)
    if train:                            # no flips: laterality is signal; mild intensity jitter only
        x = (x * (1.0 + (torch.rand(shp[0], 1, 1, 1, 1, 1, device=x.device) - 0.5) * 0.20)).clamp_(0, 1)
    return (x - mean.view(1, 1, 1, 3, 1, 1)) / std.view(1, 1, 1, 3, 1, 1), mask.to(dev, non_blocking=True)


'''

MODEL = '''class KneeNet(nn.Module):
    """Backbone per window; per-finding attention over slices within a view, then over views."""

    def __init__(self, backbone, F_dim, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
        self.viewW = nn.Parameter(torch.zeros(n, F_dim))
        self.viewb = nn.Parameter(torch.zeros(n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        nn.init.trunc_normal_(self.viewW, std=0.02)
        self.channels_last = False

    def forward(self, inp):
        x, mask = inp
        B, V, K = x.shape[:3]
        x = x.flatten(0, 2)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        h = self.norm(self.backbone(x).view(B, V, K, -1))
        a = torch.softmax(self.att(h).float(), dim=2).to(h.dtype)           # (B,V,K,n) over slices
        per_view = torch.einsum("bvkn,bvkf->bvnf", a, h)                     # (B,V,n,F)
        vlog = (per_view * self.viewW).sum(-1).float() + self.viewb          # (B,V,n)
        present = mask > 0.5
        present = present | ~present.any(1, keepdim=True)                    # no view at all: use all
        vlog = vlog.masked_fill(~present.unsqueeze(-1), float("-inf"))
        w = torch.softmax(vlog, dim=1).to(h.dtype)
        pooled = torch.einsum("bvn,bvnf->bnf", w, per_view)
        return (pooled * self.clsW).sum(-1) + self.clsb


'''

t = HEADER + src[src.index("import argparse"):]
t = cut(t, "# How the public corpus was built", "POS, NEG = ", "")
t = cut(t, "class NpyRows:", "# ---------------------------------------------------------------------------------------------\n# Frozen folds", CACHE)
t = cut(t, "# ---------------------------------------------------------------------------------------------\n# Data: per-study bag",
        "# ---------------------------------------------------------------------------------------------\n# Model:", DATA)
t = cut(t, "class RaptorClassifier(nn.Module):", "# ---------------------------------------------------------------------------------------------\n# Metrics", MODEL)
# defaults
t = sub(t, 'env("OURS_ARCH", "tf_efficientnetv2_s.in21k_ft_in1k")', 'env("OURS_ARCH", "resnet34.a1_in1k")')
t = sub(t, 'env("OURS_ARM", "effv2s")', 'env("OURS_ARM", "e13r34")')
t = sub(t, 'env("OURS_RES", 384, int)', 'env("OURS_RES", 224, int)')
t = sub(t, 'env("OURS_K", 12, int)', 'env("OURS_K", 24, int)')
t = sub(t, 'env("OURS_EPOCHS", 16, int)', 'env("OURS_EPOCHS", 15, int)')
t = sub(t, 'env("OURS_BS", 8, int)', 'env("OURS_BS", 4, int)')
t = sub(t, 'env("OURS_BB_LR", 1e-4, float)', 'env("OURS_BB_LR", 2e-4, float)')
t = sub(t, 'env("OURS_GRAD_CKPT", 1, int)', 'env("OURS_GRAD_CKPT", 0, int)')
t = sub(t, "a.epochs, a.bs, a.k, a.k_eval, a.workers, a.topk, a.bn_batches = 2, 2, 4, 6, 0, 2, 3",
        "a.epochs, a.bs, a.k, a.k_eval, a.workers, a.topk, a.bn_batches = 2, 2, 4, 4, 0, 2, 3")
# data discovery and datasets
t = sub(t, '''    ids = open_ids(a.input_root)
    id2row = {u: i for i, u in enumerate(ids)}
    paths = corpus_paths(a.input_root)
''', '''    files = cache_files(a.input_root)
    if not files:
        raise SystemExit("no E13 cache attached (kernels hosen42/rsna-knee-e13-cache-a and -b)")
    SPEC = cache_spec(files)
    ids = sorted(files)
    print(f"cache: {len(files)} studies | spec {SPEC}", flush=True)
''')
t = sub(t, "core = RaptorClassifier(bb, F_dim=bb.num_features).to(dev)", "core = KneeNet(bb, F_dim=bb.num_features).to(dev)")
t = sub(t, "    mean_t = torch.tensor(mean, dtype=torch.float32, device=dev).view(1, 1, 3, 1, 1)\n"
           "    std_t = torch.tensor(std, dtype=torch.float32, device=dev).view(1, 1, 3, 1, 1)\n",
        "    mean_t = torch.tensor(mean, dtype=torch.float32, device=dev)\n"
        "    std_t = torch.tensor(std, dtype=torch.float32, device=dev)\n")
t = sub(t, "mk = dict(paths=paths, id2row=id2row, labels=labels)", "mk = dict(files=files, labels=labels)")
t = t.replace("StudyWindows(", "E13Studies(")
t = sub(t, '"corpus": CORPUS_SPEC,', '"corpus": SPEC,', count=2)
for gone in ("CORPUS_SPEC", "NpyRows", "open_ids(", "corpus_paths(", "RaptorClassifier", "id2row"):
    assert gone not in t, gone
compile(t, "train_e13.py", "exec")
(HERE / "train_e13.py").write_text(t)
print("wrote train_e13.py", len(t.splitlines()), "lines")
