"""Plain + anatomical left-right mirror TTA for the 2.5D ConvNeXt Mirror students (goodpjw reader architecture).

Same command line as infer.py:  python infer_tta.py <data_dir> <split> <out_csv> <ckpt> [<ckpt> ...]
Writes <out_csv> = 0.5 x (plain + mirror) of the per-model probability mean (infer.run's own aggregation), plus
<stem>_plain.csv and <stem>_mirror.csv for diagnostics.  knee.py, preprocess.py and infer.py are imported unchanged.

Why this mirror is anatomical.  preprocess.load_series (CANON) stores every series in a fixed patient (LPS) frame:
  sagittal  volume axis 0 (slice order) runs patient right -> left; image columns anterior -> posterior
  coronal   image columns (array axis 2) run patient right -> left
  axial     image columns (array axis 2) run patient right -> left
There is no laterality canonicalisation, so a left knee is stored as the mirror image of a right knee, and the
students were trained on both sides.  Mirroring the patient (x -> -x) therefore gives a contralateral-looking knee:
  sagittal  reverse the slice order = reverse the 3 slice channels of every window and pos -> 1 - pos
            (window_centres is symmetric, the model is invariant to the token order within a slot)
  coronal / axial  flip the image columns; pos unchanged
Medial stays medial (the labels are anatomical), so no output is swapped.  A series whose orientation was not
canonicalised (no ImageOrientationPatient or ImagePositionPatient: load_series sets no slice_gap) sends the whole
study to the plain pass (its mirror output is its plain output), so a mirrored study is never anatomically mixed.

Runtime: every study is decoded once; with two GPUs the plain pass runs on cuda:0 and the mirror pass on cuda:1
in parallel (same batches), so the wall time stays close to the plain run.  With one GPU the passes run one after
the other on cuda:0.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

from infer import _n_slices, load_models
from knee import LABELS, N_SLOTS, build_study_table, window_centres
from preprocess import SIZE, load_series

MIRROR_CODE = {"Sagittal": 1, "Coronal": 2, "Axial": 2}   # 1 = reverse slice order, 2 = flip image columns


class MirrorStudyDataset(torch.utils.data.Dataset):
    """infer.DicomStudyDataset (x, mask, pos built by the same lines) plus one mirror code per slot."""

    def __init__(self, studies, table, root, k):
        self.studies, self.table, self.root, self.k = list(studies), table, root, k

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, i):
        st = self.studies[i]
        slots = self.table.get(st, [[] for _ in range(N_SLOTS)])
        x = np.zeros((N_SLOTS, self.k, 3, SIZE, SIZE), np.uint8)
        mask, pos = np.zeros(N_SLOTS, bool), np.zeros((N_SLOTS, self.k), np.float32)
        code = np.zeros(N_SLOTS, np.int8)
        for s, cands in enumerate(slots):
            if not cands:
                continue
            try:
                vol, info = load_series(f"{self.root}/{st}/{cands[0]}")
            except Exception:
                continue
            n = len(vol)
            if n < 3:
                continue
            c = window_centres(n, self.k, False)
            x[s] = np.stack([vol[np.clip([j - 1, j, j + 1], 0, n - 1)] for j in c])
            mask[s], pos[s] = True, c / max(n - 1, 1)
            code[s] = MIRROR_CODE.get(info.get("plane"), 0) if "slice_gap" in info else 0
        ok = bool(mask.any()) and bool((code[mask] > 0).all())
        if not ok:
            code[:] = 0   # orientation unknown for a used series (or nothing to read): mirror pass = plain pass
        return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(pos), torch.from_numpy(code), torch.tensor(ok)


def mirror_batch(x, pos, code):
    """x uint8 [B, S, K, 3, H, W], pos [B, S, K], code int8 [B, S] -> the patient-mirrored (x, pos)."""
    xm, pm = x.clone(), pos.clone()
    sag, wid = code == 1, code == 2
    if bool(sag.any()):
        xm[sag] = x[sag].flip(2)       # [n, K, 3, H, W]: the 3 slice channels reversed
        pm[sag] = 1.0 - pos[sag]
    if bool(wid.any()):
        xm[wid] = x[wid].flip(-1)      # image columns: patient right <-> left
    return xm, pm


def _spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def run_tta(data_dir, split, ckpts, k=16, studies=None, workers=4, bs=4, devices=None):
    t0 = time.time()
    se = pd.read_csv(f"{data_dir}/{split}_series.csv")
    root = f"{data_dir}/{split}_series"
    if studies is None:
        studies = pd.read_csv(f"{data_dir}/{split}.csv").StudyInstanceUID.tolist()
    se = se[se.StudyInstanceUID.isin(set(studies))]
    n_files = {}
    for st, sr in zip(se.StudyInstanceUID, se.SeriesInstanceUID):
        n_files[sr] = _n_slices(f"{root}/{st}/{sr}")
    table = build_study_table(se, n_files)
    if devices is None:
        devices = ["cuda:%d" % g for g in range(min(2, torch.cuda.device_count()))] or ["cpu"]
    on_gpu = devices[0].startswith("cuda")
    dl = torch.utils.data.DataLoader(MirrorStudyDataset(studies, table, root, k), batch_size=bs,
                                     num_workers=workers, pin_memory=on_gpu)
    models = [load_models(ckpts, d) for d in devices]

    @torch.no_grad()   # grad mode is thread-local: the mirror thread needs its own no_grad
    def score(j, x, mask, pos, code=None):
        assert not torch.is_grad_enabled()
        d = devices[j]
        x, mask, pos = x.to(d, non_blocking=True), mask.to(d), pos.to(d)
        if code is not None:
            x, pos = mirror_batch(x, pos, code.to(d))
        with torch.autocast("cuda", dtype=torch.float16, enabled=on_gpu):
            return torch.stack([m(x, mask, pos, res=res).float().sigmoid() for m, res in models[j]]).mean(0).cpu()

    pool = ThreadPoolExecutor(1) if len(devices) > 1 else None
    plain, mirror, oks = [], [], []
    for x, mask, pos, code, ok in dl:
        if pool is not None:
            fut = pool.submit(score, 1, x, mask, pos, code)
            plain.append(score(0, x, mask, pos))
            mirror.append(fut.result())
        else:
            plain.append(score(0, x, mask, pos))
            mirror.append(score(0, x, mask, pos, code))
        oks.append(ok)
    idx = pd.Index(studies, name="StudyInstanceUID")
    p = pd.DataFrame(torch.cat(plain).numpy(), index=idx, columns=LABELS)
    q = pd.DataFrame(torch.cat(mirror).numpy(), index=idx, columns=LABELS)
    ok = torch.cat(oks).numpy()
    info = {"studies": len(studies), "mirrored": int(ok.sum()), "devices": devices, "seconds": round(time.time() - t0, 1),
            "spearman_plain_mirror": {c: round(_spearman(p[c].values[ok], q[c].values[ok]), 4) for c in LABELS}}
    # direction check: a correct anatomical mirror keeps Medial with Medial (a side-swapping one would pair it with Lateral)
    for a, b in (("Medial Meniscus", "Lateral Meniscus"), ("Medial OA", "Lateral OA")):
        info["same_minus_cross_" + a.split()[1]] = round(
            min(_spearman(p[a].values[ok], q[a].values[ok]) - _spearman(p[a].values[ok], q[b].values[ok]),
                _spearman(p[b].values[ok], q[b].values[ok]) - _spearman(p[b].values[ok], q[a].values[ok])), 4)
    return p, q, info


if __name__ == "__main__":
    data_dir, split, out_csv, ckpts = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    p, q, info = run_tta(data_dir, split, ckpts)
    stem = out_csv[:-4] if out_csv.endswith(".csv") else out_csv
    p.to_csv(stem + "_plain.csv")
    q.to_csv(stem + "_mirror.csv")
    (0.5 * (p + q)).to_csv(out_csv)
    print("TTA_INFO", json.dumps(info), flush=True)
    print("wrote", out_csv)
