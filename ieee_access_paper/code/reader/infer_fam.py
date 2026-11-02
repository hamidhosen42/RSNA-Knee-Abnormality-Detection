"""Family-balanced student mean (H982 / H983): 0.5 x mean(Mirror students) + 0.5 x mean(S1 students), one decode pass.

Same command line as infer.py:  python infer_fam.py <data_dir> <split> <out_csv> <ckpt> [<ckpt> ...]
infer.py's run() is followed line for line; only the model average changes (S1 = files named cnxt_plE75*, the rest are
Mirror students).  H976's equal 7-checkpoint mean gave the 2 S1 students 2/7 of the student leg and beat H969 on the public
LB; this gives the two teacher families equal shares.  knee.py, preprocess.py and infer.py are imported unchanged.
"""
import os
import sys

import pandas as pd
import torch

from infer import DicomStudyDataset, _n_slices, load_models
from knee import LABELS, build_study_table

S1_PREFIX = "cnxt_plE75"


def family_weights(ckpts):
    """Per-checkpoint weights: each family (S1 / Mirror) gets 0.5, split equally inside it."""
    s1 = [os.path.basename(p).startswith(S1_PREFIX) for p in ckpts]
    n1, n0 = sum(s1), len(s1) - sum(s1)
    if n1 == 0 or n0 == 0:
        raise ValueError("family-balanced mean needs both S1 and Mirror checkpoints: %s" % [os.path.basename(p) for p in ckpts])
    return [0.5 / n1 if f else 0.5 / n0 for f in s1]


@torch.no_grad()
def run(data_dir, split, ckpts, k=16, studies=None, workers=4, bs=4, device="cuda"):
    se = pd.read_csv(f"{data_dir}/{split}_series.csv")
    root = f"{data_dir}/{split}_series"
    if studies is None:
        studies = pd.read_csv(f"{data_dir}/{split}.csv").StudyInstanceUID.tolist()
    se = se[se.StudyInstanceUID.isin(set(studies))]
    n_files = {}
    for st, sr in zip(se.StudyInstanceUID, se.SeriesInstanceUID):
        d = f"{root}/{st}/{sr}"
        n_files[sr] = _n_slices(d)
    table = build_study_table(se, n_files)
    dl = torch.utils.data.DataLoader(DicomStudyDataset(studies, table, root, k), batch_size=bs,
                                     num_workers=workers, pin_memory=device.startswith("cuda"))
    w = family_weights(ckpts)
    models = load_models(ckpts, device)
    wt = torch.tensor(w, dtype=torch.float32).view(-1, 1, 1)
    out = []
    for x, mask, pos in dl:
        x, mask, pos = x.to(device, non_blocking=True), mask.to(device), pos.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            p = torch.stack([m(x, mask, pos, res=res).float().sigmoid() for m, res in models])
        out.append((p.cpu() * wt).sum(0))
    return pd.DataFrame(torch.cat(out).numpy(), index=pd.Index(studies, name="StudyInstanceUID"), columns=LABELS)


if __name__ == "__main__":
    data_dir, split, out_csv, ckpts = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    run(data_dir, split, ckpts).to_csv(out_csv)
    print("wrote", out_csv, "| family weights", [round(v, 4) for v in family_weights(ckpts)], flush=True)
