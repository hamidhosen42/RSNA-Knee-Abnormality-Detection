"""Test 3: infer_tta.run_tta end to end on synthetic DICOM studies with random-weight KneeNet checkpoints (CPU).
  (a) its plain pass == infer.run (the H964 student code path) on the same studies
  (b) its mirror pass on study A == infer.run on study A_m = the same knee physically mirrored and re-acquired
      (mirrored geometry in 3 scanner conventions); fixed-window mode isolates the geometry, real mode shows the
      +-1 grey-level effect of the strided percentile window
  (c) a study with one series lacking ImageOrientationPatient: mirror output == plain output (fallback)
  (d) the mirror pass really differs from the plain pass (random filters are not mirror-symmetric)

    uv run --no-project --with numpy --with pydicom --with opencv-python-headless --with pandas --with torch --with timm \
        python -I test_model.py <runtime_src dir> <scratch dir>
"""
import json
import os
import shutil
import sys

import numpy as np

src_dir, work = sys.argv[1], sys.argv[2]
sys.path.insert(0, src_dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

import infer  # noqa: E402
import infer_tta  # noqa: E402
import preprocess  # noqa: E402
from knee import LABELS, KneeNet  # noqa: E402
from synth import acquisition, mirrored_variant, phantom, rot, write_series  # noqa: E402

torch.set_num_threads(8)
torch.Tensor.cuda = lambda self, *a, **k: self          # infer.run hard-codes .cuda(): run it on the CPU here
infer.load_models.__defaults__ = ("cpu",)

DATA = os.path.join(work, "data")
SERIES = [  # (series id, plane, fat_sup, oblique axis/deg or None, convention, mirrored-acquisition variant)
    ("sagfs", "Sagittal", 1, ("z", 15), "typical", "v0"),
    ("sagt1", "Sagittal", 0, ("z", -8), "transposed", "v1"),
    ("corfs", "Coronal", 1, ("z", 12), "typical", "v2"),
    ("cort1", "Coronal", 0, None, "neg_r", "v0"),
    ("axfs", "Axial", 1, ("x", 10), "typical", "v1"),
]
BASE = {"Sagittal": ([0, 1, 0], [0, 0, -1]), "Coronal": ([1, 0, 0], [0, 0, -1]), "Axial": ([1, 0, 0], [0, 1, 0])}


def build():
    shutil.rmtree(DATA, ignore_errors=True)
    rows = []
    for study, mirrored_study, drop_iop in (("A", "A_m", None), ("B", None, "cort1")):
        for sid, plane, fs, obl, conv, var in SERIES:
            r, c = (np.array(v, float) for v in BASE[plane])
            if obl:
                R = rot(*obl)
                r, c = R @ r, R @ c
            if conv == "transposed":
                r, c = c, r
            elif conv == "neg_r":
                r = -r
            n = {"Sagittal": 26, "Coronal": 22, "Axial": 30}[plane]
            centre = (4.0, -2.0, 3.0) if study == "A" else (-6.0, 5.0, -4.0)
            acq = acquisition(r, c, 200, 200, 0.8, 0.8, n, 3.5, centre=centre)
            arrays = [np.round(phantom(acq["P"][t])) for t in range(n)]
            write_series(f"{DATA}/test_series/{study}/{study}_{sid}", arrays, acq["iop"], acq["ipps"], acq["ps"],
                         drop_iop=(sid == drop_iop))
            rows.append((study, f"{study}_{sid}", fs, plane))
            if mirrored_study:
                arr, iop, ipps, ps = mirrored_variant(acq, arrays, var)
                write_series(f"{DATA}/test_series/{mirrored_study}/{mirrored_study}_{sid}", arr, iop, ipps, ps)
                rows.append((mirrored_study, f"{mirrored_study}_{sid}", fs, plane))
    se = pd.DataFrame(rows, columns=["StudyInstanceUID", "SeriesInstanceUID", "Fat_Suppression", "Anatomical_Plane"])
    se.to_csv(f"{DATA}/test_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": ["A", "A_m", "B"]}).to_csv(f"{DATA}/test.csv", index=False)
    ck = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        m = KneeNet("convnext_tiny", pretrained=False)
        with torch.no_grad():   # larger head weights so that the outputs spread across studies
            m.cls_w.mul_(50.0)
        p = os.path.join(work, f"rand{seed}.pt")
        torch.save({"model": m.state_dict(), "args": {"backbone": "convnext_tiny", "res": 256}}, p)
        ck.append(p)
    return ck


class _FixedPct:
    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def percentile(a, q):
        return np.array([0.0, 3000.0])


def main():
    ck = build()
    out = {}
    for mode in ("fixed_window", "real"):
        preprocess.np = _FixedPct() if mode == "fixed_window" else np
        ref = infer.run(DATA, "test", ck, workers=0)                                   # H964 code path
        p, q, info = infer_tta.run_tta(DATA, "test", ck, workers=0, devices=["cpu"])
        preprocess.np = np
        d_plain = float((p - ref).abs().values.max())
        d_mirror_vs_physical = (q.loc["A"] - ref.loc["A_m"]).abs()
        d_mirror_vs_plain_A = (q.loc["A"] - p.loc["A"]).abs()
        out[mode] = {
            "a_plain_vs_infer_run_maxabs": d_plain,
            "b_mirror(A)_vs_plain(A_m)_maxabs": float(d_mirror_vs_physical.max()),
            "b_mirror(A_m)_vs_plain(A)_maxabs": float((q.loc["A_m"] - ref.loc["A"]).abs().max()),
            "c_B_mirror_equals_plain_maxabs": float((q.loc["B"] - p.loc["B"]).abs().max()),
            "d_mirror(A)_vs_plain(A)_maxabs": float(d_mirror_vs_plain_A.max()),
            "output_spread_over_studies_maxabs": float((ref.max() - ref.min()).max()),
            "tta_info_mirrored": info["mirrored"],
        }
    # (e) the same with 2 DataLoader workers (the Kaggle setting uses 4): the dataset must pickle and give the same result
    p2, q2, _ = infer_tta.run_tta(DATA, "test", ck, workers=2, devices=["cpu"])
    p0, q0, _ = infer_tta.run_tta(DATA, "test", ck, workers=0, devices=["cpu"])
    out["e_workers2_vs_workers0_maxabs"] = float(max((p2 - p0).abs().values.max(), (q2 - q0).abs().values.max()))
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(work, "test3_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
