"""Build the E11-alone calibration notebook: our EfficientNetV2-S leg (folds 0+1 SWA) with no
public pipeline and no blend, so its public LB is the leg's own score. That number calibrates
our OOF (0.8721) against the leaderboard before any further GPU is spent.

Unlike the blend notebook, a failed leg raises instead of falling back: a fallback here would
submit something that is not the measurement.

    Runs on CPU (fp32): the weekly GPU quota was exhausted on 2026-10-02; ~1300 studies fit in 9 h.

    python build_e11_alone.py   ->  submission/e11-alone/rsna-knee-e11-alone.ipynb + kernel-metadata.json
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "submission" / "e11-alone"

RUN = r'''import glob, os, subprocess, sys, time
import pandas as pd

_markers = []
for _d in range(4):
    _markers += [m for m in sorted(glob.glob(os.path.join("/kaggle/input", *(["*"] * _d), "arms", "*", "ours_leg_arm.json")))
                 if "/competitions/" not in m]
_roots = sorted({os.path.dirname(os.path.dirname(os.path.dirname(m))) for m in _markers})
_ckpts = sorted({p for r in _roots for p in glob.glob(f"{r}/arms/*/weights/*_swa.pt")})
print("arms roots", _roots, "| checkpoints", [os.path.basename(p) for p in _ckpts], flush=True)
assert len(_ckpts) == 2, "expected the two E11 SWA checkpoints (folds 0 and 1)"
_env = {**os.environ, "RSNA_ARMS_DIRS": os.pathsep.join(_roots), "RSNA_CKPT_GLOB": "*_swa.pt",
        "RSNA_OUT": "/kaggle/working", "RSNA_WORKERS": "2", "RSNA_BUDGET_S": str(8 * 3600),
        "RSNA_EXPECTED_WEIGHTS": "2"}
_t0 = time.time()
_rc = subprocess.run([sys.executable, "/kaggle/tmp/ours_leg/infer_kernel.py"], env=_env).returncode
print(f"leg rc={_rc} in {(time.time() - _t0) / 60:.1f} min", flush=True)
assert _rc == 0, f"E11 leg failed (rc={_rc}); no fallback in the calibration notebook"
_sub = pd.read_csv("/kaggle/working/submission.csv", dtype={"StudyInstanceUID": str})
_root = next(r for r in ("/kaggle/input/competitions/rsna-knee-abnormality-detection",
                         "/kaggle/input/rsna-knee-abnormality-detection") if os.path.exists(r + "/test.csv"))
_test = pd.read_csv(_root + "/test.csv", dtype={"StudyInstanceUID": str})
_ss = pd.read_csv(_root + "/sample_submission.csv", nrows=1)
assert list(_sub.columns) == list(_ss.columns) and set(_sub.StudyInstanceUID) == set(_test.StudyInstanceUID)
assert _sub.drop(columns="StudyInstanceUID").notna().all().all()
print("submission.csv verified:", _sub.shape, flush=True)
'''


def cell(src, kind="code"):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}},
      "cells": [cell("# E11 alone (calibration)\n\nOur EfficientNetV2-S leg, folds 0+1 SWA, no public "
                     "pipeline and no blend. Measures the leg's own public LB.", "markdown"),
                cell("import os\nos.makedirs('/kaggle/tmp/ours_leg', exist_ok=True)"),
                cell("%%writefile /kaggle/tmp/ours_leg/infer_kernel.py\n" + (HERE / "infer_kernel.py").read_text()),
                cell(RUN)]}
meta = {"id": "hosen42/rsna-knee-e11-alone", "title": "rsna-knee-e11-alone", "code_file": "rsna-knee-e11-alone.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": False,
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
        "docker_image_pinning_type": "original", "dataset_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "kernel_sources": ["hosen42/rsna-knee-ours-leg-train-effv2s/3"], "model_sources": []}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "rsna-knee-e11-alone.ipynb").write_text(json.dumps(nb, indent=1))
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
compile(RUN, "run", "exec")
print("wrote", OUT)
