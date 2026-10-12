"""Leg-alone submission notebooks for the R2 pseudo-label A/B (esfersami50/rsna-knee-gpjw-pseudo).

Each notebook runs exactly one full-data reader checkpoint - cnxt_plA_full.pt (v0 labels, control) or
cnxt_plB_full.pt (0.5 v0 + 0.5 OOF targets) - through goodpjw2008's DICOM inference code (the three source cells of
E20's Part B, verbatim) and writes its probabilities as submission.csv.  Nothing else differs between A and B.

    python make_pl_submission.py [--owner=esfersami50] [--train-version=1]  ->  submission_plA/, submission_plB/
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OWNER = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--owner=")), "esfersami50")
TRAIN_VERSION = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--train-version=")), "1")
TRAIN_KERNEL = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--train-kernel=")), "esfersami50/rsna-knee-gpjw-pseudo")
ARMS = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--arms=")), "A,B").split(",")
WHAT = {"A": "v0 labels (control)", "B": "0.5 v0 + 0.5 out-of-fold reader targets (pseudo-label)",
        "C": "0.5 v0 + 0.5 Mirror teacher targets (student of the Mirror)",
        "B2": "B targets with the convnext_tiny.fb_in22k_ft_in1k backbone",
        "C2": "C recipe with seed 1 (0.5 v0 + 0.5 Mirror teacher)",
        "C3": "0.25 v0 + 0.75 Mirror teacher targets"}

e20 = json.load(open(os.path.join(ROOT, "submission_e20", "rsna-knee-e20.ipynb")))["cells"]
e20_meta = json.load(open(os.path.join(ROOT, "submission_e20", "kernel-metadata.json")))
src = lambda c: "".join(c["source"])
assert src(e20[34]).startswith("import os") and "own_src" in src(e20[34])
for i, name in ((35, "preprocess.py"), (36, "knee.py"), (37, "infer.py")):
    assert src(e20[i]).startswith(f"%%writefile /kaggle/working/own_src/{name}"), i


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.splitlines(keepends=True)}


RUN = r'''# Run one full-data reader checkpoint on the test DICOMs and write its probabilities as submission.csv.
import glob, os, subprocess, sys, time
import numpy as np
import pandas as pd
ARM = "__ARM__"
CKPT_NAME = f"cnxt_pl{ARM}_full.pt"


def find(name):
    for pat in ["/kaggle/input/*/", "/kaggle/input/*/*/", "/kaggle/input/*/*/*/", "/kaggle/input/*/*/*/*/"]:
        for d in sorted(glob.glob(pat)):
            if os.path.exists(os.path.join(d, name)):
                return d.rstrip("/")
    raise FileNotFoundError(name)


assets, data = find("cnxt_v0_fold0.pt"), find("test_series.csv")    # goodpjw2008 dataset ships the DICOM decoders
ckpts = sorted(glob.glob(f"/kaggle/input/**/{CKPT_NAME}", recursive=True))   # unique per arm
assert len(ckpts) == 1, f"expected one {CKPT_NAME}, found {ckpts}"
import torch
ck = torch.load(ckpts[0], map_location="cpu", weights_only=False)
print("checkpoint", ckpts[0], "| arm", ck["args"].get("arm"), "| alpha", ck["args"].get("alpha"),
      "| epoch", ck.get("epoch"), "| backbone", ck["args"]["backbone"], "| res", ck["args"]["res"], flush=True)
assert ck["args"].get("arm") == ARM and ck.get("epoch") == ck["args"]["epochs"] - 1, "not the finished checkpoint of this arm"
del ck
target = "/kaggle/working/_own_env"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", target,
                "--find-links", assets, "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=False)
env = dict(os.environ, PYTHONPATH=target + os.pathsep + os.environ.get("PYTHONPATH", ""))
t0 = time.time()
subprocess.run([sys.executable, "/kaggle/working/own_src/infer.py", data, "test", "/kaggle/working/_reader.csv", ckpts[0]],
               check=True, env=env)
print(f"READER_DONE {time.time() - t0:.0f}s", flush=True)

test_ids = pd.read_csv(f"{data}/test.csv")["StudyInstanceUID"].astype(str).tolist()
cols = list(pd.read_csv(f"{data}/sample_submission.csv", nrows=1).columns)
pred = pd.read_csv("/kaggle/working/_reader.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
assert sorted(pred.index) == sorted(test_ids), "rows differ from test.csv"
sub = pred.loc[test_ids, cols[1:]].reset_index()
sub.columns = cols
assert np.isfinite(sub[cols[1:]].values).all() and ((sub[cols[1:]].values >= 0) & (sub[cols[1:]].values <= 1)).all()
sub.to_csv("/kaggle/working/submission.csv", index=False)
os.remove("/kaggle/working/_reader.csv")
print("NOTEBOOK_COMPLETE", sub.shape, flush=True)
display(sub.head())
'''

for arm in ARMS:
    what = WHAT[arm]
    cells = [md(f"# RSNA Knee R2 arm {arm}: full-data 2.5D ConvNeXt reader, {what}\n\nOne checkpoint "
                f"(`cnxt_pl{arm}_full.pt`, {TRAIN_KERNEL}) through goodpjw2008's inference code; "
                "leg-alone A/B for the pseudo-label experiment.\n")]
    cells += [json.loads(json.dumps(e20[i])) for i in (34, 35, 36, 37)]
    cells += [code(RUN.replace("__ARM__", arm))]
    for c in cells:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    compile(RUN.replace("__ARM__", arm).replace("display(", "print("), "run", "exec")
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out = os.path.join(ROOT, f"submission_pl{arm}")
    os.makedirs(out, exist_ok=True)
    slug = f"rsna-knee-pl{arm.lower()}"
    json.dump(nb, open(os.path.join(out, slug + ".ipynb"), "w"), indent=1)
    meta = {"id": f"{OWNER}/{slug}", "title": slug, "code_file": slug + ".ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
            "machine_shape": "NvidiaTeslaT4", "docker_image": e20_meta["docker_image"], "docker_image_pinning_type": "original",
            "dataset_sources": ["goodpjw2008/rsna-knee-2-5d-convnext-reader/1"],
            "competition_sources": ["rsna-knee-abnormality-detection"],
            "kernel_sources": [f"{TRAIN_KERNEL}/{TRAIN_VERSION}"], "model_sources": []}
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
    print("wrote", out)
