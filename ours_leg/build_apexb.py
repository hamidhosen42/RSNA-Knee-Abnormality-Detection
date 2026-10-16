"""APEX-B: sujanmajhisuzan/rsna-knee-apex-grandmaster-stack v1 (public 0.950 = nartaa's Mirror Raptor + goodpjw2008's
3-fold reader, per-target rank fusion; the team's best submission 56928028) with ONE change: the reader checkpoints,
the author's 3 v0-label folds -> our full-data pseudo-label reader B (esfersami50/rsna-knee-gpjw-pseudo v1, leg alone
0.939 vs 0.933 on v0 labels; the public 3-fold reader alone is about 0.929).  Fusion weights, Mirror stage, infer.py
(k=16) unchanged.  Rule note: Mirror is OAI-trained (see audit/13) - a public-LB candidate, not a rule-clean final.
Pre-registered: keep if LB > 0.950, neutral at 0.950, reject below.

    python build_apexb.py [--src public_notebooks/0.950_129v_...ipynb] [--out submission_apexb] [--owner esfersami50]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import NB, inp, out_dir, write_json  # noqa: E402  (one place for every path: ours_leg/paths.py)
ap = argparse.ArgumentParser()
ap.add_argument("--src", default="public_notebooks/0.950_129v_sujanmajhisuzan__rsna-knee-apex-grandmaster-stack.ipynb")
ap.add_argument("--out", default="submission_apexb")
ap.add_argument("--owner", default="esfersami50")
a = ap.parse_args()
nb = json.load(open(inp(a.src)))
cells = nb["cells"]
s = "".join(cells[9]["source"])
OLD = '''    _o_ck = sorted(_o_glob.glob(_o_assets + "/cnxt_v0_fold*.pt"))
'''
NEW = '''    # [APEX-B] the reader is our full-data pseudo-label reader B (esfersami50/rsna-knee-gpjw-pseudo v1) instead of the
    # author's 3 v0-label folds; the goodpjw2008 dataset still supplies the DICOM decoder wheels (_o_assets)
    _o_ck = sorted(_o_glob.glob("/kaggle/input/**/rsna-knee-gpjw-pseudo/cnxt_plB_full.pt", recursive=True))
    assert len(_o_ck) == 1, f"expected the pseudo-label reader B checkpoint, found {_o_ck}"
'''
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
cells[9]["source"] = s.splitlines(keepends=True)
compile(s, "cell9", "exec")
cells[0]["source"] = ("".join(cells[0]["source"]) + "\n\n**APEX-B (ours):** identical to the public Apex Grandmaster Stack "
                      "v1 except that the ConvNeXt reader is our pseudo-label reader B.").splitlines(keepends=True)
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
out = out_dir(a.out)
os.makedirs(out, exist_ok=True)
slug = "rsna-knee-" + a.out.replace("submission_", "")
write_json(os.path.join(out, slug + ".ipynb"), nb, indent=1)
meta = {"id": f"{a.owner}/{slug}", "title": slug, "code_file": slug + ".ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
        "docker_image_pinning_type": "original",
        "dataset_sources": ["goodpjw2008/rsna-knee-2-5d-convnext-reader", "nartaa/rsna-knee-publication-swa-weights-20261007"],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "kernel_sources": ["esfersami50/rsna-knee-gpjw-pseudo/1"], "model_sources": []}
write_json(os.path.join(out, "kernel-metadata.json"), meta, indent=2)
print("wrote", out)
