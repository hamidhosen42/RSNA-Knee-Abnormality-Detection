"""E21: E20 (rule-clean 0.945) with its reader leg replaced by the adopted pseudo-label reader B.

Single change against E20: Part B's checkpoint list, 7 checkpoints (author tiny f0-2 + ours tiny f3-4 + ours small f3-4)
-> one, cnxt_plB_full.pt (esfersami50/rsna-knee-gpjw-pseudo v1: full data, targets 0.5 v0 + 0.5 5-fold OOF; leg alone
LB 0.939 vs 0.933 for the same recipe on v0 labels).  The per-target routing weights, the stack and everything else are
unchanged.  Pre-registered: keep if LB > 0.945 (E18-E20), neutral at 0.945, reject below.

    python build_e21_plreader.py [--parent submission_e20] [--out submission_e21] [--owner esfersami50]
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--parent", default="submission_e20")
ap.add_argument("--out", default="submission_e21")
ap.add_argument("--owner", default="esfersami50")
a = ap.parse_args()
par = os.path.join(ROOT, a.parent)
nb_name = next(f for f in os.listdir(par) if f.endswith(".ipynb"))
nb = json.load(open(os.path.join(par, nb_name)))
cells = nb["cells"]
s = "".join(cells[38]["source"])
OLD = '''    _o_ck = sorted(_o_glob.glob(_o_assets + "/cnxt_v0_fold*.pt"))
    # E18: + our reader checkpoints trained with the author's code on folds 3-4: convnext_tiny
    # (esfersami50/rsna-knee-gpjw-train-f34 v1) and convnext_small (hosen42/rsna-knee-gpjw-train-f34-small v1)
    _o_ck += sorted(_o_glob.glob("/kaggle/input/**/rsna-knee-gpjw-train-f34/cnxt_v0_fold[34].pt", recursive=True))
    _o_ck += sorted(_o_glob.glob("/kaggle/input/**/rsna-knee-gpjw-train-f34-small/cnxt_small_fold[34].pt", recursive=True))
    assert len(_o_ck) == 7, f"expected 7 reader checkpoints, found {_o_ck}"
'''
NEW = '''    # E21: the reader leg is our full-data pseudo-label reader B (esfersami50/rsna-knee-gpjw-pseudo v1, targets
    # 0.5 v0 + 0.5 5-fold OOF; leg alone LB 0.939) instead of the 7 v0-label fold checkpoints of E18-E20
    _o_ck = sorted(_o_glob.glob("/kaggle/input/**/rsna-knee-gpjw-pseudo/cnxt_plB_full.pt", recursive=True))
    assert len(_o_ck) == 1, f"expected the pseudo-label reader B checkpoint, found {_o_ck}"
'''
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)
s = s.replace('print(f"own reader: target-routed blend applied across {len(_o_ck)} checkpoints',
              'print(f"own reader [E21 pseudo-label B]: target-routed blend applied across {len(_o_ck)} checkpoints')
cells[38]["source"] = s.splitlines(keepends=True)
compile(s, "cell38", "exec")
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None

out = os.path.join(ROOT, a.out)
os.makedirs(out, exist_ok=True)
slug = "rsna-knee-" + a.out.replace("submission_", "")
json.dump(nb, open(os.path.join(out, slug + ".ipynb"), "w"), indent=1)
meta = json.load(open(os.path.join(par, "kernel-metadata.json")))
meta.update(id=f"{a.owner}/{slug}", title=slug, code_file=slug + ".ipynb")
meta["kernel_sources"] = [k for k in meta["kernel_sources"] if "gpjw-train-f34" not in k] + ["esfersami50/rsna-knee-gpjw-pseudo/1"]
json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
print("wrote", out, "| kernels", meta["kernel_sources"])
