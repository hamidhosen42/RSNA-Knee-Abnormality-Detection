"""H958: every component with leaderboard evidence, in one notebook, aiming above 0.957.

Evidence (public LB; inside a block of tied displayed scores the rank follows the unrounded score):
  * base = heliosli's R7 + Raptor notebook (Mirror384 0.5 + R7 0.5)                         -> 0.956
  * + nartaa's unused 224-px checkpoint in the Raptor slot (0.65 x 384 + 0.35 x 224)          -> 0.956, rank 233 -> 156
  * + nartaa's anatomical mirror view for that 224 model (varshithakolupuri "one free flip")  -> 0.956, rank 357 -> 289
  * + our 5 Mirror-student readers at 0.2 on top of the base (H957 / H957x5)                  -> 0.957 (from 0.956)
The two gains act on different parts (the Raptor slot vs a new model on top), so they can add up.

Notebook = varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up v1 (R7 base + 224 stage with mirror + blend), with
  * our reader cells inserted right after the R7 stage (before the 224 stage, whose 9-hour guard then sees the time we
    used, so a slow run drops the 224 stage, never our validated students), and
  * our final cell appended: 0.8 x rank(that notebook's submission) + 0.2 x rank(students), the H957x5 cell verbatim.
--flip-r7 (variant b) also switches on heliosli's R7 flip TTA (manifest flip_tta, the worker's --flip-tta; MCL keeps the
plain pass), as jingshuifu/rsna-knee-flip-tta did (0.956, runs fine; weaker evidence).
Every block is copied verbatim from a notebook that already scored; nothing is fitted on labels or on the test set.

    python build_h958.py [--flip-r7] [--owner hosen42]
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--parent", default="public_notebooks/0.956_7v_varshithakolupuri__rsna-0-956-one-free-flip-68-ranks-up.ipynb")
ap.add_argument("--r7", default="public_notebooks/0.956_heliosli__0-957-latest-dataset-score-in-one-click.ipynb")
ap.add_argument("--students", default="submission_h957x5/rsna-knee-h957x5.ipynb")
ap.add_argument("--flip-r7", action="store_true")
ap.add_argument("--owner", default="hosen42")
ap.add_argument("--out", default=None)
ap.add_argument("--n-ckpt", type=int, default=5,
                help="student checkpoints expected in the readers dataset (H958: 5; H959: 7 = + E75t, E75s)")
a = ap.parse_args()
out_name = a.out or ("submission_h958b" if a.flip_r7 else "submission_h958")

src = lambda c: "".join(c["source"])
P = json.load(open(os.path.join(ROOT, a.parent)))
pc = P["cells"]
r7 = [src(c) for c in json.load(open(os.path.join(ROOT, a.r7)))["cells"] if c["cell_type"] == "code"]
st = json.load(open(os.path.join(ROOT, a.students)))["cells"]

# --- the parent must be exactly what we analysed
assert len(pc) == 13
for i, j in ((3, 0), (4, 1), (6, 2), (7, 3)):          # Raptor 384 write + run, R7 setup + run: heliosli's cells verbatim
    assert src(pc[i]) == r7[j], (i, j)
assert src(pc[9]).startswith("%%writefile /tmp/raptor224_run.py") and "c08_r224_swa.pt" in src(pc[9])
assert src(pc[10]).startswith("# Stage 2b [s5]") and "R224_OK" in src(pc[10])
assert "A224 = 0.35" in src(pc[12]) and "sub.to_csv(WORK / 'submission.csv', index=False)" in src(pc[12])
# --- our parts: H957x5's reader cells (7-11) and final cell (12), verbatim (LB 0.957)
reader, final = st[7:12], st[12]
assert src(reader[0]).startswith("os.makedirs('/kaggle/working/own_src'")
assert "cnxt_pl*_full.pt" in src(reader[4]) and "assert len(_ck) == 5" in src(reader[4])
if a.n_ckpt != 5:   # H959: the same reader cell, more checkpoints in the (re-versioned) readers dataset
    reader = [json.loads(json.dumps(c)) for c in reader]
    _r = src(reader[4]).replace("assert len(_ck) == 5", f"assert len(_ck) == {a.n_ckpt}")
    reader[4]["source"] = _r.splitlines(keepends=True)
assert src(final).startswith("# [h957] final:") and "W_PARENT, W_READER = 0.8, 0.2" in src(final)

cells = [json.loads(json.dumps(c)) for c in pc]
if a.flip_r7:
    s = src(cells[6])
    old = "assert manifest['bundle_id'] == BUNDLE_ID, (manifest['bundle_id'], BUNDLE_ID)\n"
    assert s.count(old) == 1 and s.count("if manifest.get('flip_tta'):") == 1
    s = s.replace(old, old + "manifest['flip_tta'] = True   # [h958b] R7 flip TTA: every fold also scores the mirrored windows "
                             "(heliosli's v4c switch; MCL keeps the plain pass); weight files and their sha256 checks unchanged\n")
    cells[6]["source"] = s.splitlines(keepends=True)
    compile(s, "r7_setup", "exec")

md = lambda t: {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
_label = out_name.replace("submission_", "").upper()
title = ("# RSNA Knee %s: R7 + Raptor (384 + 224 with mirror) + our %d Mirror students\n\n"
         "Built from varshithakolupuri's public *one free flip* notebook (heliosli's R7 + Raptor, plus nartaa's 224-px "
         "checkpoint with the anatomical mirror view, 35%% of the Raptor slot), with our Mirror-student readers "
         "added exactly as in our H957x5 (LB 0.957): they run right after the R7 stage, and the last cell blends "
         "0.8 x rank(the notebook's submission) + 0.2 x rank(students). Weights fixed before running.%s\n"
         % (_label, a.n_ckpt, " R7 flip TTA is switched on (variant b)." if a.flip_r7 else ""))
cells[0] = md(title)
reader_md = md("## Stage 2a [h958]: our five Mirror-student readers (H957x5's reader stage, verbatim)\n")
nb_cells = cells[:8] + [reader_md] + [json.loads(json.dumps(c)) for c in reader] + cells[8:] + \
           [md("## Stage 4 [h958]: 0.8 x the notebook's blend + 0.2 x our students (H957x5's final cell, verbatim)\n"),
            json.loads(json.dumps(final))]
for c in nb_cells:
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
        compile("\n".join(l for l in src(c).splitlines() if not l.startswith(("%%", "!"))), "cell", "exec")
P["cells"] = nb_cells
out = os.path.join(ROOT, out_name)
os.makedirs(out, exist_ok=True)
slug = "rsna-knee-" + out_name.replace("submission_", "")
json.dump(P, open(os.path.join(out, slug + ".ipynb"), "w"), indent=1)
meta = {"id": f"{a.owner}/{slug}", "title": slug, "code_file": slug + ".ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
        "docker_image_pinning_type": "original",
        "dataset_sources": ["heliosli/rsna-knee-compact-r7cnxfsss", "nartaa/rsna-knee-publication-swa-weights-20261007",
                            "heliosli/rsna-orthofoundation-fold0-epoch7-20261004", "goodpjw2008/rsna-knee-2-5d-convnext-reader",
                            f"{a.owner}/rsna-knee-student-readers"],
        "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
print("wrote", out, "| cells", len(nb_cells), "| flip_r7", a.flip_r7)
