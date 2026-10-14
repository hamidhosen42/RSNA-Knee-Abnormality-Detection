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
  * our final cell appended: W_PARENT x rank(that notebook's submission) + W_READER x rank(students), copied verbatim
    from the scored notebook given by --students (H957x5: 0.8 / 0.2; H957x5-w40, the team best: 0.6 / 0.4).
--flip-r7 (variant b) also switches on heliosli's R7 flip TTA (manifest flip_tta, the worker's --flip-tta; MCL keeps the
plain pass), as jingshuifu/rsna-knee-flip-tta did (0.956, runs fine; weaker evidence).
--extra-readers (H959 only) adds new students from their OWN dataset (the 5-checkpoint dataset is never re-versioned,
so H957x5 / H958 / H958b stay re-pushable), asserts both counts, and puts a 9-hour time guard on the reader run.
Every other block is copied verbatim from a notebook that already scored; nothing is fitted on labels or on the test set.

    python build_h958.py [--flip-r7]                                              -> submission_h958 / submission_h958b
    python build_h958.py --flip-r7 --students submission_h957x5w40/rsna-knee-h957x5w40.ipynb   -> submission_h958bw40
    python build_h958.py --flip-r7 --extra-readers hosen42/rsna-knee-student-readers-e75 --n-extra N --out submission_h959b
An existing output that would change is never overwritten without --overwrite.
"""
import argparse
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
READERS = "hosen42/rsna-knee-student-readers"   # the 5 LB-validated students (private, hosen42); never re-versioned
SCORED = {"submission_h957x5": "H957x5", "submission_h957x5w30": "H957x5-w30", "submission_h957x5w40": "H957x5-w40"}
ap = argparse.ArgumentParser()
ap.add_argument("--parent", default="public_notebooks/0.956_7v_varshithakolupuri__rsna-0-956-one-free-flip-68-ranks-up.ipynb")
ap.add_argument("--r7", default="public_notebooks/0.956_heliosli__0-957-latest-dataset-score-in-one-click.ipynb")
ap.add_argument("--students", default="submission_h957x5/rsna-knee-h957x5.ipynb",
                help="scored H957x5-family notebook: its reader cells (7-11) and final cell (12) are copied verbatim")
ap.add_argument("--flip-r7", action="store_true")
ap.add_argument("--owner", default="hosen42", help="kernel owner (the readers dataset stays hosen42's)")
ap.add_argument("--extra-readers", default=None, help="H959: private dataset owner/slug holding the new students only")
ap.add_argument("--n-extra", type=int, default=0, help="H959: checkpoints expected in --extra-readers")
ap.add_argument("--out", default=None)
ap.add_argument("--overwrite", action="store_true")
a = ap.parse_args()
if bool(a.extra_readers) != (a.n_extra > 0):
    ap.error("--extra-readers and --n-extra > 0 go together")
if a.extra_readers and not a.out:
    ap.error("an H959 build needs an explicit --out (e.g. submission_h959b)")

src = lambda c: "".join(c["source"])
P = json.load(open(os.path.join(ROOT, a.parent)))
pc = P["cells"]
r7 = [src(c) for c in json.load(open(os.path.join(ROOT, a.r7)))["cells"] if c["cell_type"] == "code"]
st_name = os.path.basename(os.path.dirname(os.path.join(ROOT, a.students)))
assert st_name in SCORED, ("--students must be a scored H957x5-family notebook", st_name)
st = json.load(open(os.path.join(ROOT, a.students)))["cells"]

# --- the parent must be exactly what we analysed
assert len(pc) == 13
for i, j in ((3, 0), (4, 1), (6, 2), (7, 3)):          # Raptor 384 write + run, R7 setup + run: heliosli's cells verbatim
    assert src(pc[i]) == r7[j], (i, j)
assert src(pc[9]).startswith("%%writefile /tmp/raptor224_run.py") and "c08_r224_swa.pt" in src(pc[9])
assert src(pc[10]).startswith("# Stage 2b [s5]") and "R224_OK" in src(pc[10])
assert "A224 = 0.35" in src(pc[12]) and "sub.to_csv(WORK / 'submission.csv', index=False)" in src(pc[12])
# --- our parts: the scored notebook's reader cells (7-11) and final cell (12), verbatim
assert len(st) == 13
reader, final = st[7:12], st[12]
assert src(reader[0]).startswith("os.makedirs('/kaggle/working/own_src'")
assert "cnxt_pl*_full.pt" in src(reader[4]) and "assert len(_ck) == 5" in src(reader[4])
assert src(final).startswith("# [h957] final:")
_w = re.findall(r"^W_PARENT, W_READER = ([0-9.]+), ([0-9.]+)$", src(final), flags=re.M)
assert len(_w) == 1, _w
WP, WR = (float(x) for x in _w[0])
assert abs(WP + WR - 1) < 1e-9, (WP, WR)
n_ckpt = 5 + a.n_extra

if a.extra_readers:   # H959: the same reader cell + the new dataset (own glob, own count) + a 9-hour time guard
    xslug = a.extra_readers.split("/")[-1]
    assert a.extra_readers.count("/") == 1 and xslug != READERS.split("/")[1] and re.fullmatch(r"[a-z0-9-]+", xslug), a.extra_readers
    reader = [json.loads(json.dumps(c)) for c in reader]
    s = src(reader[4])
    old_ck = 'assert len(_ck) == 5, _ck; print("reader ckpts", _ck, flush=True)\n'
    new_ck = ('assert len(_ck) == 5, _ck\n'
              f'# [h959] + the new students, from their own dataset {a.extra_readers} (the 5-checkpoint dataset above\n'
              '# is never re-versioned); the two known mount layouts first, the slow recursive walk only as a fallback\n'
              f'_ck_new = sorted(glob.glob("/kaggle/input/datasets/*/{xslug}/cnxt_pl*_full.pt") +\n'
              f'                 glob.glob("/kaggle/input/{xslug}/cnxt_pl*_full.pt"))\n'
              f'if not _ck_new:\n'
              f'    _ck_new = sorted(glob.glob("/kaggle/input/**/{xslug}/cnxt_pl*_full.pt", recursive=True))\n'
              f'assert len(_ck_new) == {a.n_extra}, _ck_new\n'
              '_ck = _ck + _ck_new; print("reader ckpts", _ck, flush=True)\n')
    old_run = ('rc = subprocess.run([sys.executable, "/kaggle/working/own_src/infer.py", _data, "test", "/kaggle/working/reader_sub.csv"] + _ck,\n'
               '                    env=_env).returncode\n')
    new_run = ('try:   # [h959] time guard: the readers never push the session past 9 h; a timeout only drops the student leg\n'
               '    rc = subprocess.run([sys.executable, "/kaggle/working/own_src/infer.py", _data, "test", "/kaggle/working/reader_sub.csv"] + _ck,\n'
               '                        env=_env, timeout=max(60, 9 * 3600 - (time.time() - NB_T0) - 20 * 60)).returncode\n'
               'except subprocess.TimeoutExpired:\n'
               '    rc = -9\n'
               '    print("READER_TIMEOUT after %.0f s - the final cell keeps the notebook\'s own blend" % (time.time() - _t0), flush=True)\n')
    assert s.count(old_ck) == 1 and s.count(old_run) == 1
    s = s.replace(old_ck, new_ck).replace(old_run, new_run)
    reader[4]["source"] = s.splitlines(keepends=True)

cells = [json.loads(json.dumps(c)) for c in pc]
if a.flip_r7:
    s = src(cells[6])
    old = "assert manifest['bundle_id'] == BUNDLE_ID, (manifest['bundle_id'], BUNDLE_ID)\n"
    assert s.count(old) == 1 and s.count("if manifest.get('flip_tta'):") == 1
    s = s.replace(old, old + "manifest['flip_tta'] = True   # [h958b] R7 flip TTA: every fold also scores the mirrored windows "
                             "(heliosli's v4c switch; MCL keeps the plain pass); weight files and their sha256 checks unchanged\n")
    cells[6]["source"] = s.splitlines(keepends=True)
    compile(s, "r7_setup", "exec")

default_out = "submission_h958" + ("b" if a.flip_r7 else "") + ("" if (WP, WR) == (0.8, 0.2) else "w%d" % round(WR * 100))
out_name = a.out or default_out
base = os.path.basename(os.path.normpath(out_name))
assert base.startswith("submission_"), out_name
md = lambda t: {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
_label = base.replace("submission_", "").upper()
_who = SCORED[st_name]
_students = "%d Mirror students" % 5 if not a.n_extra else "5 Mirror students + %d ensemble-teacher students" % a.n_extra
title = ("# RSNA Knee %s: R7 + Raptor (384 + 224 with mirror) + our %s\n\n"
         "Built from varshithakolupuri's public *one free flip* notebook (heliosli's R7 + Raptor, plus nartaa's 224-px "
         "checkpoint with the anatomical mirror view, 35%% of the Raptor slot), with our Mirror-student readers "
         "added exactly as in our %s (LB 0.957): they run right after the R7 stage, and the last cell blends "
         "%g x rank(the notebook's submission) + %g x rank(students). Weights fixed before running.%s%s\n"
         % (_label, _students, _who, WP, WR, " R7 flip TTA is switched on (variant b)." if a.flip_r7 else "",
            (" The %d new students come from their own dataset (%s); the reader run has a 9-hour time guard."
             % (a.n_extra, a.extra_readers)) if a.n_extra else ""))
cells[0] = md(title)
if not a.n_extra:
    reader_md = md("## Stage 2a [h958]: our five Mirror-student readers (%s's reader stage, verbatim)\n" % _who)
else:
    reader_md = md("## Stage 2a [h959]: our five Mirror-student readers + %d new students (%s's reader stage; one more "
                   "dataset and a 9-hour time guard)\n" % (a.n_extra, _who))
nb_cells = cells[:8] + [reader_md] + [json.loads(json.dumps(c)) for c in reader] + cells[8:] + \
           [md("## Stage 4 [h958]: %g x the notebook's blend + %g x our students (%s's final cell, verbatim)\n" % (WP, WR, _who)),
            json.loads(json.dumps(final))]
for c in nb_cells:
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
        compile("\n".join(l for l in src(c).splitlines() if not l.startswith(("%%", "!"))), "cell", "exec")
P["cells"] = nb_cells
out = os.path.join(ROOT, out_name)
slug = "rsna-knee-" + base.replace("submission_", "")
sources = ["heliosli/rsna-knee-compact-r7cnxfsss", "nartaa/rsna-knee-publication-swa-weights-20261007",
           "heliosli/rsna-orthofoundation-fold0-epoch7-20261004", "goodpjw2008/rsna-knee-2-5d-convnext-reader", READERS]
if a.extra_readers:
    sources.append(a.extra_readers)
meta = {"id": f"{a.owner}/{slug}", "title": slug, "code_file": slug + ".ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
        "docker_image_pinning_type": "original", "dataset_sources": sources,
        "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
files = {slug + ".ipynb": json.dumps(P, indent=1), "kernel-metadata.json": json.dumps(meta, indent=2)}
changed = [f for f, t in files.items() if os.path.exists(os.path.join(out, f)) and open(os.path.join(out, f)).read() != t]
if changed and not a.overwrite:
    raise SystemExit(f"refusing to overwrite {out_name}/{changed} (it would change); pass --overwrite or another --out")
os.makedirs(out, exist_ok=True)
for f, t in files.items():
    open(os.path.join(out, f), "w").write(t)
print("wrote", out_name, "| cells", len(nb_cells), "| flip_r7", a.flip_r7, "| final %g/%g from %s" % (WP, WR, _who),
      "| checkpoints", n_ckpt, "| changed" if changed else "| unchanged or new")
if a.owner != READERS.split("/")[0]:
    print("NOTE: kernel owner", a.owner, "needs read access to the private", READERS, "dataset")
