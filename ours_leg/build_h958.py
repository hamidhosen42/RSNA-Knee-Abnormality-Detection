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
  * our final cell appended: W_PARENT x rank(that notebook's submission) + W_READER x rank(students), taken from the
    scored notebook given by --students (H957x5: 0.8 / 0.2; H957x5-w40, the team best: 0.6 / 0.4).
--flip-r7 (variant b) also switches on heliosli's R7 flip TTA (manifest flip_tta, the worker's --flip-tta; MCL keeps the
plain pass), as jingshuifu/rsna-knee-flip-tta did (0.956, runs fine; weaker evidence).
--extra-readers (H959 only) adds new students from their OWN dataset (the 5-checkpoint dataset is never re-versioned).

Robustness patches (code review 2026-10-15), each a no-op in a normal run (same files, same numbers) and active only
when something goes wrong:
  R1 reader checkpoints by NAME (other files in the dataset are ignored), known mount layouts first, the slow recursive
     walk over /kaggle/input (the DICOM tree) only as a fallback;
  R2 reader time guard: timeout = 9 h - elapsed - 20 min; a timeout only drops the student leg;
  R3 infer.py: a series stored in fewer than 3 files (e.g. one multi-frame DICOM) is counted by its frames, as in
     training (meta.n = decoded slices), instead of being dropped;
  R4 224 stage: lists the studies it could not score; the blend then gives those studies the 384 rank alone instead of
     a meaningless mid rank for 35 % of the Raptor slot;
  R5 final cell: a study the reader could not score gets its expected student rank given its parent rank (linear fit
     over the scored studies) instead of the parent rank itself, which pushed such studies to the extremes;
  R6 final-cell header states the weights the code really uses (w40: 0.4 was chosen after the LB probes).
Nothing is fitted on labels or on leaderboard feedback.

    python build_h958.py [--flip-r7]                                              -> submission_h958 / submission_h958b
    python build_h958.py --flip-r7 --students submission_h957x5w40/rsna-knee-h957x5w40.ipynb   -> submission_h958bw40
    python build_h958.py --flip-r7 --extra-readers hosen42/rsna-knee-student-readers-e75 \\
        --extra-ckpts cnxt_plE75t_full.pt,cnxt_plE75s_full.pt --out submission_h959b
    another account: --owner foysalemonshanto --out <scratch dir>/submission_h958b (build at push time; no repo copies)
An existing output that would change is never overwritten without --overwrite.
"""
import argparse
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
READERS = "hosen42/rsna-knee-student-readers"   # the 5 LB-validated students (private, hosen42); never re-versioned
READER_CKPTS = ["cnxt_plC2_full.pt", "cnxt_plC3_full.pt", "cnxt_plC_full.pt", "cnxt_plS75_full.pt", "cnxt_plT75s1_full.pt"]
SCORED = {"submission_h957x5": "H957x5", "submission_h957x5w30": "H957x5-w30", "submission_h957x5w40": "H957x5-w40"}
ap = argparse.ArgumentParser()
ap.add_argument("--parent", default="public_notebooks/0.956_7v_varshithakolupuri__rsna-0-956-one-free-flip-68-ranks-up.ipynb")
ap.add_argument("--r7", default="public_notebooks/0.956_heliosli__0-957-latest-dataset-score-in-one-click.ipynb")
ap.add_argument("--students", default="submission_h957x5/rsna-knee-h957x5.ipynb",
                help="scored H957x5-family notebook: its reader cells (7-11) and final cell (12) are the base of ours")
ap.add_argument("--flip-r7", action="store_true")
ap.add_argument("--owner", default="hosen42", help="kernel owner (the readers dataset stays hosen42's)")
ap.add_argument("--extra-readers", default=None, help="H959: private dataset owner/slug holding the new students only")
ap.add_argument("--extra-ckpts", default="", help="H959: comma-separated checkpoint file names in --extra-readers")
ap.add_argument("--out", default=None)
ap.add_argument("--overwrite", action="store_true")
a = ap.parse_args()
extra = [x for x in a.extra_ckpts.split(",") if x]
if bool(a.extra_readers) != bool(extra):
    ap.error("--extra-readers and --extra-ckpts go together")
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
# --- our parts: the scored notebook's reader cells (7-11) and final cell (12)
assert len(st) == 13
reader, final = [json.loads(json.dumps(c)) for c in st[7:12]], json.loads(json.dumps(st[12]))
assert src(reader[0]).startswith("os.makedirs('/kaggle/working/own_src'")
assert src(reader[3]).startswith("%%writefile /kaggle/working/own_src/infer.py")
assert "cnxt_pl*_full.pt" in src(reader[4]) and "assert len(_ck) == 5" in src(reader[4])
assert src(final).startswith("# [h957] final:")
_w = re.findall(r"^W_PARENT, W_READER = ([0-9.]+), ([0-9.]+)$", src(final), flags=re.M)
assert len(_w) == 1, _w
WP, WR = (float(x) for x in _w[0])
assert abs(WP + WR - 1) < 1e-9, (WP, WR)
cells = [json.loads(json.dumps(c)) for c in pc]


def patch(cell, pairs):
    s = src(cell)
    for old, new in pairs:
        assert s.count(old) == 1, (old[:90], s.count(old))
        s = s.replace(old, new)
    cell["source"] = s.splitlines(keepends=True)
    compile("\n".join(l for l in s.splitlines() if not l.startswith(("%%", "!"))), "patched", "exec")


# R1 + R2: reader checkpoints by name, cheap lookup first; time guard on the reader run
_lookup = ('# [h958 R1] checkpoints by NAME: other files in a dataset are ignored, so adding files can never break this\n'
           '# run; the two known mount layouts first, the slow recursive walk over /kaggle/input (the DICOM tree) only as a fallback\n'
           'def _ckpts(slug, names):\n'
           '    def _all(d):\n'
           '        return [os.path.join(d, n) for n in names] if all(os.path.isfile(os.path.join(d, n)) for n in names) else None\n'
           '    for d in sorted(glob.glob("/kaggle/input/datasets/*/" + slug)) + ["/kaggle/input/" + slug]:\n'
           '        if _all(d):\n'
           '            return _all(d)\n'
           '    for p in glob.glob("/kaggle/input/**/" + slug + "/" + names[0], recursive=True):\n'
           '        if _all(os.path.dirname(p)):\n'
           '            return _all(os.path.dirname(p))\n'
           '    raise FileNotFoundError("%s: %s" % (slug, names))\n'
           f'_ck = _ckpts("{READERS.split("/")[1]}", {READER_CKPTS!r})\n')
if extra:
    xslug = a.extra_readers.split("/")[-1]
    assert a.extra_readers.count("/") == 1 and xslug != READERS.split("/")[1] and re.fullmatch(r"[a-z0-9-]+", xslug), a.extra_readers
    assert len(set(extra)) == len(extra) and all(re.fullmatch(r"cnxt_pl\w+_full\.pt", x) for x in extra), extra
    _lookup += f'_ck += _ckpts("{xslug}", {extra!r})   # [h959] the new students, from their own dataset {a.extra_readers}\n'
patch(reader[4], [
    ('_ck = sorted(glob.glob("/kaggle/input/**/rsna-knee-student-readers/cnxt_pl*_full.pt", recursive=True))\n'
     'assert len(_ck) == 5, _ck; print("reader ckpts", _ck, flush=True)\n',
     _lookup + 'print("reader ckpts", _ck, flush=True)\n'),
    ('rc = subprocess.run([sys.executable, "/kaggle/working/own_src/infer.py", _data, "test", "/kaggle/working/reader_sub.csv"] + _ck,\n'
     '                    env=_env).returncode\n',
     'try:   # [h958 R2] time guard: the readers never push the session past 9 h; a timeout only drops the student leg\n'
     '    rc = subprocess.run([sys.executable, "/kaggle/working/own_src/infer.py", _data, "test", "/kaggle/working/reader_sub.csv"] + _ck,\n'
     '                        env=_env, timeout=max(60, 9 * 3600 - (time.time() - NB_T0) - 20 * 60)).returncode\n'
     'except subprocess.TimeoutExpired:\n'
     '    rc = -9\n'
     '    print("READER_TIMEOUT after %.0f s - the final cell keeps the notebook\'s own blend" % (time.time() - _t0), flush=True)\n')])

# R3: infer.py counts a series' slices as training did when the series has fewer than 3 files
patch(reader[3], [
    ('@torch.no_grad()\ndef run(',
     'def _n_slices(d):\n'
     '    """[h958 R3] slice count as in training (decoded frames, capped at MAX_SLICES).  Counting files is the same except\n'
     '    for a series stored in fewer than 3 files (e.g. one multi-frame DICOM): its frames are read from the headers."""\n'
     '    if not os.path.isdir(d):\n'
     '        return 0\n'
     '    files = os.listdir(d)\n'
     '    if len(files) >= 3:\n'
     '        return min(len(files), MAX_SLICES)\n'
     '    import pydicom\n'
     '    n = 0\n'
     '    for fn in files:\n'
     '        try:\n'
     '            n += int(pydicom.dcmread(os.path.join(d, fn), stop_before_pixels=True).get("NumberOfFrames") or 1)\n'
     '        except Exception:\n'
     '            pass\n'
     '    return min(n, MAX_SLICES)\n\n\n'
     '@torch.no_grad()\ndef run('),
    ('        n_files[sr] = min(len(os.listdir(d)), MAX_SLICES) if os.path.isdir(d) else 0\n',
     '        n_files[sr] = _n_slices(d)\n')])

# R4: the 224 stage lists the studies it could not score; the blend gives them the 384 rank alone
patch(cells[9], [
    ('    print(f"[gpu] done: {done_n[0]}/{N} studies on {len(devs)} device(s) | {time.time()-t0:.0f}s", flush=True)\n',
     '    print(f"[gpu] done: {done_n[0]}/{N} studies on {len(devs)} device(s) | {time.time()-t0:.0f}s", flush=True)\n'
     '    _fail = np.stack([(p == 0.5).all(axis=1) for p in arm_probs]).any(axis=0)   # [h958 R4] left at 0.5 = not scored\n'
     '    with open("/kaggle/working/raptor224_failed.txt", "w") as _fh:\n'
     '        _fh.write("".join(test_ids[i] + "\\n" for i in np.flatnonzero(_fail)))\n'
     '    print(f"[h958] 224 stage could not score {int(_fail.sum())} studies (raptor224_failed.txt)", flush=True)\n')])
patch(cells[12], [
    ("        print('R224_UNUSABLE', repr(_e), flush=True); r224 = None\n",
     "        print('R224_UNUSABLE', repr(_e), flush=True); r224 = None\n"
     "fail224 = np.zeros(len(test_ids), bool)   # [h958 R4] studies the 224 stage could not score keep the 384 rank alone\n"
     "if r224 is not None and (WORK / 'raptor224_failed.txt').is_file():\n"
     "    _f224 = set((WORK / 'raptor224_failed.txt').read_text().split())\n"
     "    fail224 = np.array([u in _f224 for u in test_ids])\n"),
    ("        rr = (1.0 - A224) * rr + A224 * pct_on(r224[c].values, ~rap_fail)\n",
     "        _ok224 = ~rap_fail & ~fail224   # [h958 R4] the 224 rank only over the studies both Raptor models scored\n"
     "        rr = np.where(_ok224, (1.0 - A224) * rr + A224 * pct_on(r224[c].values, _ok224), rr)\n")])

# R5 + R6: the final cell
_head_old = ("# [h957] final: 0.8 x rank(the R7 notebook's own submission.csv) + 0.2 x rank(our Mirror-student ensemble), per\n"
             "# finding, weights fixed before any run (public 0.955 notebook / our H955C).  A failed reader leaves the R7 file as is;\n"
             "# a study the reader could not score keeps its R7 rank.\n")
if (WP, WR) == (0.8, 0.2):
    _head_new = ("# [h957] final: 0.8 x rank(the R7 notebook's own submission.csv) + 0.2 x rank(our Mirror-student ensemble), per\n"
                 "# finding, weights fixed before any run (public 0.955 notebook / our H955C).  A failed reader leaves the R7 file as is;\n")
else:   # R6: say what the code really does
    _head_new = ("# [h957] final: %g x rank(the notebook's own submission.csv) + %g x rank(our Mirror-student ensemble), per\n"
                 "# finding.  The %g student weight was chosen after the 0.957 LB probes (H957x5-w30 / -w40), not before any run.\n"
                 "# A failed reader leaves the notebook's file as is;\n" % (WP, WR, WR))
_head_new += "# [h958 R5] a study the reader could not score gets its expected student rank given its parent rank.\n"
patch(final, [
    (_head_old, _head_new),
    ("        _out[_c] = _hn.where(_bad, _p0, W_PARENT * _p0 + W_READER * _hn.nan_to_num(_r)).astype(_hn.float32)\n",
     "        if _bad.any():   # [h958 R5] linear fit of the student rank on the parent rank over the scored studies\n"
     "            _r[_bad] = (_hn.clip(_hn.polyval(_hn.polyfit(_p0[~_bad], _r[~_bad], 1), _p0[_bad]), 0, 1)\n"
     "                        if (~_bad).sum() >= 3 else _p0[_bad])\n"
     "        _out[_c] = (W_PARENT * _p0 + W_READER * _r).astype(_hn.float32)\n")])

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
_students = "5 Mirror students" if not extra else "5 Mirror students + %d ensemble-teacher students" % len(extra)
title = ("# RSNA Knee %s: R7 + Raptor (384 + 224 with mirror) + our %s\n\n"
         "Built from varshithakolupuri's public *one free flip* notebook (heliosli's R7 + Raptor, plus nartaa's 224-px "
         "checkpoint with the anatomical mirror view, 35%% of the Raptor slot), with our Mirror-student readers "
         "added as in our %s (LB 0.957): they run right after the R7 stage, and the last cell blends "
         "%g x rank(the notebook's submission) + %g x rank(students).%s%s Robustness patches [h958 R1-R6] change nothing "
         "in a normal run: they act only when a stage fails or runs late.\n"
         % (_label, _students, _who, WP, WR, " R7 flip TTA is switched on (variant b)." if a.flip_r7 else "",
            (" The %d new students come from their own dataset (%s)." % (len(extra), a.extra_readers)) if extra else ""))
cells[0] = md(title)
reader_md = md("## Stage 2a: our Mirror-student readers (%s's reader stage + R1-R3: checkpoints by name, 9-hour time guard, "
               "slice count as in training)%s\n" % (_who, "; + %d new students" % len(extra) if extra else ""))
nb_cells = cells[:8] + [reader_md] + reader + cells[8:] + \
           [md("## Stage 4: %g x the notebook's blend + %g x our students (%s's final cell + R5/R6)\n" % (WP, WR, _who)), final]
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
      "| checkpoints", 5 + len(extra), "| changed" if changed else "| unchanged or new")
if a.owner != READERS.split("/")[0]:
    print("NOTE: kernel owner", a.owner, "needs read access to the private", READERS, "dataset")
