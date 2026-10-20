"""H960 family: the frozen public 0.958 notebook (heliosli blend-mine v1 = our team's best, kishormorol 57252502) plus
ONE change each, so every submission reads as a one-variable test against 0.958 (audit/17_research_0958_to_0959.md).

  --students   E1 (H960):  + our 5 Mirror students after the R6 stage; last cell = 0.8 x rank(0.958 file) + 0.2 x
               rank(students) -- a student leg at 0.2 lifted 0.954 -> 0.955 (1 student, H955C) and 0.956 -> 0.957 (3 students,
               H957); 5 students (H957x5) left the displayed 0.957 unchanged; weight fixed before any run.  Reader cells and final cell from student_leg.py (H957x5's cells + R1-R3 / R5-R6).
  --mirror224  E2 (H958M): nartaa's anatomical mirror view for the 224 model (the [s9] lines of varshithakolupuri's
               one-free-flip 224 cell, copied from our scored H958b), and that stage's timeout 2 h -> 2.5 h (the mirror
               doubles the 224 inference).  Unlike H958b / one-free-flip / goodpjw v26, whose 224+mirror stage was guarded
               (a timeout dropped the 224 model), the baseline's stage is fatal on timeout, and no scored run proves that
               224+mirror finishes within 2.5 h.  The evidence that it does: the baseline's 384 stage (plain + mirror, the same
               CoAtNet family at a larger input, the same prep) passed its hard 2 h limit on the hidden set.
  both         H960M (only after E1 and E2 are read).
  --parent c2  H961: the same hardened student leg (--students --robust) on foysal's C2 v2 instead of the 0.958 notebook.  C2 =
               the 0.958 notebook + nartaa S6fall-B at 1/3 rank (sub 57315983, public 0.958, the team best since 10-17); its 13
               cells run unchanged (only the title markdown differs), then the snapshot, the guarded students, 0.8 / 0.2.
  --v2         heliosli's V2 per-finding Raptor weights (public notebook "this-is-best-public-dataset-new-version-0-9586" v6,
               scored 0.958 displayed, titled 0.9586): Lateral OA 0.20 -> 0.15, PF OA 0.60 -> 0.65, Synovitis 0.80 -> 0.85,
               Baker's 0.35 -> 0.30 (Medial OA 0.50 and every other finding unchanged).  The weights were chosen on the public LB
               by heliosli, not by us.  That notebook's tiny goodpjw-reader leg (weights 0.003-0.035) is not taken.
               --parent c2 --v2 = H963; --parent c2 --students --robust --v2 = H964.
  --student-weight W   the student share of the last blend (default 0.2 as in every scored student run): (1-W) x rank(parent)
               + W x rank(students).  Needs --name.
  --partb      heliosli's "Part B" (the same public notebook's cell 9: goodpjw2008's 2.5D ConvNeXt reader, 3 folds, blended at
               his per-finding weights 0.003-0.035), appended AFTER our final cell ("end" mode, as heliosli runs it), so the
               student leg keeps its time; [h965] patches add a subprocess timeout (9 h - elapsed - 15 min) and a backup /
               verify pair around it (any problem restores the pre-Part-B file).  Needs --name.
  --student-tta  anatomical left-right mirror TTA for the 5 students (ours_leg/infer_tta.py, drafted and tested by the 10-19
               research workflow): each study decoded once, plain pass on cuda:0 = H964's exact computation, mirror pass on
               cuda:1 (sagittal: slice order reversed, pos -> 1 - pos; coronal / axial: image columns flipped; no label swap,
               preprocess.CANON keeps every series in the patient LPS frame); reader file = 0.5 x (plain + mirror).  A label-free
               guard on >= 200 studies (every finding's plain-vs-mirror Spearman >= 0.80, medial/lateral same-minus-cross >= 0.05)
               falls back to the plain pass; a crashed TTA script reruns H964's infer.py when >= 45 min are left.  Needs --name.
  --mavit W    hengck23's MaViT (MIT, public 0.935; foysal's C3 cell 4 script + marked [h967] patches, ours_leg/mavit_leg.py)
               appended AFTER our final cell: (1 - W) x rank(final file) + W x rank(MaViT), W fixed before any run; >= 70 min
               start guard, killed at 9 h - 12 min, every failure restores the pre-MaViT file.  Needs --name; not with --partb.

The baseline file is checked by sha256; its nine cells stay byte-identical apart from the marked changes.

    python ours_leg/build_h960.py --students --robust        -> candidate_notebook/submission_h960r/   (drafts; the pushed
    python ours_leg/build_h960.py --students --robust --mirror224 -> candidate_notebook/submission_h960rm/   ones are moved to our_submission_notebook/)
    python ours_leg/build_h960.py --students --owner kishormorol --out /abs/scratch/submission_h960   (other account)
    python ours_leg/build_h960.py --parent c2 --students --robust --owner kishormorol -> candidate_notebook/submission_h961/
    python ours_leg/build_h960.py --parent c2 --v2                                   -> candidate_notebook/submission_h963/
    python ours_leg/build_h960.py --parent c2 --students --robust --v2               -> candidate_notebook/submission_h964/
    python ours_leg/build_h960.py --parent c2 --students --robust --v2 --partb --name h965
    python ours_leg/build_h960.py --parent c2 --students --robust --v2 --student-weight 0.3 --name h966
    python ours_leg/build_h960.py --parent c2 --students --robust --v2 --mavit 0.1 --name h967
    python ours_leg/build_h960.py --parent c2 --students --robust --v2 --student-tta --name h968
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import CANDIDATES, inp, out_dir, write_json, write_text  # noqa: E402
from student_leg import READERS, READER_ASSETS, harden, patch, src, student_cells  # noqa: E402
from mavit_leg import MAVIT_DATASET, mavit_cells  # noqa: E402

BASE = "our_base_notebook/0.958_kishormorol__rsna-knee-helios-blend-fork.ipynb"
BASE_SHA = "00ca614beec23736346df921e3305742bf8533b7cfd69a208880022f899fa1d7"   # full sha256 of the scored 0.958 notebook
MIRROR_SRC = "submission_h958b/rsna-knee-h958b.ipynb"   # scored (0.957) notebook whose 224 cell carries the [s9] mirror lines
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
BASE_SOURCES = ["heliosli/rsna-knee-r6-frozen-fe055406-w1", "nartaa/rsna-knee-publication-swa-weights-20261007",
                "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"]
C2 = "submission_c2/rsna-knee-sub-c2.ipynb"   # foysalemonshanto/rsna-knee-sub-c2 v2, sub 57315983 (public 0.958, team best 10-17)
C2_SHA = "148eb4387de1aac2ccd374a8e052cdf759a0962f52c40ac06e6abe13af2942d7"
C2_SOURCES = ["heliosli/rsna-knee-r6-frozen-fe055406-w1", "nartaa/rsna-knee-s6fall-weights",
              "nartaa/rsna-knee-publication-swa-weights-20261007", "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"]
W_RAPTOR_C_OLD = "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, \"Baker's\": 0.35}"
W_RAPTOR_C_V2 = "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.15, 'PF OA': 0.65, 'Synovitis': 0.85, \"Baker's\": 0.30}"
TTA_SRC = "ours_leg/infer_tta.py"   # plain + anatomical mirror pass for the students (the draft's sha256 below)
TTA_SHA = "604fd196536ec9059b2d43513ede4737e618b1e1c1fa5d3fcb9e536d08886cb8"
TTA_OLD = (  # H964 cell 16: the reader-run block inside its try (replaced as a whole by --student-tta)
    '    if os.path.exists("/kaggle/working/reader_sub.csv"):   # [h960r] never accept a stale file from an earlier run\n'
    '        os.remove("/kaggle/working/reader_sub.csv")\n'
    '    _t0 = time.time()\n'
    '    try:   # [h958 R2] time guard: the readers never push the session past 9 h; a timeout only drops the student leg\n'
    '        rc = subprocess.run([sys.executable, "/kaggle/working/own_src/infer.py", _data, "test", "/kaggle/working/reader_sub.csv"] + _ck,\n'
    '                            env=_env, timeout=max(60, 9 * 3600 - (time.time() - NB_T0) - 20 * 60)).returncode\n'
    '    except subprocess.TimeoutExpired:\n'
    '        rc = -9\n'
    '        print("READER_TIMEOUT after %.0f s - the final cell keeps the notebook\'s own blend" % (time.time() - _t0), flush=True)\n'
    '    READER_OK = rc == 0 and os.path.isfile("/kaggle/working/reader_sub.csv")\n'
    '    READER_INFO["reader_rc"] = rc\n')
TTA_NEW = (
    '    for _stale in ("reader_sub.csv", "reader_sub_plain.csv", "reader_sub_mirror.csv"):   # [h960r][h968] never accept a stale file\n'
    '        if os.path.exists("/kaggle/working/" + _stale):\n'
    '            os.remove("/kaggle/working/" + _stale)\n'
    '    _t0 = time.time()\n'
    '    def _run_reader(script):   # [h958 R2] time guard: the readers never push the session past 9 h; a timeout only drops the student leg\n'
    '        try:\n'
    '            return subprocess.run([sys.executable, "/kaggle/working/own_src/" + script, _data, "test", "/kaggle/working/reader_sub.csv"] + _ck,\n'
    '                                  env=_env, timeout=max(60, 9 * 3600 - (time.time() - NB_T0) - 20 * 60)).returncode\n'
    '        except subprocess.TimeoutExpired:   # [h968 review] its own sentinel: a child killed by a signal (e.g. -9) is a crash\n'
    '            print("READER_TIMEOUT after %.0f s - the final cell keeps the notebook\'s own blend" % (time.time() - _t0), flush=True)\n'
    '            return "timeout"\n'
    '    rc = _run_reader("infer_tta.py")   # [h968] plain + anatomical mirror; each study decoded once; mirror pass on cuda:1\n'
    '    READER_INFO["tta_rc"] = rc\n'
    '    if rc not in (0, "timeout") and 9 * 3600 - (time.time() - NB_T0) > 45 * 60:\n'
    '        rc = _run_reader("infer.py")   # [h968] the TTA script failed or was killed: H964\'s own plain reader, unchanged\n'
    '        READER_INFO["tta"] = "infer_tta.py failed (rc %s) - H964 plain reader rerun, rc %s" % (READER_INFO["tta_rc"], rc)\n'
    '    elif rc == 0:\n'
    '        # [h968] label-free acceptance on the hidden studies, thresholds fixed before any run, applied only with >= 200 studies\n'
    '        # (3 visible studies cannot judge it: logged only): every finding\'s plain-vs-mirror Spearman >= 0.80 and, for both\n'
    '        # medial/lateral pairs, same-side minus cross-side Spearman >= 0.05 (a side-swapping mirror would pair Medial with\n'
    '        # Lateral).  Rejected, or any error here: reader_sub.csv = the plain pass = H964\'s students exactly.\n'
    '        _tta = {}\n'
    '        try:\n'
    '            _P = _rp.read_csv("/kaggle/working/reader_sub_plain.csv", index_col=0)\n'
    '            _Q = _rp.read_csv("/kaggle/working/reader_sub_mirror.csv", index_col=0)\n'
    '            _sp = lambda a, b: float(_rp.Series(a).rank().corr(_rp.Series(b).rank()))\n'
    '            _rho = {c: _sp(_P[c].values, _Q[c].values) for c in _P.columns}\n'
    '            _ml = min(_rho[a] - _sp(_P[a].values, _Q[b].values)\n'
    '                      for a, b in (("Medial Meniscus", "Lateral Meniscus"), ("Lateral Meniscus", "Medial Meniscus"),\n'
    '                                   ("Medial OA", "Lateral OA"), ("Lateral OA", "Medial OA")))\n'
    '            import numpy as _rn   # [h968 review] a non-finite value in either pass is never averaged in\n'
    '            _fin = bool(_P.shape == _Q.shape and _rn.isfinite(_P.values).all() and _rn.isfinite(_Q.values).all())\n'
    '            _tta.update(n=len(_P), finite=_fin, min_spearman=round(min(_rho.values()), 4), ml_margin=round(_ml, 4))\n'
    '            _tta["accepted"] = bool(_fin and (len(_P) < 200 or (all(v >= 0.80 for v in _rho.values()) and _ml >= 0.05)))\n'
    '            if len(_P) < 200:\n'
    '                _tta["guard"] = "logged only (< 200 studies)"\n'
    '        except Exception as _ge:\n'
    '            _tta.update(accepted=False, guard_error="%s: %s" % (type(_ge).__name__, str(_ge)[:200]))\n'
    '        if not _tta.get("accepted"):\n'
    '            try:\n'
    '                import shutil as _rsh\n'
    '                _rsh.copyfile("/kaggle/working/reader_sub_plain.csv", "/kaggle/working/reader_sub.csv")\n'
    '                _tta["used"] = "plain pass (= H964 students)"\n'
    '            except Exception as _ce:\n'
    '                rc = -8\n'
    '                _tta["used"] = "no reader file (%s)" % type(_ce).__name__\n'
    '        else:\n'
    '            _tta["used"] = "0.5 x (plain + mirror)"\n'
    '        READER_INFO["tta"] = _tta\n'
    '    READER_OK = rc == 0 and os.path.isfile("/kaggle/working/reader_sub.csv")\n'
    '    READER_INFO["reader_rc"] = rc\n')
PARTB_SRC = "public_notebooks/0.958_159v_heliosli__this-is-best-public-dataset-new-version-0-9586.ipynb"   # v6, its cell 9
PARTB_SHA = "cb0e1eac33a00f96b7d1bcc0e9914f7eb045af2bc9cd01bc290b2213e7f17ddd"
ap = argparse.ArgumentParser()
ap.add_argument("--parent", choices=["helios", "c2"], default="helios", help="helios = the scored 0.958 notebook (H960 family); "
                "c2 = foysal's C2 v2 (H961, needs --students --robust)")
ap.add_argument("--students", action="store_true")
ap.add_argument("--mirror224", action="store_true")
ap.add_argument("--v2", action="store_true", help="heliosli's V2 per-finding Raptor weights (LB-chosen by heliosli); parent c2 only")
ap.add_argument("--robust", action="store_true", help="H960r: baseline first + snapshot, guarded student stage, checkpoint sha256, "
                "strict acceptance (independent review 10-17); requires --students")
ap.add_argument("--student-nb", default="submission_h957x5/rsna-knee-h957x5.ipynb", help="scored 0.8/0.2 student notebook (H957x5)")
ap.add_argument("--student-weight", type=float, default=0.2, help="student share of the final blend (default 0.2)")
ap.add_argument("--partb", action="store_true", help="append heliosli's Part B goodpjw-reader leg after the final cell")
ap.add_argument("--student-tta", action="store_true", help="plain + anatomical mirror TTA for the students (ours_leg/infer_tta.py)")
ap.add_argument("--mavit", type=float, default=0.0, help="weight of hengck23's MaViT leg after the final cell (0 = off)")
ap.add_argument("--name", default=None, help="experiment name (submission_<name>, rsna-knee-<name>) for --student-weight / --partb")
ap.add_argument("--owner", default="hosen42")
ap.add_argument("--out", default=None)
a = ap.parse_args()
if not (a.students or a.mirror224 or a.v2):
    ap.error("choose --students, --mirror224 and/or --v2")
if a.robust and not a.students:
    ap.error("--robust hardens the student stage: use it with --students")
if a.v2 and a.parent != "c2":
    ap.error("--v2 is built on --parent c2 only")
if a.parent == "c2" and (a.mirror224 or (a.students and not a.robust)):
    ap.error("--parent c2 takes --students only with --robust, and no --mirror224")
if (a.partb or a.student_weight != 0.2) and not (a.parent == "c2" and a.students and a.robust and a.name):
    ap.error("--partb / --student-weight need --parent c2 --students --robust and --name")
if a.mavit and (a.partb or not (a.parent == "c2" and a.students and a.robust and a.name) or not 0.0 < a.mavit <= 0.3):
    ap.error("--mavit W (0 < W <= 0.3) needs --parent c2 --students --robust and --name, and is not combined with --partb")
if a.student_tta and (a.partb or a.mavit or not (a.parent == "c2" and a.students and a.robust and a.name)):
    ap.error("--student-tta needs --parent c2 --students --robust and --name (not combined with --partb / --mavit yet)")
if not 0.05 <= a.student_weight <= 0.5:
    ap.error("--student-weight must be in [0.05, 0.5]")

if a.parent == "c2":   # H961: C2's 13 cells stay byte-identical apart from the title markdown (cell 0)
    raw = open(inp(C2), "rb").read()
    assert hashlib.sha256(raw).hexdigest() == C2_SHA, "the archived C2 v2 notebook changed"
    P = json.loads(raw)
    cells = [json.loads(json.dumps(c)) for c in P["cells"]]
    assert len(cells) == 13 and [c["cell_type"] for c in cells] == ["markdown"] * 3 + ["code"] * 10
    assert src(cells[0]).startswith("# C2: the 0.958 notebook + S6fall-B (1/3)")
    assert "NB_T0 = time.time()" in src(cells[4]) and src(cells[3]).startswith("%%writefile /tmp/raptor_run.py")
    assert src(cells[8]).startswith("# Stage 1c [C2 10-17]") and "S6F_CAP_S = 2.5 * 3600" in src(cells[8])
    assert "BUNDLE_ID = 'fe055406fad6a12f'" in src(cells[9]) and "FLIP_TTA = True" in src(cells[9]) and "WORK = " in src(cells[9])
    assert "W_S6F = 1.0 / 3.0" in src(cells[12])
    assert "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, \"Baker's\": 0.35}" in src(cells[12])
    assert "sub.to_csv(WORK / 'submission.csv', index=False)" in src(cells[12])
    if a.v2:   # the one-line weight change, in C2's final blend cell
        assert src(cells[12]).count(W_RAPTOR_C_OLD) == 1
        patch(cells[12], [(W_RAPTOR_C_OLD, W_RAPTOR_C_V2)])
else:
    raw = open(inp(BASE), "rb").read()
    assert hashlib.sha256(raw).hexdigest() == BASE_SHA, "the frozen 0.958 baseline changed"
    P = json.loads(raw)
    cells = [json.loads(json.dumps(c)) for c in P["cells"]]
    assert len(cells) == 9 and [c["cell_type"] for c in cells] == ["markdown"] * 2 + ["code"] * 7
    assert src(cells[2]).startswith("%%writefile /tmp/raptor_run.py") and "NB_T0 = time.time()" in src(cells[3])
    assert src(cells[4]).startswith("%%writefile /tmp/raptor224_run.py") and "c08_r224_swa.pt" in src(cells[4])
    assert src(cells[5]).startswith("# Stage 1b [d957]") and "RAPTOR224_SECONDS" in src(cells[5])
    assert "BUNDLE_ID = 'fe055406fad6a12f'" in src(cells[6]) and "FLIP_TTA = True" in src(cells[6]) and "WORK = " in src(cells[6])
    assert "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, \"Baker's\": 0.35}" in src(cells[8])
    assert "sub.to_csv(WORK / 'submission.csv', index=False)" in src(cells[8])
NP = len(cells)   # the parent's cells: 9 (0.958 notebook) or 13 (C2)

if a.mirror224:   # E2: the [s9] lines, taken verbatim from the scored H958b 224 cell (one-free-flip, varshithakolupuri v1)
    m = src(json.load(open(inp(MIRROR_SRC)))["cells"][15])
    assert m.startswith("%%writefile /tmp/raptor224_run.py") and m.count("[s9]") == 3
    i0 = m.index("# [s9] anatomical L/R mirror TTA"); i1 = m.index("_WLUT = {}\n")
    s9_block = m[i0:i1]
    s9_if = m[m.index("    if mirror:   # [s9]\n"):m.index("    key = (str(dev), norm)\n")]
    s9_else = m[m.index("                    else:   # [s9] average"):m.index("            except Exception as e:\n", m.index("                    else:   # [s9] average"))]
    patch(cells[4], [
        ("_WLUT = {}\ndef gpu_windows(vol, mask, k, res, dev, norm=NORM):\n",
         s9_block + "_WLUT = {}\ndef gpu_windows(vol, mask, k, res, dev, norm=NORM, mirror=False):\n"),
        ("(k,3,IMG,IMG) uint8\n    key = (str(dev), norm)\n", "(k,3,IMG,IMG) uint8\n" + s9_if + "    key = (str(dev), norm)\n"),
        ("                    else:\n                        arm_probs[a][i] = infer_probs(model, xw, d)\n", s9_else)])
    patch(cells[5], [("rc = subprocess.run([sys.executable, '-u', '/tmp/raptor224_run.py'], timeout=2 * 3600).returncode\n",
                      "rc = subprocess.run([sys.executable, '-u', '/tmp/raptor224_run.py'], timeout=int(2.5 * 3600)).returncode"
                      "   # [h958m] the mirror view doubles the 224 inference\n")])

name = ("h964" if a.students else "h963") if a.v2 else "h961" if a.parent == "c2" else ("h960r" if a.robust else "h960") if a.students else "h958m"
name += "m" if (a.students and a.mirror224) else ""
name = a.name or name
WS = round(a.student_weight, 4)
WS_TXT = "%g x rank(parent file) + %g x rank(students)" % (round(1 - WS, 4), WS)
out_name = a.out or f"submission_{name}"
base = os.path.basename(os.path.normpath(out_name))
assert base.startswith("submission_"), out_name
slug = "rsna-knee-" + base.replace("submission_", "")
md = lambda t: {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
changes = []
if a.students:
    if a.robust:   # wording corrected after the 10-17 reviews (the pushed H960 keeps its original title text)
        changes.append("our 5 Mirror-student readers (H957x5) after the parent notebook, blended last as 0.8 x rank(parent file) + "
                       "0.2 x rank(students) (weight fixed before any run; a student leg at 0.2 lifted 0.954 -> 0.955 with 1 student "
                       "and 0.956 -> 0.957 with 3 students; 5 students left the displayed 0.957 unchanged)")
    else:
        changes.append("our 5 Mirror-student readers after the R6 stage, blended last as 0.8 x rank(the 0.958 notebook's file) + "
                       "0.2 x rank(students) (weight fixed before any run; the same add-on lifted 0.954 -> 0.955 and 0.956 -> 0.957)")
if a.students and WS != 0.2:   # [h966] only the weight words change; the 0.2 texts stay as pushed
    changes[-1] = changes[-1].replace("0.8 x rank(parent file) + 0.2 x rank(students)", WS_TXT) \
        .replace("(weight fixed before any run; ", "(student weight %g chosen before this run, single change vs H964; the 0.2 runs: " % WS)
if a.partb:
    changes.append("heliosli's Part B (goodpjw2008's 2.5D ConvNeXt reader, 3 folds, per-finding weights 0.003-0.035, his notebook "
                   "cell 9 unchanged apart from a subprocess timeout), run after our final cell with a backup / verify guard")
if a.student_tta:
    changes.append("anatomical left-right mirror TTA for the 5 students (each study decoded once, mirror pass on the second T4, "
                   "reader file = 0.5 x (plain + mirror); a label-free guard on >= 200 studies falls back to the plain pass)")
if a.mavit:
    changes.append("hengck23's MaViT (MIT, public 0.935; foysal's C3 script) after our final cell at weight %g, rank-blended, "
                   "start guard 70 min, every failure keeps the file before it" % a.mavit)
if a.v2:
    changes.insert(0, "heliosli's V2 per-finding Raptor weights in the final blend (Lateral OA 0.15, PF OA 0.65, Synovitis 0.85, "
                      "Baker's 0.30; chosen on the public LB by heliosli)")
if a.mirror224:
    changes.append("nartaa's anatomical mirror view for the 224 model (as in one-free-flip), with that stage's timeout 2 h -> 2.5 h")
if a.parent == "c2":
    cells[0] = md(f"# RSNA Knee {base.replace('submission_', '').upper()}: C2 (the 0.958 notebook + S6fall-B 1/3) + "
                  f"{' + '.join(['V2 weights'] * a.v2 + ['our 5 students' + ('' if WS == 0.2 else ' at %g' % WS)] * a.students + ['Part B reader'] * a.partb + ['MaViT %g' % a.mavit] * bool(a.mavit) + ['student mirror TTA'] * a.student_tta)}\n\n"
                  "Frozen parent: foysal's C2 v2 (`foysalemonshanto/rsna-knee-sub-c2`, sub 57315983, public 0.958, the team best since "
                  "10-17) = heliosli's public blend-mine 0.958 notebook (Raptor 384 + 224 + compact R6 with flip TTA, per-finding Raptor "
                  "weights) + nartaa's S6fall model B at 1/3 rank.  "
                  + ("Cells 1-12 run unchanged (this title cell is the only edit).  Then: " + "; ".join(changes)   # H961 text as pushed
                     + ".  If the student stage fails, runs late or is not accepted, the C2 snapshot is the submission.\n" if not a.v2 else
                     "Cells 1-12 run unchanged apart from this title cell and the W_RAPTOR_C line of cell 12.  Changes: "
                     + "; ".join(changes) + "."
                     + ("  If the student stage fails, runs late or is not accepted, the parent snapshot is the submission." if a.students else "")
                     + "\n"))
else:
    cells[0] = md(f"# RSNA Knee {base.replace('submission_', '').upper()}: the public 0.958 notebook + {' + '.join(['students'] * a.students + ['224 mirror'] * a.mirror224)}\n\n"
                  "Frozen base: heliosli's public *Blend mine in - private LB luck to all upvoters* v1 (public LB 0.958): Raptor 384 + "
                  "224 (35% of the Raptor half) + compact R6 with flip TTA, per-finding Raptor weights.  Its cells run unchanged except: "
                  + "; ".join(changes) + ".  Robustness patches [h958 R1-R6] act only when a stage fails or runs late.\n")
nb_cells = cells
if a.students:
    st = json.load(open(inp(a.student_nb)))["cells"]
    reader, final, WP, WR = student_cells(st, parent_name="the parent notebook (see PARENT_LABEL)" if a.robust else "the 0.958 notebook")
    assert (WP, WR) == (0.8, 0.2), (WP, WR)
    if a.robust:   # baseline first: c0-c8 run unchanged and contiguous, then the snapshot, then the optional student stage
        _plabel = ("foysal's C2 v2 cells (the 0.958 notebook + nartaa S6fall-B 1/3, sub 57315983) with heliosli's V2 per-finding "
                   "Raptor weights (Lateral OA 0.15, PF OA 0.65, Synovitis 0.85, Baker's 0.30)" if a.parent == "c2" and a.v2 else
                   "foysal's C2 v2 cells (the 0.958 notebook + nartaa S6fall-B 1/3, sub 57315983), unchanged" if a.parent == "c2" else
                   "the 0.958 notebook's cells + nartaa's 224 anatomical mirror (the H958M parent, NOT the scored 0.958 file)"
                   if a.mirror224 else "the scored 0.958 notebook's cells, unchanged")
        snap, reader, final = harden(reader, final, _plabel)
        if a.student_tta:   # [h968] the writer also writes infer_tta.py; the run cell's reader block runs it (plain fallback)
            import ast
            _tta_raw = open(inp(TTA_SRC), "rb").read()
            assert hashlib.sha256(_tta_raw).hexdigest() == TTA_SHA, "ours_leg/infer_tta.py changed"
            _wline = next(l for l in src(reader[0]).splitlines() if l.startswith("_SOURCES = "))
            _files = ast.literal_eval(_wline[len("_SOURCES = "):])
            assert sorted(_files) == ["infer.py", "knee.py", "preprocess.py"], sorted(_files)
            _files["infer_tta.py"] = _tta_raw.decode()
            patch(reader[0], [(_wline + "\n", "_SOURCES = " + repr(_files) + "\n")])
            patch(reader[1], [(TTA_OLD, TTA_NEW)])
        if WS != 0.2:   # [h966] the one-number change in the final cell (and its header line)
            patch(final, [("W_PARENT, W_READER = 0.8, 0.2\n", "W_PARENT, W_READER = %g, %g   # [h966] student weight %g (was 0.2)\n" % (round(1 - WS, 4), WS, WS)),
                          ("# [h957] final: 0.8 x rank(", "# [h957] final: %g x rank(" % round(1 - WS, 4)),
                          (") + 0.2 x rank(our Mirror-student ensemble)", ") + %g x rank(our Mirror-student ensemble)" % WS)])
        nb_cells = cells[:NP] + [snap, md("## Stage 2a: our 5 Mirror-student readers, optional and guarded (H957x5's reader stage + R1-R3 + "
                                          "checkpoint sha256, data identity, time left)\n")] + reader + \
                   [md("## Stage 4: %g x the parent snapshot + %g x our students, strict acceptance (else the parent snapshot is kept)\n"
                       % (round(1 - WS, 4), WS)), final]
        if a.partb:   # heliosli's Part B after our final cell, between a backup cell and a verify cell that never raise
            _pr = open(inp(PARTB_SRC), "rb").read()
            assert hashlib.sha256(_pr).hexdigest() == PARTB_SHA, "the archived heliosli v6 notebook changed"
            pb = json.loads(json.dumps(json.loads(_pr)["cells"][9]))
            assert src(pb).startswith("# Part B (V2): goodpjw2008's 2.5D ConvNeXt MIL reader") and "OWN_W_PER_TARGET = {" in src(pb)
            patch(pb, [('_o_sp.run([_o_sys.executable, "/kaggle/working/own_src/infer.py", _o_data, "test", "/kaggle/working/_own.csv"] + _o_ck,\n'
                        '                  check=True, env=_o_env)\n',
                        '_o_sp.run([_o_sys.executable, "/kaggle/working/own_src/infer.py", _o_data, "test", "/kaggle/working/_own.csv"] + _o_ck,\n'
                        '                  check=True, env=_o_env,   # [h965] never past 9 h: a timeout lands in the except below (file restored)\n'
                        '                  timeout=max(60, int(9 * 3600 - (_o_time.time() - globals().get("NB_T0", _o_time.time())) - 15 * 60)))\n')])
            code = lambda t: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.splitlines(keepends=True)}
            pb_backup = code("# [h965] backup of our final submission before heliosli's Part B; the verify cell after it restores this file if\n"
                             "# Part B leaves anything unusable.  Never raises.\n"
                             "import shutil as _kb_sh, hashlib as _kb_hl\n"
                             "PB_BACKUP_SHA256 = None\n"
                             "try:\n"
                             "    _kb_sh.copyfile('/kaggle/working/submission.csv', '/kaggle/working/_h965_pre_partb.csv')\n"
                             "    PB_BACKUP_SHA256 = _kb_hl.sha256(open('/kaggle/working/_h965_pre_partb.csv', 'rb').read()).hexdigest()\n"
                             "except Exception as _kb_e:\n"
                             "    print('PB_BACKUP_FAILED', type(_kb_e).__name__, str(_kb_e)[:200], flush=True)\n"
                             "print('PB_BACKUP sha256', PB_BACKUP_SHA256, flush=True)\n")
            pb_verify = code("# [h965] verify the file Part B left: same columns and studies in the same order as the backup, every target\n"
                             "# finite in [0, 1]; otherwise restore the backup (copy, then atomic replace).  Never raises.\n"
                             "import os as _kv_os, shutil as _kv_sh, hashlib as _kv_hl, json as _kv_js\n"
                             "import numpy as _kv_np, pandas as _kv_pd\n"
                             "_kv = {'status': 'unchecked'}\n"
                             "try:\n"
                             "    _kv_b = _kv_pd.read_csv('/kaggle/working/_h965_pre_partb.csv', dtype={'StudyInstanceUID': str})\n"
                             "    try:\n"
                             "        _kv_s = _kv_pd.read_csv('/kaggle/working/submission.csv', dtype={'StudyInstanceUID': str})\n"
                             "        _kv_l = list(_kv_b.columns)[1:]\n"
                             "        _kv_ok = (list(_kv_s.columns) == list(_kv_b.columns) and _kv_s['StudyInstanceUID'].tolist() == _kv_b['StudyInstanceUID'].tolist()\n"
                             "                  and _kv_np.isfinite(_kv_s[_kv_l].values).all() and ((_kv_s[_kv_l].values >= 0) & (_kv_s[_kv_l].values <= 1)).all())\n"
                             "    except Exception as _kv_e1:\n"
                             "        _kv_ok = False\n"
                             "        _kv['read_error'] = '%s: %s' % (type(_kv_e1).__name__, str(_kv_e1)[:200])\n"
                             "    if _kv_ok:\n"
                             "        _kv['status'] = 'part B file accepted'\n"
                             "        _kv['changed'] = _kv_hl.sha256(open('/kaggle/working/submission.csv', 'rb').read()).hexdigest() != PB_BACKUP_SHA256\n"
                             "    else:\n"
                             "        _kv_sh.copyfile('/kaggle/working/_h965_pre_partb.csv', '/kaggle/working/_h965_restore.csv')\n"
                             "        _kv_os.replace('/kaggle/working/_h965_restore.csv', '/kaggle/working/submission.csv')\n"
                             "        _kv['status'] = 'part B file rejected - pre-Part-B file restored'\n"
                             "except Exception as _kv_e:\n"
                             "    _kv['status'] = 'verify failed (%s) - submission.csv left as is' % type(_kv_e).__name__\n"
                             "print('PB_VERIFY', _kv_js.dumps(_kv), flush=True)\n")
            nb_cells += [md("## Stage 5 [h965]: heliosli's Part B (goodpjw2008's 2.5D reader at tiny per-finding weights) after our final "
                            "cell, guarded by a backup and a verify cell\n"), pb_backup, pb, pb_verify]
    else:
        nb_cells = cells[:8] + [md("## Stage 2a: our 5 Mirror-student readers (H957x5's reader stage + R1-R3)\n")] + reader + cells[8:] + \
                   [md("## Stage 4: 0.8 x the 0.958 notebook's blend + 0.2 x our students (H957x5's final cell + R5)\n"), final]
for c in nb_cells:
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
        compile("\n".join(l for l in src(c).splitlines() if not l.startswith(("%%", "!"))), "cell", "exec")
if a.parent == "c2":   # the parent's code and markdown after the title, unchanged and in order (cell 12: only the V2 line)
    _orig = [src(c) for c in json.loads(raw)["cells"]]
    if a.v2:
        _orig[12] = _orig[12].replace(W_RAPTOR_C_OLD, W_RAPTOR_C_V2)
    assert [src(c) for c in nb_cells[1:NP]] == _orig[1:NP]
P["cells"] = nb_cells
if a.mavit:   # hengck23's MaViT after our final cell (writer / runner / blend cells, none raises)
    nb_cells.extend(dict(c, source=list(c["source"])) for c in mavit_cells(a.mavit)[0])   # in place: P["cells"] is this list
    assert P["cells"] is nb_cells
    for c in nb_cells[-3:]:
        compile("".join(c["source"]), "cell", "exec")
sources = (C2_SOURCES if a.parent == "c2" else BASE_SOURCES) + ([READER_ASSETS, READERS] if a.students else []) + \
          ([MAVIT_DATASET] if a.mavit else [])
meta = {"id": f"{a.owner}/{slug}", "title": slug, "code_file": slug + ".ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "docker_image": DOCKER, "docker_image_pinning_type": "original", "dataset_sources": sources,
        "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
out = out_dir(out_name, base=CANDIDATES)   # drafts go to candidate_notebook/; move to our_submission_notebook/ once pushed
write_text(os.path.join(out, slug + ".ipynb"), json.dumps(P, indent=1))
write_json(os.path.join(out, "kernel-metadata.json"), meta, indent=2)
print("wrote", out_name, "| cells", len(nb_cells), "| students", a.students, "| mirror224", a.mirror224, "| sources", sources)
if a.students and a.owner != READERS.split("/")[0]:
    print("NOTE: kernel owner", a.owner, "needs read access to the private", READERS, "dataset")
