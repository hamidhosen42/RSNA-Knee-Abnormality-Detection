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

The baseline file is checked by sha256; its nine cells stay byte-identical apart from the marked changes.

    python ours_leg/build_h960.py --students --robust        -> candidate_notebook/submission_h960r/   (drafts; the pushed
    python ours_leg/build_h960.py --students --robust --mirror224 -> candidate_notebook/submission_h960rm/   ones are moved to our_submission_notebook/)
    python ours_leg/build_h960.py --students --owner kishormorol --out /abs/scratch/submission_h960   (other account)
    python ours_leg/build_h960.py --parent c2 --students --robust --owner kishormorol -> candidate_notebook/submission_h961/
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import CANDIDATES, inp, out_dir, write_json, write_text  # noqa: E402
from student_leg import READERS, READER_ASSETS, harden, patch, src, student_cells  # noqa: E402

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
ap = argparse.ArgumentParser()
ap.add_argument("--parent", choices=["helios", "c2"], default="helios", help="helios = the scored 0.958 notebook (H960 family); "
                "c2 = foysal's C2 v2 (H961, needs --students --robust)")
ap.add_argument("--students", action="store_true")
ap.add_argument("--mirror224", action="store_true")
ap.add_argument("--robust", action="store_true", help="H960r: baseline first + snapshot, guarded student stage, checkpoint sha256, "
                "strict acceptance (independent review 10-17); requires --students")
ap.add_argument("--student-nb", default="submission_h957x5/rsna-knee-h957x5.ipynb", help="scored 0.8/0.2 student notebook (H957x5)")
ap.add_argument("--owner", default="hosen42")
ap.add_argument("--out", default=None)
a = ap.parse_args()
if not (a.students or a.mirror224):
    ap.error("choose --students and/or --mirror224")
if a.robust and not a.students:
    ap.error("--robust hardens the student stage: use it with --students")
if a.parent == "c2" and (not (a.students and a.robust) or a.mirror224):
    ap.error("--parent c2 (H961) takes --students --robust and no --mirror224")

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

name = "h961" if a.parent == "c2" else ("h960r" if a.robust else "h960") if a.students else "h958m"
name += "m" if (a.students and a.mirror224) else ""
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
if a.mirror224:
    changes.append("nartaa's anatomical mirror view for the 224 model (as in one-free-flip), with that stage's timeout 2 h -> 2.5 h")
if a.parent == "c2":
    cells[0] = md(f"# RSNA Knee {base.replace('submission_', '').upper()}: C2 (the 0.958 notebook + S6fall-B 1/3) + our 5 students\n\n"
                  "Frozen parent: foysal's C2 v2 (`foysalemonshanto/rsna-knee-sub-c2`, sub 57315983, public 0.958, the team best since "
                  "10-17) = heliosli's public blend-mine 0.958 notebook (Raptor 384 + 224 + compact R6 with flip TTA, per-finding Raptor "
                  "weights) + nartaa's S6fall model B at 1/3 rank.  Cells 1-12 run unchanged (this title cell is the only edit).  Then: "
                  + "; ".join(changes) + ".  If the student stage fails, runs late or is not accepted, the C2 snapshot is the submission.\n")
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
        _plabel = ("foysal's C2 v2 cells (the 0.958 notebook + nartaa S6fall-B 1/3, sub 57315983), unchanged" if a.parent == "c2" else
                   "the 0.958 notebook's cells + nartaa's 224 anatomical mirror (the H958M parent, NOT the scored 0.958 file)"
                   if a.mirror224 else "the scored 0.958 notebook's cells, unchanged")
        snap, reader, final = harden(reader, final, _plabel)
        nb_cells = cells[:NP] + [snap, md("## Stage 2a: our 5 Mirror-student readers, optional and guarded (H957x5's reader stage + R1-R3 + "
                                          "checkpoint sha256, data identity, time left)\n")] + reader + \
                   [md("## Stage 4: 0.8 x the parent snapshot + 0.2 x our students, strict acceptance (else the parent snapshot is kept)\n"), final]
    else:
        nb_cells = cells[:8] + [md("## Stage 2a: our 5 Mirror-student readers (H957x5's reader stage + R1-R3)\n")] + reader + cells[8:] + \
                   [md("## Stage 4: 0.8 x the 0.958 notebook's blend + 0.2 x our students (H957x5's final cell + R5)\n"), final]
for c in nb_cells:
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
        compile("\n".join(l for l in src(c).splitlines() if not l.startswith(("%%", "!"))), "cell", "exec")
if a.parent == "c2":   # the parent's code and markdown after the title, unchanged and in order
    assert [src(c) for c in nb_cells[1:NP]] == [src(c) for c in json.loads(raw)["cells"][1:NP]]
P["cells"] = nb_cells
sources = (C2_SOURCES if a.parent == "c2" else BASE_SOURCES) + ([READER_ASSETS, READERS] if a.students else [])
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
