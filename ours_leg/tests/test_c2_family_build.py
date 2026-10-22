"""Structure of the built C2-family notebooks (no GPU, stdlib only):
  H961 = C2 v2 + our 5 students; H963 = C2 v2 + heliosli's V2 per-finding Raptor weights; H964 = C2 v2 + V2 weights + students.
The C2 parent is intact and runs first (cell 12 differs only by the V2 line), the student leg is the hardened H960r one, and
the metadata mounts what the cells read.  Run from the repo root after the build:

    python -I ours_leg/tests/test_c2_family_build.py our_submission_notebook/submission_h961 [our_submission_notebook/submission_h963 ...]
"""
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "ours_leg"))
from student_leg import CKPT_SHA256, READER_ASSETS, READERS  # noqa: E402

src = lambda c: "".join(c["source"])
C2P = os.path.join(ROOT, "our_submission_notebook", "submission_c2", "rsna-knee-sub-c2.ipynb")
C2_SHA = "148eb4387de1aac2ccd374a8e052cdf759a0962f52c40ac06e6abe13af2942d7"
OLD = "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, \"Baker's\": 0.35}"
V2 = "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.15, 'PF OA': 0.65, 'Synovitis': 0.85, \"Baker's\": 0.30}"
C2_SOURCES = ["heliosli/rsna-knee-r6-frozen-fe055406-w1", "nartaa/rsna-knee-s6fall-weights",
              "nartaa/rsna-knee-publication-swa-weights-20261007", "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"]
VARIANTS = {"h961": dict(v2=False, students=True), "h963": dict(v2=True, students=False), "h964": dict(v2=True, students=True),
            "h965": dict(v2=True, students=True, partb=True), "h966": dict(v2=True, students=True, ws=0.3),
            "h967": dict(v2=True, students=True, mavit=0.1), "h968": dict(v2=True, students=True, tta=True),
            "h969": dict(v2=True, students=True, bk=True), "h970": dict(v2=True, students=True, bk=True, tta=True),
            "h971": dict(v2=True, students=True, bk=True, fk=True), "h972": dict(v2=True, students=True, bk=True, fk=True, tta=True),
            "h973": dict(v2=False, students=True, bk=True, tta=True), "h974": dict(v2=False, students=True, bk=True, fk=True),
            "h975": dict(v2=True, students=True, bk=True, partb=True), "h976": dict(v2=True, students=True, bk=True, s1=True),
            "h977": dict(v2=True, students=True, bk=True, partb=True, s1=True), "h978": dict(v2=True, students=True, bk=True, fk=True, partb=True, s1=True),
            "h979": dict(v2=True, students=True, bk=True, fk=True, partb=True)}
S1_SHA = {"cnxt_plE75s_full.pt": "9efa571b1a1f9c64fe1a837950b270c7530db5e565a4129f07a2c5f836c559a3",   # hosen42/rsna-knee-gpjw-student5 output,
          "cnxt_plE75t_full.pt": "6e55ab9126a3409fc14b91c191507de91a825a4b400720d0607ac321c4ec0439"}   # computed from the downloaded files
# R6 switches in C2 cell 9.  BK = foysal's lines, cut by r6_switches.bk_new() from HIS archived notebook (sha256-checked),
# so the check is independent of the candidate; FK = the same pattern for the fluid stacker.
from r6_switches import BK_CELL9_SHA, BK_OLD, FK_NEW, FK_OLD, bk_new  # noqa: E402
PARTB_SRC = os.path.join(ROOT, "public_notebooks", "0.958_159v_heliosli__this-is-best-public-dataset-new-version-0-9586.ipynb")
PARTB_SHA = "cb0e1eac33a00f96b7d1bcc0e9914f7eb045af2bc9cd01bc290b2213e7f17ddd"

raw = open(C2P, "rb").read()
assert hashlib.sha256(raw).hexdigest() == C2_SHA
c2 = json.loads(raw)["cells"]
c2meta = json.load(open(os.path.join(ROOT, "our_submission_notebook", "submission_c2", "kernel-metadata.json")))

for d in sys.argv[1:] or [os.path.join(ROOT, "candidate_notebook", "submission_h961")]:
    name = os.path.basename(os.path.normpath(d)).replace("submission_", "")
    v = VARIANTS[name]
    ok = lambda m: print("OK  ", name.upper(), m)
    nb = json.load(open(os.path.join(d, "rsna-knee-%s.ipynb" % name)))["cells"]
    meta = json.load(open(os.path.join(d, "kernel-metadata.json")))

    assert len(nb) == (19 if v["students"] else 13) + (4 if v.get("partb") or v.get("mavit") else 0), len(nb)
    assert src(nb[0]).startswith("# RSNA Knee %s: C2" % name.upper()) and "57315983" in src(nb[0])
    want = [src(c) for c in c2]
    if v["v2"]:
        assert want[12].count(OLD) == 1
        want[12] = want[12].replace(OLD, V2)
        assert src(nb[12]).count(V2) == 1 and OLD not in src(nb[12])
    if v.get("bk"):   # cell 9: exactly one line replaced by foysal's 4 lines (from his notebook); the stacker adopts Baker's only
        assert want[9].count(BK_OLD) == 1 and BK_OLD not in src(nb[9]) and 'assert _sst_adopted == ["Baker\'s"]' in src(nb[9])
        want[9] = want[9].replace(BK_OLD, bk_new())
        if not v.get("fk"):   # BK alone: cell 9 is foysal's scored cell 9, byte for byte
            assert hashlib.sha256(src(nb[9]).encode()).hexdigest() == BK_CELL9_SHA
    if v.get("fk"):   # cell 9: the fluid-stacker line replaced by FK_NEW; the fluid stacker adopts Effusion only
        assert want[9].count(FK_OLD) == 1 and FK_OLD not in src(nb[9]) and 'assert _fst_adopted == ["Effusion"]' in src(nb[9])
        want[9] = want[9].replace(FK_OLD, FK_NEW)
    assert [src(c) for c in nb[1:13]] == want[1:13]
    assert [c["cell_type"] for c in nb[:13]] == [c["cell_type"] for c in c2]
    ok("cells 1-12 = C2's cells 1-12%s, same order and type" % "".join(t for t, f in (
        (" (cell 9: foysal's BK lines)", v.get("bk")), (" (cell 9: FK lines)", v.get("fk")), (" (cell 12: only W_RAPTOR_C -> V2)", v["v2"])) if f))

    for i, c in enumerate(nb):
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None, i
            compile("\n".join(l for l in src(c).splitlines() if not l.startswith(("%%", "!"))), "cell%d" % i, "exec")
    ok("all %d code cells compile, no stored outputs" % sum(c["cell_type"] == "code" for c in nb))
    t0 = [i for i, c in enumerate(nb) if re.search(r"^NB_T0 = ", src(c), re.M)]
    assert t0 == [4], t0

    if v["students"]:
        snap, writer, run, final = src(nb[13]), src(nb[15]), src(nb[16]), src(nb[18])
        assert snap.startswith("# [h960r] the parent notebook") and "C2 v2" in snap and "57315983" in snap
        assert ("V2 per-finding" in snap) == v["v2"]
        assert writer.startswith("# [h960r] the student sources") and run.startswith("# [h960r] the whole optional")
        assert final.startswith("# [h957] final:") and "E1_INVALID" in final
        assert [c["cell_type"] for c in nb[13:19]] == ["code", "markdown", "code", "code", "markdown", "code"]
        ok("after C2: snapshot (PARENT_LABEL says what the parent is) -> writer -> guarded run -> strict final")
        assert "if 9 * 3600 - (time.time() - NB_T0) < 50 * 60:" in run and "if 9 * 3600 - (time.time() - NB_T0) < 45 * 60:" in run
        assert "timeout=max(60, 9 * 3600 - (time.time() - NB_T0) - 20 * 60)" in run
        for k, h in CKPT_SHA256.items():
            assert run.count(h) == 1 and k in run, k
        ws = v.get("ws", 0.2)
        assert re.search(r"^W_PARENT, W_READER = %g, %g(\s|$)" % (round(1 - ws, 4), ws), final, re.M), "student weight"
        assert ("# [h957] final: %g x rank(" % round(1 - ws, 4)) in final and (") + %g x rank(our Mirror-student ensemble)" % ws) in final
        assert "_snap_sha() != BASELINE_SHA256" in final and "_fo.replace(_hw / '_h960_candidate.csv', _hw / 'submission.csv')" in final
        assert re.findall(r"copyfile\(([^)]*)\)", snap) == ["WORK / 'submission.csv', WORK / '_baseline_submission.csv'"]
        writes = [i for i, c in enumerate(nb[14:19], 14) if c["cell_type"] == "code" and "submission.csv'" in src(c).replace("_submission.csv'", "")]
        assert writes == [18], writes
        if v.get("partb"):   # Part B after our final cell: backup, heliosli's cell 9 + the timeout patch only, verify
            raw_pb = open(PARTB_SRC, "rb").read()
            assert hashlib.sha256(raw_pb).hexdigest() == PARTB_SHA
            pb0 = src(json.loads(raw_pb)["cells"][9])
            backup, pb, verify = src(nb[20]), src(nb[21]), src(nb[22])
            assert nb[19]["cell_type"] == "markdown" and backup.startswith("# [h965] backup") and verify.startswith("# [h965] verify")
            old_run = ('                  check=True, env=_o_env)\n')
            new_run = ('                  check=True, env=_o_env,   # [h965] never past 9 h: a timeout lands in the except below (file restored)\n'
                       '                  timeout=max(60, int(9 * 3600 - (_o_time.time() - globals().get("NB_T0", _o_time.time())) - 15 * 60)))\n')
            assert pb0.count(old_run) == 1 and pb == pb0.replace(old_run, new_run)
            assert "_h965_pre_partb.csv" in backup and "_kv_os.replace('/kaggle/working/_h965_restore.csv', '/kaggle/working/submission.csv')" in verify
            ok("Part B = heliosli v6 cell 9 + one timeout patch, after our final cell, between backup and verify cells")
        if v.get("mavit"):   # the MaViT leg after our final cell = exactly mavit_leg.mavit_cells(W)
            from mavit_leg import mavit_cells, CKPT_SHA
            want_cells, script = mavit_cells(v["mavit"])
            assert [src(c) for c in nb[19:23]] == [src(c) for c in want_cells] and [c["cell_type"] for c in nb[19:23]] == ["markdown"] + ["code"] * 3
            assert CKPT_SHA in src(nb[20]) and "W_MAVIT = %r" % v["mavit"] in src(nb[22]) and "_mr_left < 70 * 60" in src(nb[21])
            ok("MaViT leg = mavit_cells(%g) after our final cell: writer (backup + script with checkpoint sha256), runner, blend" % v["mavit"])
        ok("time guards on C2's NB_T0, 5 checkpoint sha256, 0.8/0.2, snapshot check, only the final cell replaces submission.csv")
        h964 = json.load(open(os.path.join(ROOT, "our_submission_notebook", "submission_h964", "rsna-knee-h964.ipynb")))["cells"]
        if v.get("tta"):   # H968: the writer also writes ours_leg/infer_tta.py; the run cell's reader block runs it, plain fallback
            import ast
            line = next(l for l in writer.splitlines() if l.startswith("_SOURCES = "))
            files = ast.literal_eval(line[len("_SOURCES = "):])
            assert sorted(files) == ["infer.py", "infer_tta.py", "knee.py", "preprocess.py"]
            assert files["infer_tta.py"] == open(os.path.join(ROOT, "ours_leg", "infer_tta.py")).read()
            line0 = next(l for l in src(h964[15]).splitlines() if l.startswith("_SOURCES = "))
            files0 = ast.literal_eval(line0[len("_SOURCES = "):])
            assert all(files[k] == files0[k] for k in files0) and writer.replace(line, line0) == src(h964[15])
            assert 'rc = _run_reader("infer_tta.py")' in run and 'rc = _run_reader("infer.py")' in run and "len(_P) < 200" in run
            assert run.count("_rsh.copyfile(\"/kaggle/working/reader_sub_plain.csv\", \"/kaggle/working/reader_sub.csv\")") == 1
            ok("student TTA: writer = H964's + infer_tta.py (repo file); reader block = TTA with plain fallback and the >= 200-study guard")
        elif v.get("s1"):   # H976: H964's student cells + the S1 lookup, 2 more sha256 pins, 7 instead of 5 checkpoints
            assert writer == src(h964[15])
            assert "_ck += _s1_ckpts(['cnxt_plE75s_full.pt', 'cnxt_plE75t_full.pt'])" in run and "    if len(_ck) != 7 or _bad_ck:" in run
            assert all(run.count(h) == 1 for h in S1_SHA.values()) and "    if len(_ck) != 5 or _bad_ck:" not in run
            assert "    if _ri.get('n_ckpt') != 7 or not _ri.get('ckpt_sha_ok'):" in final and "n_ckpt') != 5" not in final
            r0 = run[:run.index("    def _s1_ckpts(names):")] + run[run.index('    print("S1 students", _ck[-2:], flush=True)\n') + len('    print("S1 students", _ck[-2:], flush=True)\n'):]
            r0 = r0.replace("    if len(_ck) != 7 or _bad_ck:   # [s1] 5 Mirror + 2 S1 students\n", "    if len(_ck) != 5 or _bad_ck:\n")
            r0 = re.sub(r", 'cnxt_plE75s_full.pt': '[0-9a-f]{64}', 'cnxt_plE75t_full.pt': '[0-9a-f]{64}'\}   # \[s1\] \+ the 2 S1 students", "}", r0)
            assert r0 == src(h964[16]), "student run cell differs from H964 beyond the S1 lines"
            assert meta["kernel_sources"] == ["hosen42/rsna-knee-gpjw-student5"]
            ok("S1: H964's student cells + the S1 lookup (kernel source), 2 sha256 pins, 7 checkpoints in the run and final cells")
        elif v.get("v2"):
            assert writer == src(h964[15]) and run == src(h964[16]), "student cells differ from H964"

    want_sources = C2_SOURCES + ([READER_ASSETS, READERS] if v["students"] else []) + (["hengck23/hengck23-rnsa-knee"] if v.get("mavit") else [])
    assert meta["dataset_sources"] == want_sources, meta["dataset_sources"]
    assert meta["competition_sources"] == ["rsna-knee-abnormality-detection"] and meta["id"].endswith("/rsna-knee-" + name)
    assert meta["enable_gpu"] is True and meta["enable_internet"] is False and meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["is_private"] is True and meta["code_file"] == "rsna-knee-%s.ipynb" % name
    assert meta["docker_image"] == c2meta["docker_image"] and meta["dataset_sources"][:4] == c2meta["dataset_sources"]
    ok("metadata %s: %d datasets, C2's docker image, T4, internet off, private" % (meta["id"], len(want_sources)))
    print("ALL %s BUILD CHECKS PASSED" % name.upper())
