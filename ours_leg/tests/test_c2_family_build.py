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
VARIANTS = {"h961": dict(v2=False, students=True), "h963": dict(v2=True, students=False), "h964": dict(v2=True, students=True)}

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

    assert len(nb) == (19 if v["students"] else 13), len(nb)
    assert src(nb[0]).startswith("# RSNA Knee %s: C2" % name.upper()) and "57315983" in src(nb[0])
    want = [src(c) for c in c2]
    if v["v2"]:
        assert want[12].count(OLD) == 1
        want[12] = want[12].replace(OLD, V2)
        assert src(nb[12]).count(V2) == 1 and OLD not in src(nb[12])
    assert [src(c) for c in nb[1:13]] == want[1:13]
    assert [c["cell_type"] for c in nb[:13]] == [c["cell_type"] for c in c2]
    ok("cells 1-12 = C2's cells 1-12%s, same order and type" % (" (cell 12: only W_RAPTOR_C -> V2)" if v["v2"] else ""))

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
        assert [c["cell_type"] for c in nb[13:]] == ["code", "markdown", "code", "code", "markdown", "code"]
        ok("after C2: snapshot (PARENT_LABEL says what the parent is) -> writer -> guarded run -> strict final")
        assert "if 9 * 3600 - (time.time() - NB_T0) < 50 * 60:" in run and "if 9 * 3600 - (time.time() - NB_T0) < 45 * 60:" in run
        assert "timeout=max(60, 9 * 3600 - (time.time() - NB_T0) - 20 * 60)" in run
        for k, h in CKPT_SHA256.items():
            assert run.count(h) == 1 and k in run, k
        assert "W_PARENT, W_READER = 0.8, 0.2" in final
        assert "_snap_sha() != BASELINE_SHA256" in final and "_fo.replace(_hw / '_h960_candidate.csv', _hw / 'submission.csv')" in final
        assert re.findall(r"copyfile\(([^)]*)\)", snap) == ["WORK / 'submission.csv', WORK / '_baseline_submission.csv'"]
        writes = [i for i, c in enumerate(nb[14:], 14) if c["cell_type"] == "code" and "submission.csv'" in src(c).replace("_submission.csv'", "")]
        assert writes == [18], writes
        ok("time guards on C2's NB_T0, 5 checkpoint sha256, 0.8/0.2, snapshot check, only the final cell replaces submission.csv")

    want_sources = C2_SOURCES + ([READER_ASSETS, READERS] if v["students"] else [])
    assert meta["dataset_sources"] == want_sources, meta["dataset_sources"]
    assert meta["competition_sources"] == ["rsna-knee-abnormality-detection"] and meta["id"].endswith("/rsna-knee-" + name)
    assert meta["enable_gpu"] is True and meta["enable_internet"] is False and meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["is_private"] is True and meta["code_file"] == "rsna-knee-%s.ipynb" % name
    assert meta["docker_image"] == c2meta["docker_image"] and meta["dataset_sources"][:4] == c2meta["dataset_sources"]
    ok("metadata %s: %d datasets, C2's docker image, T4, internet off, private" % (meta["id"], len(want_sources)))
    print("ALL %s BUILD CHECKS PASSED" % name.upper())
