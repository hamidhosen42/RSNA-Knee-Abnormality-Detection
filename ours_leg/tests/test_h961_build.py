"""Structure of the built H961 (C2 v2 + our 5 students): the C2 parent is intact and runs first, the student leg is the
hardened H960r one, the metadata mounts what the cells read.  Pure stdlib; run from the repo root after the build:

    python ours_leg/build_h960.py --parent c2 --students --robust --owner kishormorol
    python -I ours_leg/tests/test_h961_build.py [candidate_notebook/submission_h961]
"""
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "ours_leg"))
from student_leg import CKPT_SHA256, READER_ASSETS, READERS  # noqa: E402

d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "candidate_notebook", "submission_h961")
src = lambda c: "".join(c["source"])
C2P = os.path.join(ROOT, "our_submission_notebook", "submission_c2", "rsna-knee-sub-c2.ipynb")
C2_SHA = "148eb4387de1aac2ccd374a8e052cdf759a0962f52c40ac06e6abe13af2942d7"
ok = lambda m: print("OK  ", m)

raw = open(C2P, "rb").read()
assert hashlib.sha256(raw).hexdigest() == C2_SHA
c2 = json.loads(raw)["cells"]
nb = json.load(open(os.path.join(d, "rsna-knee-h961.ipynb")))["cells"]
meta = json.load(open(os.path.join(d, "kernel-metadata.json")))
ok("parent = archived C2 v2 (sha256 %s...)" % C2_SHA[:12])

assert len(nb) == 19, len(nb)
assert src(nb[0]).startswith("# RSNA Knee H961: C2") and "57315983" in src(nb[0])
assert [src(c) for c in nb[1:13]] == [src(c) for c in c2[1:13]]
assert [c["cell_type"] for c in nb[:13]] == [c["cell_type"] for c in c2]
ok("cells 1-12 = C2's cells 1-12, same order and type (only the title cell 0 differs)")

snap, writer, run, final = src(nb[13]), src(nb[15]), src(nb[16]), src(nb[18])
assert snap.startswith("# [h960r] the parent notebook") and "C2 v2" in snap and "57315983" in snap
assert writer.startswith("# [h960r] the student sources") and run.startswith("# [h960r] the whole optional")
assert final.startswith("# [h957] final:") and "E1_INVALID" in final
assert [c["cell_type"] for c in nb[13:]] == ["code", "markdown", "code", "code", "markdown", "code"]
ok("after C2: snapshot (PARENT_LABEL = C2) -> writer -> guarded run -> strict final")

for i, c in enumerate(nb):
    if c["cell_type"] == "code":
        assert c["outputs"] == [] and c["execution_count"] is None, i
        compile("\n".join(l for l in src(c).splitlines() if not l.startswith(("%%", "!"))), "cell%d" % i, "exec")
ok("all %d code cells compile, no stored outputs" % sum(c["cell_type"] == "code" for c in nb))

t0 = [i for i, c in enumerate(nb) if re.search(r"^NB_T0 = ", src(c), re.M)]
assert t0 == [4], t0
assert "if 9 * 3600 - (time.time() - NB_T0) < 50 * 60:" in run and "if 9 * 3600 - (time.time() - NB_T0) < 45 * 60:" in run
assert "timeout=max(60, 9 * 3600 - (time.time() - NB_T0) - 20 * 60)" in run
ok("time guards read C2's NB_T0 (set once, cell 4): skip at < 50 min, < 45 min before inference, timeout = 9 h - elapsed - 20 min")

for k, v in CKPT_SHA256.items():
    assert run.count(v) == 1 and k in run, k
ok("the run cell checks the sha256 of all 5 student checkpoints")

assert "W_PARENT, W_READER = 0.8, 0.2" in final or ("W_PARENT = 0.8" in final and "W_READER = 0.2" in final), "weights"
assert "_snap_sha() != BASELINE_SHA256" in final and "_fo.replace(_hw / '_h960_candidate.csv', _hw / 'submission.csv')" in final
ok("final: 0.8 x rank(C2 snapshot) + 0.2 x rank(students); snapshot sha check; atomic replace")

assert re.findall(r"copyfile\(([^)]*)\)", snap) == ["WORK / 'submission.csv', WORK / '_baseline_submission.csv'"]   # reads it only
writes = [i for i, c in enumerate(nb[14:], 14) if c["cell_type"] == "code" and "submission.csv'" in src(c).replace("_submission.csv'", "")]
assert writes == [18], writes
ok("after the snapshot only the final cell can replace submission.csv")

want = ["heliosli/rsna-knee-r6-frozen-fe055406-w1", "nartaa/rsna-knee-s6fall-weights", "nartaa/rsna-knee-publication-swa-weights-20261007",
        "heliosli/rsna-orthofoundation-fold0-epoch7-20261004", READER_ASSETS, READERS]
assert meta["dataset_sources"] == want, meta["dataset_sources"]
assert meta["competition_sources"] == ["rsna-knee-abnormality-detection"] and meta["id"].endswith("/rsna-knee-h961")
assert meta["enable_gpu"] is True and meta["enable_internet"] is False and meta["machine_shape"] == "NvidiaTeslaT4"
assert meta["is_private"] is True and meta["code_file"] == "rsna-knee-h961.ipynb"
c2meta = json.load(open(os.path.join(ROOT, "our_submission_notebook", "submission_c2", "kernel-metadata.json")))
assert meta["docker_image"] == c2meta["docker_image"] and meta["dataset_sources"][:4] == c2meta["dataset_sources"]
ok("metadata %s: C2's 4 datasets + %s + %s, C2's docker image, T4, internet off, private" % (meta["id"], READER_ASSETS, READERS))
print("ALL H961 BUILD CHECKS PASSED")
