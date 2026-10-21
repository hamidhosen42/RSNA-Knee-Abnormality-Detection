"""Switches inside C2's R6 setup cell (cell 9), each replacing exactly one WORKER_ARGS line.

  BK  foysal's H961BK (sub 57417566, team best 10-20): the R6 worker gets no --state-stacker, so Baker's is the plain fold
      mean (+ flip).  His 4 lines are read from his archived notebook (sha256-checked), never retyped.
  FK  the same pattern for the fluid stacker: no --fluid-stacker, so Effusion is the plain fold mean (+ flip).  R6's own
      fluid_stacker.json records that Effusion was adopted "by user decision ... regardless of the pre-registered +0.003
      rule", which said DROP (explicit-only gain +0.0008, corrected +0.0009); the Baker's state stacker had a far larger CV
      gain (+0.0379 / +0.0077) and still lost on the public LB (BK).  The worker docstring: without --fluid-stacker the
      output is v3b's.  The assert below fails the build-time json check only if the stacker adopts anything but Effusion.
"""
import hashlib
import json

from paths import inp

BK_NB = "submission_h961bk/rsna-knee-h961bk.ipynb"   # foysalemonshanto/rsna-knee-h961bk v1 = sub 57417566
BK_NB_SHA = "c7a4c2fa7c80a8e6558caf582985d44873006d1afadfdc8c0947498c868ba9fa"
BK_CELL9_SHA = "cade3a089be4081281606b557c57ac1a973229b66ca5033db8eaebc674fd817c"   # sha256 of his joined cell 9

BK_OLD = "    WORKER_ARGS += ['--state-stacker', str(BUNDLE / manifest['state_stacker'])]\n"
FK_OLD = "    WORKER_ARGS += ['--fluid-stacker', str(BUNDLE / manifest['fluid_stacker'])]\n"
FK_NEW = (
    "    _fst = json.loads((BUNDLE / manifest['fluid_stacker']).read_text(encoding='utf-8'))   # [fk 10-20] fluid stacker (adopts only Effusion) NOT passed: Effusion = the plain fold mean (+ flip)\n"
    "    _fst_adopted = sorted(k for k, r in _fst['classes'].items() if r.get('adopt'))\n"
    "    assert _fst_adopted == [\"Effusion\"], ('the fluid stacker adopts more than Effusion', _fst_adopted)\n"
    "    print('FLUID_STACKER_OFF adopted-by-json', _fst_adopted, 'sha256', hashlib.sha256((BUNDLE / manifest['fluid_stacker']).read_bytes()).hexdigest()[:16], flush=True)   # --fluid-stacker intentionally not passed\n")


def bk_new():
    """foysal's 4 BK lines, cut from his archived, sha-checked notebook (cell 9)."""
    raw = open(inp(BK_NB), "rb").read()
    assert hashlib.sha256(raw).hexdigest() == BK_NB_SHA, "foysal's archived H961BK notebook changed"
    c9 = "".join(json.loads(raw)["cells"][9]["source"])
    assert hashlib.sha256(c9.encode()).hexdigest() == BK_CELL9_SHA
    i = c9.index("    _sst = json.loads(")
    j = c9.index("--state-stacker intentionally not passed\n") + len("--state-stacker intentionally not passed\n")
    return c9[i:j]
