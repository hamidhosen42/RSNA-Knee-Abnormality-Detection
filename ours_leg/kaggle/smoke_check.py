"""Smoke-test verdict for one of our commit runs: stage markers, final blend receipt, student leg, submission.csv shape.

    python -I ours_leg/kaggle/smoke_check.py <slug> <out_dir> [owner=hosen42]     (read-only API calls, ~/.kaggle/access_token)
"""
import json
import os
import sys
import urllib.parse
import urllib.request

slug, out = sys.argv[1], sys.argv[2]
owner = sys.argv[3] if len(sys.argv) > 3 else "hosen42"
os.makedirs(out, exist_ok=True)
tok = open(os.path.expanduser("~/.kaggle/access_token")).read().strip()


def get(path, q):
    r = urllib.request.Request("https://www.kaggle.com/api/v1/" + path + "?" + urllib.parse.urlencode(q), headers={"Authorization": "Bearer " + tok})
    return json.load(urllib.request.urlopen(r, timeout=120))


q = {"userName": owner, "kernelSlug": slug}
st = get("kernels/status", q)
o = get("kernels/output", q)
log = o.get("log") or ""
open(os.path.join(out, slug + ".log.json"), "w").write(log if isinstance(log, str) else json.dumps(log))
try:
    ev = json.loads(log) if isinstance(log, str) else log
    lines = [(e.get("time", 0), l) for e in ev for l in (e.get("data") or "").splitlines()]
except Exception:
    lines = [(0, l) for l in str(log).splitlines()]
files = {f.get("fileName"): f for f in o.get("files", [])}
print("status", st.get("status"), st.get("failureMessage") or "", "| log lines", len(lines), "| last t %.0f s" % (lines[-1][0] if lines else 0))
keys = ["STAGE_DONE", "OURS_STAGE", "NOTEBOOK_COMPLETE", "BLEND_FALLBACK", "PARENT_SNAPSHOT", "READER_", "H960_FINAL", "RECEIPT_WRITE_FAILED",
        "own leg", "Part B", "MAVIT", "EXTRA_", "Traceback", "Error", "error:", "FAILED"]
for t_, l in lines:
    if any(k in l for k in keys) and "XLA unavailable" not in l:
        print("%6.0f | %s" % (t_, l[:900]))
sub = files.get("submission.csv")
if sub and sub.get("url"):
    data = urllib.request.urlopen(urllib.request.Request(sub["url"]), timeout=120).read().decode()
    open(os.path.join(out, slug + ".submission.csv"), "w").write(data)
    rows = data.strip().splitlines()
    print("submission.csv rows", len(rows) - 1, "| cols", len(rows[0].split(",")), "|", rows[0][:160])
else:
    print("submission.csv listed:", bool(sub))
