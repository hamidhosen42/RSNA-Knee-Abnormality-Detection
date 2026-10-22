"""End-to-end check of the family-balanced student leg (H982 / H983) on the Kaggle smoke outputs (3 public studies, GPU,
read-only).  A 5-student run's reader_sub.csv is the Mirror mean m and an S1 run's is the equal 7-checkpoint mean e7, so the
S1 mean is s = (7 e7 - 5 m) / 2 and a fam run's reader_sub.csv must equal 0.5 m + 0.5 s = (7 e7 - 3 m) / 4 (up to fp16 noise).

    python -I ours_leg/kaggle/fam_check.py <5-student slug> <7-checkpoint slug> <fam slug> <out dir>     (owner hosen42)
"""
import csv
import json
import os
import sys
import urllib.parse
import urllib.request

*slugs, out = sys.argv[1:]
os.makedirs(out, exist_ok=True)
tok = open(os.path.expanduser("~/.kaggle/access_token")).read().strip()


def reader(slug):
    r = urllib.request.Request("https://www.kaggle.com/api/v1/kernels/output?" + urllib.parse.urlencode({"userName": "hosen42", "kernelSlug": slug}),
                               headers={"Authorization": "Bearer " + tok})
    f = next(f for f in json.load(urllib.request.urlopen(r, timeout=120))["files"] if f["fileName"] == "reader_sub.csv")
    p = os.path.join(out, slug + ".reader_sub.csv")
    urllib.request.urlretrieve(f["url"], p)
    rows = list(csv.reader(open(p)))
    return rows[0], {row[0]: [float(v) for v in row[1:]] for row in rows[1:]}


(h0, m), (h1, e7), (h2, fam) = (reader(s) for s in slugs)
assert h0 == h1 == h2 and sorted(m) == sorted(e7) == sorted(fam), "reader files do not line up"
want = {k: [(7 * a - 3 * b) / 4 for a, b in zip(e7[k], m[k])] for k in m}
d_want = max(abs(x - y) for k in m for x, y in zip(fam[k], want[k]))
d_e7 = max(abs(x - y) for k in m for x, y in zip(fam[k], e7[k]))
d_s1 = max(abs(x - y) for k in m for x, y in zip(e7[k], m[k]))
print("studies %d x %d findings | max |fam - (7 e7 - 3 m) / 4| = %.2e | max |fam - e7| = %.4f | max |e7 - m| = %.4f"
      % (len(m), len(h0) - 1, d_want, d_e7, d_s1))
print("FAM_CHECK", "OK" if d_want < 2e-3 and d_e7 > 10 * d_want else "FAILED")
