"""Rebuild our_submission_notebook/_index.csv: one row per folder (submission notebooks + training kernels), with the
Kaggle kernel, who ran it, the team's submission ids / public scores for that kernel slug, and the description.

    python ours_leg/update_our_index.py        (needs ~/.kaggle/access_token; read-only API calls)
"""
import csv
import glob
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import NB, TRAINING  # noqa: E402

tok = open(os.path.expanduser("~/.kaggle/access_token")).read().strip()
req = urllib.request.Request("https://www.kaggle.com/api/v1/competitions/submissions/list/rsna-knee-abnormality-detection?pageSize=500",
                             headers={"Authorization": "Bearer " + tok})   # `page=` is ignored by this endpoint; pageSize works
subs = sorted({s["ref"]: s for s in json.load(urllib.request.urlopen(req, timeout=60))}.values(), key=lambda s: s["date"])
by_slug = {}
for s in subs:
    m = re.match(r"/code/([^/]+)/([^?]+)", s.get("url") or "")
    if m:
        by_slug.setdefault(m.group(2), []).append(s)


def kernel_id(d):
    p = os.path.join(d, "kernel-metadata.json")
    return json.load(open(p))["id"] if os.path.exists(p) else ""


rows = []
for d in sorted(glob.glob(os.path.join(NB, "submission_*"))):
    k = kernel_id(d)
    ss = by_slug.get(k.split("/")[-1], []) if k else []
    scored = [s for s in ss if s.get("publicScore")]
    rows.append({"folder": os.path.relpath(d, NB), "kind": "submission notebook",
                 "files": ";".join(sorted(os.path.basename(x) for x in glob.glob(d + "/*.ipynb"))), "kaggle_kernel": k,
                 "run_by": ";".join(sorted({s["submittedBy"] for s in ss})), "submission_ids": ";".join(str(s["ref"]) for s in ss),
                 "public_score": ";".join(sorted({s["publicScore"] for s in scored}, reverse=True)) or ("error" if ss else "not submitted"),
                 "submitted": ss[0]["date"][:10] if ss else "", "description": ss[-1]["description"][:150] if ss else ""})
for d in sorted(glob.glob(os.path.join(TRAINING, "*"))):
    k = kernel_id(d)
    rows.append({"folder": os.path.relpath(d, NB), "kind": "colab training notebook (not used)" if d.endswith("colab") else "training / teacher / cache kernel",
                 "files": ";".join(sorted(os.path.basename(x) for x in glob.glob(d + "/*") if not x.endswith("kernel-metadata.json"))),
                 "kaggle_kernel": k, "run_by": k.split("/")[0] if k else "", "submission_ids": "", "public_score": "", "submitted": "", "description": ""})
with open(os.path.join(NB, "_index.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("index rows:", len(rows), "| team submissions read:", len(subs))
