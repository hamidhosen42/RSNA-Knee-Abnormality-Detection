"""Poll our kernels until one reaches a terminal state; one line per state change, then exit.

    python -I ours_leg/kaggle/wait_kernels.py <slug> [<slug> ...]      (owner hosen42)
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

tok = open(os.path.expanduser("~/.kaggle/access_token")).read().strip()
last = {}
while True:
    done = []
    for s in sys.argv[1:]:
        try:
            r = urllib.request.Request("https://www.kaggle.com/api/v1/kernels/status?" + urllib.parse.urlencode({"userName": "hosen42", "kernelSlug": s}),
                                       headers={"Authorization": "Bearer " + tok})
            v = json.load(urllib.request.urlopen(r, timeout=60)).get("status")
        except Exception as e:
            v = "poll-error " + type(e).__name__
        if v != last.get(s):
            print(time.strftime("%H:%M", time.gmtime()), s, v, flush=True)
            last[s] = v
        if v in ("complete", "error", "cancelAcknowledged", "cancelRequested"):
            done.append(s)
    if done:
        break
    time.sleep(45)
