"""Poll the team's submissions until every listed ref is scored or failed; print changes, then our leaderboard row.

    python -I ours_leg/kaggle/wait_subs.py <ref> [<ref> ...]
"""
import json
import os
import sys
import time
import urllib.request

tok = open(os.path.expanduser("~/.kaggle/access_token")).read().strip()
refs = {int(x) for x in sys.argv[1:]}
last = {}
while True:
    try:
        r = urllib.request.Request("https://www.kaggle.com/api/v1/competitions/submissions/list/rsna-knee-abnormality-detection?pageSize=40",
                                   headers={"Authorization": "Bearer " + tok})
        cur = {x["ref"]: (x.get("status"), x.get("publicScore"), (x.get("errorDescription") or "")[:160]) for x in json.load(urllib.request.urlopen(r, timeout=60)) if x["ref"] in refs}
    except Exception as e:
        print(time.strftime("%H:%M", time.gmtime()), "poll error", type(e).__name__, flush=True)
        time.sleep(300)
        continue
    for k, v in cur.items():
        if v != last.get(k):
            print(time.strftime("%H:%M", time.gmtime()), k, *v, flush=True)
            last[k] = v
    if len(cur) == len(refs) and all("pending" not in str(v[0]).lower() for v in cur.values()):
        break
    time.sleep(600)
H = {"Authorization": "Bearer " + tok, "Content-Type": "application/json"}
lb = json.load(urllib.request.urlopen(urllib.request.Request("https://www.kaggle.com/api/i/competitions.LeaderboardService/GetLeaderboard",
               data=json.dumps({"competitionId": 154281, "leaderboardMode": "LEADERBOARD_MODE_DEFAULT"}).encode(), headers=H, method="POST"), timeout=120))
rows = lb.get("publicLeaderboard") or []
print("TEAM ROW", next(x for x in rows if x.get("teamId") == 16669023), "of", len(rows), flush=True)
