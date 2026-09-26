"""Clone every topic of the competition discussion forum into one Markdown file per topic.

Stdlib only. Re-runnable: topics whose comment activity has not changed since the last run
are not re-downloaded; all Markdown is regenerated from the saved raw JSON.

    python3 clone_discussions.py            # sync (new/changed topics only)
    python3 clone_discussions.py --full     # re-download every topic

Auth: reads the Kaggle access token from ~/.kaggle/access_token at runtime and sends it as a
Bearer header. The token is never written anywhere.
"""
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

COMPETITION = "rsna-knee-abnormality-detection"
FORUM_ID = 12044884  # competitions.CompetitionService/GetCompetition -> forumId
API = "https://www.kaggle.com/api/i/"
BASE = "https://www.kaggle.com"
OUT = Path(__file__).resolve().parents[1]  # .../discussions
RAW = OUT / "_raw"
ASSETS = OUT / "_assets"
PAUSE_S = 0.8
MAX_ASSET_BYTES = 25 << 20
# Markdown link destinations may contain balanced parentheses, e.g. ".../3n%20(2).png?generation=..."
IMG_MD = re.compile(r"(!\[[^\]]*\]\()(\s*<?)(https?://(?:[^\s()<>]|\([^\s()<>]*\))+)(>?)")
IMG_HTML = re.compile(r"""(<img\b[^>]*?\bsrc=["'])(https?://[^"']+)(["'])""", re.IGNORECASE)
_asset_cache = {}

TOKEN = (Path.home() / ".kaggle" / "access_token").read_text().strip()


def call(method, body, tries=6):
    data = json.dumps(body).encode()
    for attempt in range(tries):
        req = urllib.request.Request(
            API + method, data=data, method="POST",
            headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json",
                     "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < tries - 1:
                wait = 5 * 2 ** attempt
                print(f"  HTTP {exc.code} on {method}; retry in {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise
        except urllib.error.URLError:
            if attempt < tries - 1:
                time.sleep(5 * 2 ** attempt)
                continue
            raise


def list_topics():
    """All topics, paged by post date (stable under new comments). Deduplicated by id."""
    topics, page, expected = {}, 1, None
    while True:
        res = call("discussions.DiscussionsService/GetTopicListByForumId",
                   {"forumId": FORUM_ID, "page": page, "sortBy": "TOPIC_LIST_SORT_BY_NEW"})
        expected = res.get("count", expected)
        batch = res.get("topics", [])
        if not batch:
            break
        for t in batch:
            topics[t["id"]] = t
        page += 1
        time.sleep(PAUSE_S)
        if page > 200:
            raise RuntimeError("pagination did not terminate")
    return list(topics.values()), expected


def slugify(text, limit=70):
    s = re.sub(r"[^A-Za-z0-9]+", "-", text or "").strip("-").lower()
    return (s[:limit].rstrip("-") or "topic")


def fmt_date(value):
    if not value:
        return ""
    try:
        v = re.sub(r"(\.\d{6})\d*", r"\1", value.replace("Z", "+00:00"))
        return datetime.fromisoformat(v).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return value


def author_line(msg):
    a = msg.get("author") or {}
    name = a.get("displayName") or "(unknown)"
    handle = (a.get("url") or "").strip("/")
    tier = (a.get("tier") or "").title()
    bits = [f"**{name}**"]
    meta = ", ".join(x for x in (f"@{handle}" if handle else "", tier) if x)
    if meta:
        bits.append(f"({meta})")
    role = {"HOST": "competition host", "TOPIC": "topic author"}.get(msg.get("authorType"))
    if role:
        bits.append(f"· _{role}_")
    bits.append(f"· {fmt_date(msg.get('postDate'))}")
    votes = (msg.get("votes") or {}).get("totalVotes")
    if votes:
        bits.append(f"· ▲{votes}")
    rank = msg.get("competitionRanking")
    if rank:
        bits.append(f"· LB rank {rank}")
    return " ".join(bits)


def fetch_asset(url):
    """Download an embedded image once (no Kaggle token: it only goes to the Kaggle API)."""
    if url in _asset_cache:
        return _asset_cache[url]
    import hashlib
    import mimetypes
    stem = hashlib.sha1(url.encode()).hexdigest()[:16]
    existing = sorted(ASSETS.glob(stem + ".*"))
    if existing:
        _asset_cache[url] = existing[0].name
        return existing[0].name
    local = None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read(MAX_ASSET_BYTES + 1)
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip()
        if data and len(data) <= MAX_ASSET_BYTES:
            ext = Path(urllib.parse.urlparse(url).path).suffix.lower()
            if not re.fullmatch(r"\.(png|jpe?g|gif|webp|svg|bmp)", ext or ""):
                ext = mimetypes.guess_extension(ctype) or ".bin"
            ASSETS.mkdir(parents=True, exist_ok=True)
            (ASSETS / (stem + ext)).write_bytes(data)
            local = stem + ext
            time.sleep(0.2)
    except Exception as exc:  # keep the remote link if the image cannot be fetched
        print(f"  asset not downloaded ({type(exc).__name__}): {url[:100]}", flush=True)
    _asset_cache[url] = local
    return local


def localize(text):
    def md_sub(m):
        local = fetch_asset(m.group(3))
        return m.group(1) + m.group(2) + (f"_assets/{local}" if local else m.group(3)) + m.group(4)

    def html_sub(m):
        local = fetch_asset(m.group(2))
        return m.group(1) + (f"_assets/{local}" if local else m.group(2)) + m.group(3)
    return IMG_HTML.sub(html_sub, IMG_MD.sub(md_sub, text))


def body(msg):
    text = (msg.get("rawMarkdown") or "").strip()
    if not text:
        text = (msg.get("content") or "").strip()  # HTML fallback; Markdown renders it
    return localize(text) if text else "_(no content — deleted or empty)_"


def count_messages(comments):
    return sum(1 + count_messages(c.get("replies") or []) for c in comments)


def render_comments(comments, prefix, depth, out):
    ordered = sorted(comments, key=lambda c: c.get("postDate") or "")
    for i, c in enumerate(ordered, 1):
        num = f"{prefix}{i}"
        level = min(3 + depth, 6)
        arrow = "↳ " * depth
        out.append(f"{'#' * level} {arrow}[{num}] {(c.get('author') or {}).get('displayName', '(unknown)')}")
        out.append("")
        out.append(author_line(c))
        out.append("")
        out.append(body(c))
        out.append("")
        render_comments(c.get("replies") or [], num + ".", depth + 1, out)


def render_topic(listing, detail):
    ft = detail["forumTopic"]
    first = ft.get("firstMessage") or {}
    comments = ft.get("comments") or []
    n_comments = count_messages(comments)
    url = BASE + (ft.get("url") or listing.get("topicUrl", ""))
    lines = [f"# {ft.get('name') or listing.get('title')}", ""]
    lines += [
        f"- **URL:** {url}",
        f"- **Author:** {author_line(first)}",
        f"- **Topic votes:** {ft.get('totalVotes', 0)} · **Comments/replies:** {n_comments}"
        f" · **Pinned:** {'yes' if listing.get('isSticky') else 'no'}",
        f"- **Last comment:** {fmt_date(listing.get('lastCommentPostDate'))}"
        + (f" by {listing.get('lastCommenterName')}" if listing.get("lastCommenterName") else ""),
        f"- **Cloned:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        f" (topic id {ft.get('id')}, raw JSON `_raw/{ft.get('id')}.json`)",
        "", "---", "", body(first), "", "---", "",
        f"## Comments ({n_comments}, threaded, oldest first)", "",
    ]
    if not comments:
        lines.append("_No comments._")
    render_comments(comments, "", 0, lines)
    return "\n".join(lines).rstrip() + "\n"


def main():
    full = "--full" in sys.argv
    RAW.mkdir(parents=True, exist_ok=True)
    topics, expected = list_topics()
    print(f"listed {len(topics)} topics (forum reports {expected})", flush=True)
    if expected is not None and len(topics) != expected:
        print("WARNING: listed topic count differs from the forum count", flush=True)
    fetched = skipped = 0
    for t in sorted(topics, key=lambda x: x["id"]):
        raw_path = RAW / f"{t['id']}.json"
        if raw_path.exists() and not full:
            saved = json.loads(raw_path.read_text())
            if (saved.get("_listing", {}).get("lastCommentPostDate") == t.get("lastCommentPostDate")
                    and saved.get("_listing", {}).get("commentCount") == t.get("commentCount")):
                skipped += 1
                continue
        detail = call("discussions.DiscussionsService/GetForumTopicById",
                      {"forumTopicId": t["id"], "includeComments": True})
        detail["_listing"] = t
        raw_path.write_text(json.dumps(detail, ensure_ascii=False, indent=1))
        fetched += 1
        print(f"  fetched {t['id']}  {t.get('title', '')[:70]}", flush=True)
        time.sleep(PAUSE_S)
    print(f"fetched {fetched}, unchanged {skipped}", flush=True)

    # Regenerate every Markdown file and the indexes from raw JSON.
    for old in OUT.glob("*.md"):
        if re.fullmatch(r"\d+_.*\.md", old.name):  # only files this script generates
            old.unlink()
    rows = []
    for raw_path in sorted(RAW.glob("*.json")):
        detail = json.loads(raw_path.read_text())
        listing = detail.get("_listing", {})
        ft = detail["forumTopic"]
        title = ft.get("name") or listing.get("title") or ""
        name = f"{ft['id']}_{slugify(title)}.md"
        (OUT / name).write_text(render_topic(listing, detail))
        first = ft.get("firstMessage") or {}
        rows.append({
            "topic_id": ft["id"], "title": title, "file": name,
            "url": BASE + (ft.get("url") or ""),
            "author": (first.get("author") or {}).get("displayName", ""),
            "author_handle": ((first.get("author") or {}).get("url") or "").strip("/"),
            "author_type": listing.get("authorType", ""),
            "posted": listing.get("postDate", ft.get("postDate", "")),
            "last_comment": listing.get("lastCommentPostDate", ""),
            "comments": count_messages(ft.get("comments") or []),
            "votes": ft.get("totalVotes", 0),
            "pinned": bool(listing.get("isSticky")),
        })
    # Same order as the forum's "recent comments" view: pinned first, then latest activity.
    rows.sort(key=lambda r: (r["pinned"], r["last_comment"] or r["posted"]), reverse=True)
    with (OUT / "_index.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    total_msgs = sum(r["comments"] for r in rows)
    md = [
        f"# {COMPETITION} — discussion archive",
        "",
        f"Cloned from https://www.kaggle.com/competitions/{COMPETITION}/discussion "
        f"on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.",
        f"{len(rows)} topics, {total_msgs} comments/replies. One file per topic; raw API JSON in `_raw/`; "
        "embedded images downloaded to `_assets/` (original URLs stay in the raw JSON).",
        "Order: pinned topics first, then by most recent comment (like the forum's recent-comments view).",
        "Re-sync with `python3 _tools/clone_discussions.py`.",
        "",
        "| # | Last activity (UTC) | Topic | Author | Comments | Votes |",
        "|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        pin = "📌 " if r["pinned"] else ""
        title = r["title"].replace("|", "\\|")
        md.append(f"| {i} | {fmt_date(r['last_comment'] or r['posted'])} | {pin}[{title}]({r['file']}) "
                  f"| {r['author'].replace('|', '/')}{' (host)' if r['author_type'] == 'HOST' else ''} "
                  f"| {r['comments']} | {r['votes']} |")
    (OUT / "README.md").write_text("\n".join(md) + "\n")
    print(f"wrote {len(rows)} topic files + README.md + _index.csv to {OUT}", flush=True)


if __name__ == "__main__":
    main()
