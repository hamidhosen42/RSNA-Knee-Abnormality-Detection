"""Where our files live: the single place every builder takes its paths from.

ROOT      the repository
NB        our_submission_notebook/            our own notebooks, as pushed / submitted (an archive)
TRAINING  our_submission_notebook/training/   our training, teacher and cache kernels
PUBLIC    public_notebooks/                   public notebooks (never ours)

One rule for every path argument of every builder:
  inp(p)      an absolute path is used as is; a path that starts with our_submission_notebook/ or public_notebooks/,
              or that exists from the repo root, is relative to ROOT; anything else (e.g. submission_h957x5/x.ipynb)
              is relative to NB (or the base given).
  out_dir(p)  an absolute path is used as is; our_submission_notebook/... is relative to ROOT; a bare folder name
              (submission_h958b) goes under NB (or the base given); any other relative path (../x/submission_h958b) is
              relative to the current directory.
write_text / write_json never replace an existing file with different content unless OVERWRITE=1: the archive holds
what was pushed, and a newer builder must not silently replace it.  Several builders evolved after their pushes
(e.g. build_h957, build_e21_plreader, make_pl_submission, make_student_kernel round 1, make_train_kernel), so
rebuilding them with default arguments is refused instead of overwriting the archived notebook.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "our_submission_notebook")
TRAINING = os.path.join(NB, "training")
PUBLIC = os.path.join(ROOT, "public_notebooks")
_ROOT_PREFIXES = ("our_submission_notebook/", "public_notebooks/")


def inp(p, base=NB):
    if os.path.isabs(p):
        return p
    if p.startswith(_ROOT_PREFIXES) or os.path.exists(os.path.join(ROOT, p)):
        return os.path.join(ROOT, p)
    return os.path.join(base, p)


def out_dir(p, base=NB):
    if os.path.isabs(p):
        return p
    if p.startswith(_ROOT_PREFIXES):
        return os.path.join(ROOT, p)
    if "/" not in os.path.normpath(p):
        return os.path.join(base, p)
    return os.path.abspath(p)


def write_text(path, text):
    if os.path.exists(path) and os.environ.get("OVERWRITE") != "1":
        with open(path, encoding="utf-8") as fh:
            if fh.read() != text:
                raise SystemExit(f"refusing to change {os.path.relpath(path, ROOT)}: it holds what was pushed. "
                                 "Use another --out (or folder), or OVERWRITE=1 if you really mean to replace it.")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def write_json(path, obj, indent=1, **kw):
    write_text(path, json.dumps(obj, indent=indent, **kw))
