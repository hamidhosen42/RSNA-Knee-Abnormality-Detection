"""The S1 student checkpoints must load into the SAME KneeNet our notebook's knee.py builds (strict), and run a forward pass,
before they are added to the student mean (a load failure on Kaggle would drop the whole student leg).  CPU only.

    python -I ours_leg/tests/tta/extract_sources.py <notebook.ipynb> <src dir>
    uv run --no-project --with numpy --with torch --with timm --with pandas --with pydicom --with opencv-python-headless \
        python -I ours_leg/tests/test_s1_ckpt_load.py <src dir> <ckpt> [<ckpt> ...]
"""
import hashlib
import sys

src_dir, ckpts = sys.argv[1], sys.argv[2:]
sys.path.insert(0, src_dir)
import torch  # noqa: E402

from infer import load_models  # noqa: E402   the notebook's own loader (strict load_state_dict)
from knee import N_SLOTS  # noqa: E402

for p in ckpts:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    a = ck["args"]
    (m, res), = load_models([p], device="cpu")
    k = 4
    x = torch.randint(0, 255, (1, N_SLOTS, k, 3, res, res), dtype=torch.uint8)
    mask = torch.ones(1, N_SLOTS, dtype=torch.bool)
    pos = torch.linspace(0, 1, k).repeat(1, N_SLOTS, 1)
    with torch.no_grad():
        out = m(x, mask, pos, res=res).float().sigmoid()
    assert out.shape == (1, 12) and torch.isfinite(out).all(), out.shape
    print("OK  %-24s sha256 %s | backbone %s res %s epoch %s | forward %s" % (p.split("/")[-1], h.hexdigest(), a.get("backbone"), a.get("res"),
          ck.get("epoch"), [round(float(v), 3) for v in out[0][:4]]))
print("ALL S1 CHECKPOINTS LOAD AND RUN")
