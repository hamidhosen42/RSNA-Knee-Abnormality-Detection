"""Write the student sources a notebook's writer cell carries (preprocess.py, knee.py, infer.py[, infer_tta.py]) to a folder,
so the TTA geometry / model / thread tests run on exactly the code that will run on Kaggle.

    python -I ours_leg/tests/tta/extract_sources.py <notebook.ipynb> <out dir>
"""
import ast
import json
import os
import sys

nb, out = sys.argv[1], sys.argv[2]
src = lambda c: "".join(c["source"])
writer = next(src(c) for c in json.load(open(nb))["cells"] if src(c).startswith("# [h960r] the student sources"))
line = next(l for l in writer.splitlines() if l.startswith("_SOURCES = "))
files = ast.literal_eval(line[len("_SOURCES = "):])
os.makedirs(out, exist_ok=True)
for name, text in files.items():
    open(os.path.join(out, name), "w").write(text)
print("wrote", sorted(files), "to", out)
