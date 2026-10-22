"""infer_fam.py (H982 / H983) on synthetic DICOM studies with random-weight KneeNet checkpoints (CPU).
  (a) family-balanced output == 0.5 x infer.run(Mirror checkpoints) + 0.5 x infer.run(S1 checkpoints) (the H976 code path)
  (b) the checkpoint order does not matter
  (c) it really differs from the equal mean of all checkpoints (H976's 7-checkpoint mean) when the families disagree
  (d) the weights for the real 7 file names are 0.1 x 5 Mirror + 0.25 x 2 S1; one family missing raises (the notebook then
      reruns infer.py = H976)
  (e) the command line is infer.py's (same argv order, same table written)
The synthetic studies and the two Mirror checkpoints come from test_model.build(); the S1 one is named cnxt_plE75z_full.pt.

    uv run --no-project --with numpy --with pydicom --with opencv-python-headless --with pandas --with torch --with timm \
        python -I test_fam.py <student sources dir (extract_sources.py)> <scratch dir>
"""
import json
import os
import sys
import types

import numpy as np

sys.modules.setdefault("infer_tta", types.ModuleType("infer_tta"))   # test_model imports it; nothing here uses it
import test_model as tm  # noqa: E402  (argv, CPU patches of infer.run, build())
import torch  # noqa: E402

import infer  # noqa: E402
import infer_fam  # noqa: E402
from knee import KneeNet  # noqa: E402

REAL = ["cnxt_plC2_full.pt", "cnxt_plC3_full.pt", "cnxt_plC_full.pt", "cnxt_plS75_full.pt", "cnxt_plT75s1_full.pt",
        "cnxt_plE75s_full.pt", "cnxt_plE75t_full.pt"]


def main():
    mirror = tm.build()
    torch.manual_seed(2)
    m = KneeNet("convnext_tiny", pretrained=False)
    with torch.no_grad():
        m.cls_w.mul_(50.0)
    s1 = os.path.join(tm.work, "cnxt_plE75z_full.pt")
    torch.save({"model": m.state_dict(), "args": {"backbone": "convnext_tiny", "res": 256}}, s1)

    ref_m = infer.run(tm.DATA, "test", mirror, workers=0)
    ref_s = infer.run(tm.DATA, "test", [s1], workers=0)
    equal = infer.run(tm.DATA, "test", mirror + [s1], workers=0)
    fam = infer_fam.run(tm.DATA, "test", [mirror[0], s1, mirror[1]], workers=0, device="cpu")
    fam2 = infer_fam.run(tm.DATA, "test", [s1, mirror[1], mirror[0]], workers=0, device="cpu")
    want = 0.5 * ref_m + 0.5 * ref_s
    out = {
        "a_fam_vs_half_half_maxabs": float((fam - want).abs().values.max()),
        "b_order_maxabs": float((fam2 - fam).abs().values.max()),
        "c_fam_vs_equal_mean_maxabs": float((fam - equal).abs().values.max()),
        "families_disagree_maxabs": float((ref_m - ref_s).abs().values.max()),
        "index_columns_as_infer_run": bool(fam.index.equals(ref_m.index) and list(fam.columns) == list(ref_m.columns)),
    }
    w = infer_fam.family_weights(["/kaggle/input/x/" + n for n in REAL])
    out["d_real_weights"] = [round(v, 6) for v in w]
    raised = []
    for names in (REAL[:5], REAL[5:]):
        try:
            infer_fam.family_weights(names)
        except ValueError:
            raised.append(True)
    main_of = lambda p: open(p).read().split('if __name__ == "__main__":\n')[1].splitlines()[:2]
    out["e_cli_as_infer"] = main_of(infer.__file__) == main_of(infer_fam.__file__)
    print(json.dumps(out, indent=1))
    assert out["a_fam_vs_half_half_maxabs"] < 1e-5 and out["b_order_maxabs"] < 1e-5 and out["index_columns_as_infer_run"]
    assert out["families_disagree_maxabs"] > 1e-2 and out["c_fam_vs_equal_mean_maxabs"] > 1e-3
    assert np.allclose(w, [0.1] * 5 + [0.25] * 2) and abs(sum(w) - 1) < 1e-9 and raised == [True, True]
    assert out["e_cli_as_infer"]
    print("ALL INFER_FAM CHECKS PASSED")


if __name__ == "__main__":
    main()
