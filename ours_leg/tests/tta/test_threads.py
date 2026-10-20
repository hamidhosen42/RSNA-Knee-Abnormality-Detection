"""Test 4: the two-device path of run_tta (plain on device 0, mirror on device 1 in a worker thread) gives the same
numbers as the one-device path.  CPU stands in for both GPUs; reuses the synthetic data + random checkpoints of test_model.py.

    uv run ... python -I test_threads.py <runtime_src dir> <scratch dir>
"""
import json
import os
import sys

src_dir, work = sys.argv[1], sys.argv[2]
sys.path.insert(0, src_dir)
import torch  # noqa: E402

import infer_tta  # noqa: E402

torch.set_num_threads(8)
DATA = os.path.join(work, "data")
ck = [os.path.join(work, "rand0.pt"), os.path.join(work, "rand1.pt")]


def main():
    p1, q1, i1 = infer_tta.run_tta(DATA, "test", ck, workers=0, devices=["cpu"])
    p2, q2, i2 = infer_tta.run_tta(DATA, "test", ck, workers=2, devices=["cpu", "cpu"])
    out = {"plain_maxabs": float((p1 - p2).abs().values.max()), "mirror_maxabs": float((q1 - q2).abs().values.max()),
           "info_two_devices": i2}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
