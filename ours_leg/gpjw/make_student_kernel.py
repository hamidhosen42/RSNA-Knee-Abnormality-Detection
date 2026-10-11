"""Two more full-data reader arms, each ONE change away from the adopted pseudo-label reader B (LB 0.939), trained in
parallel on a T4x2 with B's exact training code (make_pseudo_kernel.py's train_arm, reused verbatim):

  C   targets = clip(0.5 * v0 + 0.5 * Mirror teacher probability) - the teacher is nartaa's Anatomical Mirror Raptor
      (public 0.949) run on the train studies by esfersami50/rsna-knee-mirror-teacher; a study the teacher did not score
      keeps v0.  Same idea as the public 0.955 notebook's "student of the mirror".  OAI-derived (teacher) -> public-LB
      candidate only.  Backbone convnext_tiny as B.
  B2  targets = B's labels_pl.csv (0.5 v0 + 0.5 5-fold reader OOF, esfersami50/rsna-knee-gpjw-pseudo v1) unchanged;
      backbone convnext_tiny.fb_in22k_ft_in1k = the tag goodpjw2008's released folds were trained with (ours were
      plain convnext_tiny, and our folds' val AUC was 0.88-0.89 vs the author's 0.93).  Rule-clean.

Output: cnxt_plC_full.pt, cnxt_plB2_full.pt (infer.py format), gold_arm{C,B2}.csv, logs.
Pre-registered (each leg alone vs B 0.939): adopt C / B2 if LB >= 0.942 (B + 0.003) and gold-58 not below B's 0.9037 - 0.01.

    python make_student_kernel.py [--owner=esfersami50] [--teacher-version=1] [--pseudo-version=1]
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OWNER = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--owner=")), "esfersami50")
TV = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--teacher-version=")), "1")
PV = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--pseudo-version=")), "1")
N_SHARDS = 5
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"

pseudo_src = open(os.path.join(HERE, "make_pseudo_kernel.py")).read()
MAIN = re.search(r"MAIN = r'''(.*?)\n'''\n", pseudo_src, re.S).group(1)
knee = open(os.path.join(HERE, "src", "knee.py")).read()


def sub(old, new):
    global MAIN
    assert MAIN.count(old) == 1, old[:80]
    MAIN = MAIN.replace(old, new)


sub('''def train_arm(arm):
    a = dict(ARGS, arm=arm, alpha=ALPHA if arm == "B" else None)''', '''ARM_CFG = {"C": dict(backbone="convnext_tiny", targets="0.5 v0 + 0.5 Mirror teacher"),
           "B2": dict(backbone="convnext_tiny.fb_in22k_ft_in1k", targets="B labels_pl.csv (0.5 v0 + 0.5 OOF)")}


def build_targets_C():
    lab = build_labels()
    m = pd.read_csv(find("mirror_train_raw.csv"), dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[LABELS]
    m = m.reindex(lab.index.astype(str))
    m.index = lab.index
    ok = m.notna().all(axis=1) & ~(m == 0.5).all(axis=1)
    weak = lab.is_gold == 0
    idx = lab.index[weak & ok]
    y = (lab.loc[idx, LABELS].values > 0.5).astype(int)
    t_auc, t_each = macro_auc(y, m.loc[idx].values)
    say("targets", f"teacher scored {int(ok[weak].sum())}/{int(weak.sum())} weak studies | teacher vs v0 macro {t_auc:.4f} [" +
        " ".join(f"{x:.3f}" for x in t_each) + "]")
    gold = lab[lab.is_gold == 1]
    gok = ok[gold.index]
    if gok.sum() > 10:
        g_auc, _ = macro_auc(gold.loc[gok, [f"gold_{c}" for c in LABELS]].values, m.loc[gold.index[gok]].values)
        say("targets", f"teacher on gold-58 (diagnostic, teacher was selected on gold): {g_auc:.4f}")
    new = (0.5 * lab.loc[idx, LABELS].values + 0.5 * m.loc[idx].values).clip(0.01, 0.99)
    say("targets", f"mean |C - v0| {np.abs(new - lab.loc[idx, LABELS].values).mean():.4f}")
    lab.loc[idx, LABELS] = new
    lab.to_csv(f"{OUT}/labels_C.csv")
    return lab


def train_arm(arm):
    a = dict(ARGS, arm=arm, **ARM_CFG[arm])''')
sub('''    lab = pd.read_csv(f"{OUT}/labels_pl.csv", index_col=0) if arm == "B" else build_labels()''',
    '''    lab = build_targets_C() if arm == "C" else pd.read_csv(find("labels_pl.csv"), index_col=0)''')
sub('''    print(f"gpus {torch.cuda.device_count()} | script {script_path()} | probe {PROBE} | alpha {ALPHA}", flush=True)
    codes = run_parallel([("GPJW_OOF", "0,2,4"), ("GPJW_OOF", "1,3")])
    print("oof exit codes:", codes, flush=True)
    if any(c != 0 for c in codes.values()):
        raise SystemExit("OOF phase failed")
    build_targets()
    codes = run_parallel([("GPJW_ARM", "A"), ("GPJW_ARM", "B")])''', '''    print(f"gpus {torch.cuda.device_count()} | script {script_path()} | probe {PROBE} | arms C, B2", flush=True)
    codes = run_parallel([("GPJW_ARM", "C"), ("GPJW_ARM", "B2")])''')
src = knee + MAIN.replace("__PROBE__", "False").replace("__ALPHA__", "0.5")
compile(src, "gpjwstudent.py", "exec")
d = os.path.join(HERE, "..", "kaggle", "gpjwstudent" + ("" if OWNER == "esfersami50" else f"_{OWNER}"))
os.makedirs(d, exist_ok=True)
open(os.path.join(d, "gpjwstudent.py"), "w").write(src)
meta = {"id": f"{OWNER}/rsna-knee-gpjw-student", "title": "rsna-knee-gpjw-student", "code_file": "gpjwstudent.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "docker_image": DOCKER, "docker_image_pinning_type": "original",
        "dataset_sources": ["stevenleehans/rsna-knee-llm-report-labels/6", "pilkwang/rsna-knee-llm-labels/1",
                            "lixin73/rsna-knee-llm-report-labels-sol56/1",
                            "laymond/rsna-knee-abnormality-qwen3-8b-weak-labels/2"],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "kernel_sources": [f"esfersami50/rsna-knee-gpjw-cache-{i}/1" for i in range(N_SHARDS)]
                          + [f"esfersami50/rsna-knee-gpjw-pseudo/{PV}", f"esfersami50/rsna-knee-mirror-teacher/{TV}"],
        "model_sources": []}
json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
print("wrote", d)
