#!/bin/zsh
# Push one of our kernels to Kaggle. Each real push creates a new kernel version and starts a run
# on your GPU quota, so run it only when you mean it. DRY_RUN=1 assembles and shows, pushes nothing.
#   ./push.sh train        EfficientNetV2-S leg training (fold = OURS_FOLD in the script, default 0)
#   ./push.sh train-cons   E03b: same training, team consensus labels as the target (arm effv2s_cons)
#   ./push.sh submission   the 0.943 base + our leg notebook (rebuilt from the base first)
set -euo pipefail
HERE=${0:A:h}
LEG=${HERE:h}
PY=${PYTHON:-$HOME/.local/share/uv/python/cpython-3.12.14-macos-aarch64-none/bin/python3.12}
if [[ -n ${KAGGLE_CLI:-} ]]; then
  KAGGLE=(${=KAGGLE_CLI})
else
  UVX=${UVX:-$(command -v uvx || echo $HOME/Library/Python/3.9/bin/uvx)}
  KAGGLE=($UVX --from 'kaggle>=2.2' kaggle)
fi
DIR=$(mktemp -d)
case ${1:-} in
  train)
    cp "$LEG/train_ours_leg.py" "$DIR/"
    cp "$HERE/train.kernel-metadata.json" "$DIR/kernel-metadata.json" ;;
  train-cons)
    # E03b: identical to `train` except the training target (team consensus labels, shared kernel
    # esfersami50/nx1-01-consensus-labels v2) and the arm name. Script kernels take no arguments,
    # so the two defaults are rewritten in the pushed copy.
    "$PY" - "$LEG/train_ours_leg.py" "$DIR/train_ours_leg.py" <<'EOF'
import sys
src, dst = sys.argv[1:3]
text = open(src).read()
for old, new in (('env("OURS_LABELS", "labels_llm_soft.parquet")', 'env("OURS_LABELS", "labels_consensus.parquet")'),
                 ('env("OURS_ARM", "effv2s")', 'env("OURS_ARM", "effv2s_cons")')):
    assert text.count(old) == 1, old
    text = text.replace(old, new)
compile(text, dst, "exec")
open(dst, "w").write(text)
EOF
    cp "$HERE/traincons.kernel-metadata.json" "$DIR/kernel-metadata.json" ;;
  submission)
    "$PY" "$LEG/build_submission_notebook.py"
    cp "$LEG/submission/rsna-knee-base-plus-ours.ipynb" "$DIR/"
    cp "$HERE/submission.kernel-metadata.json" "$DIR/kernel-metadata.json" ;;
  verify)
    # CPU check that infer_kernel.py's DICOM rebuild matches the training corpus (E11 diagnosis).
    # infer_kernel.py is embedded verbatim so the check runs the exact code the submission ran.
    "$PY" - "$LEG/verify_corpus_rebuild.py" "$LEG/infer_kernel.py" "$DIR/verify_corpus_rebuild.py" <<'EOF'
import base64, sys
src, ik, dst = sys.argv[1:4]
text = open(src).read()
assert text.count("__INFER_KERNEL_B64__") == 1
text = text.replace("__INFER_KERNEL_B64__", base64.b64encode(open(ik, "rb").read()).decode())
compile(text, dst, "exec")
open(dst, "w").write(text)
EOF
    cp "$HERE/verify.kernel-metadata.json" "$DIR/kernel-metadata.json" ;;
  *) echo "usage: $0 train|train-cons|submission|verify" >&2; rm -rf "$DIR"; exit 2 ;;
esac
# Teammate's account: KAGGLE_OWNER=esfersami50 KAGGLE_TOKEN_FILE=<first line = their KGAT token>
# pushes the kernel under their namespace, on their GPU quota.
if [[ -n ${KAGGLE_OWNER:-} ]]; then
  sed -i '' "s|\"id\": \"hosen42/|\"id\": \"$KAGGLE_OWNER/|" "$DIR/kernel-metadata.json"
fi
grep '"id"' "$DIR/kernel-metadata.json"
if [[ -n ${DRY_RUN:-} ]]; then
  echo "DRY RUN: would run: ${KAGGLE[*]} kernels push -p $DIR"
  ls -la "$DIR"
  rm -rf "$DIR"
  exit 0
fi
export KAGGLE_API_TOKEN=$(head -1 "${KAGGLE_TOKEN_FILE:-$HOME/.kaggle/access_token}" | tr -d '\n\r ')
"${KAGGLE[@]}" kernels push -p "$DIR"
rm -rf "$DIR"
