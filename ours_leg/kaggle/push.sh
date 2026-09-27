#!/bin/zsh
# Push one of our kernels to Kaggle. Each real push creates a new kernel version and starts a run
# on your GPU quota, so run it only when you mean it. DRY_RUN=1 assembles and shows, pushes nothing.
#   ./push.sh train        EfficientNetV2-S leg training (fold = OURS_FOLD in the script, default 0)
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
  submission)
    "$PY" "$LEG/build_submission_notebook.py"
    cp "$LEG/submission/rsna-knee-base-plus-ours.ipynb" "$DIR/"
    cp "$HERE/submission.kernel-metadata.json" "$DIR/kernel-metadata.json" ;;
  *) echo "usage: $0 train|submission" >&2; rm -rf "$DIR"; exit 2 ;;
esac
if [[ -n ${DRY_RUN:-} ]]; then
  echo "DRY RUN: would run: ${KAGGLE[*]} kernels push -p $DIR"
  ls -la "$DIR"
  rm -rf "$DIR"
  exit 0
fi
export KAGGLE_API_TOKEN=$(tr -d '\n\r ' < ~/.kaggle/access_token)
"${KAGGLE[@]}" kernels push -p "$DIR"
rm -rf "$DIR"
