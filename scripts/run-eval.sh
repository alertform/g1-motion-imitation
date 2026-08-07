#!/usr/bin/env bash
cd "$HOME/tools/rl" || exit 1
source .venv/bin/activate
cp /mnt/d/g1-imitation/src/rl/rl_eval.py /mnt/d/g1-imitation/src/rl/rl_eval_mjx.py /mnt/d/g1-imitation/src/rl/rl_play.py .
export JAX_PLATFORMS=cpu
python rl_eval.py "$@" 2>&1 \
  | grep -vE 'UserWarning|warnings.warn|Failed to import|cuda|CUDA'
