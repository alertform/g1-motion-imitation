#!/usr/bin/env bash
cd "$HOME/tools/rl" || exit 1
source .venv/bin/activate
cp /mnt/d/WSL/rl_eval.py /mnt/d/WSL/rl_play.py .
export JAX_PLATFORMS=cpu
python rl_eval.py "$@" 2>&1 \
  | grep -vE 'UserWarning|warnings.warn|Failed to import|cuda|CUDA'
