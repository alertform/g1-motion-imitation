#!/usr/bin/env bash
cd "$HOME/tools/rl" || exit 1
source .venv/bin/activate
cp /mnt/d/WSL/rl_eval_mjx.py /mnt/d/WSL/rl_play.py .
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python rl_eval_mjx.py "$@" 2>&1 \
  | grep -vE 'UserWarning|warnings.warn|Failed to import|cuda_timer'
