#!/usr/bin/env bash
cd "$HOME/tools/rl" || exit 1
source .venv/bin/activate
cp /mnt/d/g1-imitation/src/rl/*.py .
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python rl_eval_multi.py "$@" 2>&1 \
  | grep -vE 'UserWarning|warnings.warn|Failed to import|cuda_timer'
