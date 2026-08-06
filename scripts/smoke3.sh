#!/usr/bin/env bash
cd "$HOME/tools/rl" || exit 1
source .venv/bin/activate
cp /mnt/d/g1-imitation/src/rl/rl_env.py /mnt/d/g1-imitation/src/rl/rl_train.py .
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python rl_train.py --smoke --envs 1024 2>&1 \
  | grep -vE 'UserWarning|warnings.warn|Failed to import|cuda_timer'
echo "--- 存档文件 ---"
ls -la runs/ 2>/dev/null | tail -5
