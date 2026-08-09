#!/usr/bin/env bash
# 多段训练。用法: train-multi.sh <段数> <步数> [续训存档]
cd "$HOME/tools/rl" || exit 1
cp /mnt/d/g1-imitation/src/rl/*.py .

LIMIT="${1:-8}"
STEPS="${2:-400000000}"
RESTORE="${3:-}"
OUT="$HOME/tools/rl/runs/multi"
mkdir -p "$OUT"

EXTRA=""
[ -n "$RESTORE" ] && EXTRA="--restore $RESTORE"

nohup setsid env XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$HOME/tools/rl/.venv/bin/python" -u rl_train.py \
    --grade 可用 \
    --limit "$LIMIT" \
    --envs 4096 \
    --steps "$STEPS" \
    --ep-len 500 \
    --out "$OUT" \
    $EXTRA \
    > "$OUT/train.log" 2>&1 < /dev/null &

PID=$!
disown
for i in $(seq 1 10); do
    sleep 1
    kill -0 "$PID" 2>/dev/null || { echo "启动失败：进程在 ${i}s 内退出"; \
        cat "$OUT/train.log"; exit 1; }
done
echo "已启动 PID $PID   $LIMIT 段 × $STEPS 步   （存活 10s 确认）"
echo "日志: $OUT/train.log"
