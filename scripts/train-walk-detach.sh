#!/usr/bin/env bash
# 以脱离会话的方式启动训练：setsid 让 python 不再属于本次 wsl 会话，
# 本脚本立刻返回。日志直接落盘，进程生死与调用方无关。
cd "$HOME/tools/rl" || exit 1
cp /mnt/d/g1-imitation/src/rl/rl_env.py /mnt/d/g1-imitation/src/rl/rl_train.py /mnt/d/g1-imitation/src/rl/rl_play.py \
   /mnt/d/g1-imitation/src/rl/rl_eval.py /mnt/d/g1-imitation/src/rl/rl_eval_mjx.py .

OUT="$HOME/tools/rl/runs/walk"
mkdir -p "$OUT"

# 步数走位置参数，不要用环境变量前缀——那需要外面再套一层 bash -c，
# 而多出来的那层 shell 退出时会把 setsid 的子进程一起带走（v9 首次启动
# 就是这么静默死掉的，日志 0 字节、进程无踪）。
STEPS="${1:-100000000}"
RESTORE="${2:-}"          # 可选：续训用的存档路径

EXTRA=""
[ -n "$RESTORE" ] && EXTRA="--restore $RESTORE"

nohup setsid env XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$HOME/tools/rl/.venv/bin/python" -u rl_train.py \
    --clips walk1_subject1 \
    --envs 4096 \
    --steps "$STEPS" \
    --ep-len 500 \
    --out "$OUT" \
    $EXTRA \
    > "$OUT/train.log" 2>&1 < /dev/null &

PID=$!
disown

# 必须在这里等一会儿再退出：父 shell 立刻退出会和 setsid 的脱离过程
# 形成竞态，子进程有时来不及脱离就被一起带走（日志 0 字节、进程无踪）。
# 同一条命令曾经成功过一次、失败过一次，正是竞态的特征。
for i in $(seq 1 10); do
    sleep 1
    kill -0 "$PID" 2>/dev/null || { echo "启动失败：进程 $PID 在 ${i}s 内退出"; \
        echo "--- 日志 ---"; cat "$OUT/train.log"; exit 1; }
done

echo "已脱离启动 PID $PID   步数 $STEPS   （存活 10s 确认）"
echo "日志: $OUT/train.log"
