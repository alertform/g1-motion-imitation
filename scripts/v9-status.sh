#!/usr/bin/env bash
L="$HOME/tools/rl/runs/walk/train.log"
echo "=== 进程 ==="
ps -eo pid,etime,cmd | grep '[r]l_train.py' || echo "  没有训练进程在跑"
echo
echo "=== 日志文件 ==="
ls -la "$L" 2>/dev/null || echo "  日志不存在"
echo
echo "=== 日志全文（尾部 25 行）==="
tail -25 "$L" 2>/dev/null || echo "  无内容"
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "  (WSL 内无法查询)"
