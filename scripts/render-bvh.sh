#!/usr/bin/env bash
# 把 BVH 原始动捕（人体骨架 + G1 重定向对照）离线渲染成 mp4。
# 实时窗口版每帧做 IK+渲染太卡，离线渲染慢点无所谓，看的时候流畅。
export MUJOCO_GL=egl                    # 无头渲染，不开窗口
export OMP_NUM_THREADS=6                # 别把 CPU 吃满，训练还在跑

CLIP="${1:-walk1_subject1}"
OUT="/mnt/e/bvh_${CLIP}.mp4"

cd "$HOME/tools/GMR" || exit 1
source .venv/bin/activate
python scripts/bvh_to_robot.py \
    --bvh_file "$HOME/tools/lafan1/bvh/${CLIP}.bvh" \
    --robot unitree_g1 \
    --record_video --video_path "$OUT" 2>&1 | tail -5
echo "视频: $OUT"
ls -la "$OUT" 2>/dev/null
