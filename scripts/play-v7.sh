#!/usr/bin/env bash
# 起 viewer 回放训练好的策略。
# rl venv 里没装那个 zz-wsl-gl.pth，所以显式设 GL 环境变量，
# 否则 WSLg 会退回软件渲染，帧率惨不忍睹。
export GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
export JAX_PLATFORMS=cpu          # 策略网络很小，别去占显存

cd "$HOME/tools/rl" || exit 1
source .venv/bin/activate
cp /mnt/d/WSL/rl_play.py /mnt/d/WSL/rl_env.py .

exec python rl_play.py --ckpt runs/walk/policy.pkl "$@"
