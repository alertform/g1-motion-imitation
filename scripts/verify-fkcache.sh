#!/usr/bin/env bash
cd "$HOME/tools/rl" || exit 1
source .venv/bin/activate
cp /mnt/d/g1-imitation/src/rl/*.py .
rm -rf "$HOME/tools/g1_dataset/fk_cache"      # 清掉旧缓存重来
export JAX_PLATFORMS=cpu

python - <<'PYEOF' 2>&1 | grep -vE 'UserWarning|warnings.warn|Failed to import|cuda|CUDA'
import time, numpy as np
import rl_env

NAMES = ["walk1_subject1", "sprint1_subject2"]

t0 = time.perf_counter()
r1 = rl_env.load_reference(NAMES)
t_cold = time.perf_counter() - t0
print(f"首次（冷，需算 FK）: {t_cold:.1f}s")

t0 = time.perf_counter()
r2 = rl_env.load_reference(NAMES)
t_warm = time.perf_counter() - t0
print(f"二次（命中缓存）  : {t_warm:.1f}s   加速 {t_cold/max(t_warm,1e-9):.0f}x")

print()
print("=== 缓存前后结果必须完全一致 ===")
for i, nm in enumerate(["qpos", "qvel", "body", "ncon", "lens"]):
    a, b = np.asarray(r1[i]), np.asarray(r2[i])
    d = np.abs(a.astype(float) - b.astype(float)).max()
    print(f"  {nm:<6} shape={a.shape}  最大差 {d:.2e}  {'OK' if d == 0 else '不一致！'}")

print()
print("=== 换一个批次组合，同段的缓存应当复用 ===")
t0 = time.perf_counter()
r3 = rl_env.load_reference(["walk1_subject1", "aiming2_subject2"])  # 补齐长度不同
t3 = time.perf_counter() - t0
print(f"  walk1 + aiming2（aiming2 首次算）: {t3:.1f}s")
# walk1 在两个批次里补齐长度不同，但真实帧的 FK 应当一致
L = int(np.asarray(r1[4])[0])
b_a = np.asarray(r1[2])[0][:L]
b_c = np.asarray(r3[2])[0][:L]
print(f"  walk1 真实帧的 body 位置在两批次间最大差 "
      f"{np.abs(b_a-b_c).max():.2e}  "
      f"{'OK 缓存复用正确' if np.abs(b_a-b_c).max() == 0 else '不一致！'}")

import pathlib
cache = pathlib.Path.home()/"tools"/"g1_dataset"/"fk_cache"
files = sorted(cache.glob("*.npz"))
print(f"\n  缓存文件 {len(files)} 个，共 "
      f"{sum(f.stat().st_size for f in files)/1e6:.1f} MB")
for f in files:
    print(f"    {f.name}  {f.stat().st_size/1e6:.1f} MB")
PYEOF
