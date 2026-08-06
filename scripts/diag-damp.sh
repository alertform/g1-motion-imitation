#!/usr/bin/env bash
cd "$HOME/tools/rl" || exit 1
source .venv/bin/activate
export JAX_PLATFORMS=cpu
python - <<'PYEOF' 2>&1 | grep -vE 'UserWarning|warnings.warn|Failed to import|cuda|CUDA'
import numpy as np, mujoco
import rl_env

G1 = rl_env.XML.parent
# 原模型：<position kp="500" dampratio="1"/>，MuJoCo 按各关节等效惯量算 kd
m0 = mujoco.MjModel.from_xml_path(str(G1/"scene.xml"))
kd0 = -m0.actuator_biasprm[:, 2]
kp0 = m0.actuator_gainprm[:, 0]

print("=== 原模型 dampratio=1 算出的 per-joint kd vs 我用的全局常数 ===")
print(f"  我的设置: kp={rl_env.KP_SCALE:.0f} 全部关节 kd={rl_env.KD_VALUE:.1f}（一个常数）")
print(f"  原模型  : kp={kp0[0]:.0f} 全部关节 kd 范围 [{kd0.min():.1f}, {kd0.max():.1f}]")
print()
print(f"  {'关节':<30} {'原kd':>8} {'按比例缩到kp=250':>16} {'我的kd':>8} {'过阻尼倍数':>10}")
# kd ∝ sqrt(kp)，从 kp=500 缩到 250 应乘 sqrt(0.5)
scale = np.sqrt(rl_env.KP_SCALE / kp0[0])
rows = []
for i in range(m0.nu):
    nm = mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_JOINT, m0.actuator_trnid[i, 0])
    want = kd0[i] * scale
    rows.append((str(nm), kd0[i], want, rl_env.KD_VALUE / max(want, 1e-9)))
rows.sort(key=lambda r: -r[3])
for nm, k0, want, ratio in rows[:6]:
    print(f"  {nm:<30} {k0:>8.2f} {want:>16.2f} {rl_env.KD_VALUE:>8.1f} {ratio:>9.1f}x")
print("  ...")
for nm, k0, want, ratio in rows[-4:]:
    print(f"  {nm:<30} {k0:>8.2f} {want:>16.2f} {rl_env.KD_VALUE:>8.1f} {ratio:>9.1f}x")

print()
over = [r for r in rows if r[3] > 2.0]
print(f"  过阻尼超过 2 倍的关节: {len(over)}/{m0.nu}")
print(f"  其中手臂/腕部关节: {sum(1 for r in over if any(k in r[0] for k in ('elbow','wrist','shoulder')))}")
PYEOF
