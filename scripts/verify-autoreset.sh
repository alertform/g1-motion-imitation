#!/usr/bin/env bash
cd "$HOME/tools/rl" || exit 1
source .venv/bin/activate
cp /mnt/d/g1-imitation/src/rl/rl_env.py /mnt/d/g1-imitation/src/rl/rl_train.py /mnt/d/g1-imitation/src/rl/rl_play.py /mnt/d/g1-imitation/src/rl/rl_eval.py .
export JAX_PLATFORMS=cpu

python - <<'PYEOF' 2>&1 | grep -vE 'UserWarning|warnings.warn|Failed to import|cuda|CUDA'
"""用 brax 真实的 wrapper 链验证：自动重置后参考帧是否跟着复位。

判据：重置发生的那一步之后，机器人被恢复到起始位姿。如果参考索引
也正确复位，pose_err（当前关节角 vs 参考关节角）应该回到很小；
如果索引脱钩，pose_err 会停在一个很大的值。
"""
import numpy as np, jax, jax.numpy as jp
import jax_compat  # noqa
import rl_env
from brax.envs.wrappers import training as W

ref, refv = rl_env.load_reference(["walk1_subject1"], max_frames=3000)
base = rl_env.G1Imitate(ref, refv, ep_len=40)

EP = 40
env = W.AutoResetWrapper(W.VmapWrapper(W.EpisodeWrapper(base, EP, 1)))

N = 4
st = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), N))
step = jax.jit(env.step)
zero = jp.zeros((N, rl_env.NU))

print(f"  回合长上限 = {EP} 步，跑 3 个回合看跨重置的表现")
print(f"  起始帧 start = {np.asarray(st.info['start'])}")
print()
print(f"  {'步':>4} {'steps':>6} {'done':>5} {'pose_err(rad²)':>15} {'各环境 pose_err':>34}")
errs = []
for i in range(1, EP*3 + 3):
    st = step(st, zero)
    pe = np.asarray(st.metrics["pose_err"])
    errs.append(pe.mean())
    if i % 8 == 0 or i in (EP, EP+1, EP+2, 2*EP, 2*EP+1, 2*EP+2):
        d = np.asarray(st.info["episode_done"]).astype(int)
        s = np.asarray(st.info["steps"]).astype(int)
        mark = "  <-- 回合结束" if d.any() else ""
        print(f"  {i:>4} {str(s):>16} {str(d):>12} {pe.mean():>10.5f}"
              f"   {np.array2string(pe, precision=4)}{mark}")

errs = np.array(errs)
print()
print(f"  重置前 (1~{EP} 步)  pose_err 均值 = {errs[:EP].mean():.5f}")
print(f"  重置后 ({EP+1}~{2*EP} 步) pose_err 均值 = {errs[EP:2*EP].mean():.5f}")
r = errs[EP:2*EP].mean() / max(errs[:EP].mean(), 1e-9)
print(f"  比值 = {r:.2f}")
print()
if r < 3.0:
    print("  通过：重置后参考帧跟着复位，误差量级不变")
else:
    print("  失败：重置后误差暴涨，参考帧仍与物理状态脱钩")
PYEOF
