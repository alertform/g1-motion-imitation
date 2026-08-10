#!/usr/bin/env bash
cd "$HOME/tools/rl" || exit 1
source .venv/bin/activate
cp /mnt/d/g1-imitation/src/rl/*.py .
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python - <<'PYEOF' 2>&1 | grep -vE 'UserWarning|warnings.warn|Failed to import|cuda_timer'
"""容量判定实验。

同一段动作（walk1_subject1）、同一评估协议（16 起点 × 1500 步），
对比两个策略：
  v15 单段专家      —— 只学这一段
  v17 八段通才      —— 这一段只是它学的八分之一
两者网络结构完全相同 (512,256,128)。若差距巨大，说明容量被其他
七段挤占，而不是训练量或动作条件的问题。
"""
import numpy as np, pathlib
import jax_compat  # noqa
import rl_env, rl_play
from rl_eval_multi import pick_clips
from rl_eval_mjx import eval_starts, rollout

# 与训练完全一致的选段方式（按帧数排序），walk1_subject1 在索引 5
names = pick_clips("可用", "", 8)
CI = names.index("walk1_subject1")
print(f"训练时的段顺序: {names}")
print(f"walk1_subject1 的索引 = {CI}\n")

ref, refv, refb, refc, refl = rl_env.load_reference(names)
env8 = rl_env.G1Imitate(ref, refv, refb, refc, refl, ep_len=1500)
starts = eval_starts(int(np.asarray(refl)[CI]), 16)
print(f"起始帧: {starts.tolist()}\n")

HOME = pathlib.Path.home()/"tools"/"rl"
rows = []

# 前馈基线（在八段环境里，clip=CI）
s0, j0, r0, v0, _ = rollout(env8, starts, None, 1500, CI)
rows.append(("零动作前馈", s0, j0, r0, v0))

# v17 八段通才
pol17 = rl_play.build_policy(HOME/"runs"/"multi"/"policy_latest.pkl",
                             rl_env.OBS_SIZE, rl_env.NU)
s2, j2, r2, v2, _ = rollout(env8, starts, pol17, 1500, CI)
rows.append(("v17 八段通才", s2, j2, r2, v2))

print(f"  {'':>14} {'存活均值':>9} {'中位':>7} {'跑满':>7} {'关节°':>8} {'漂移cm':>9}")
for lbl, s, j, r, v in rows:
    print(f"  {lbl:>14} {s.mean():>9.1f} {np.median(s):>7.0f} "
          f"{int((s>=1500).sum()):>4}/16 {np.nanmedian(j):>8.2f} "
          f"{np.nanmedian(r):>9.2f}")
print()
for lbl, s, j, r, v in rows:
    print(f"  {lbl:>14} {s.tolist()}")

print()
print("  === v15 单段专家（历史记录，同协议 16 起点 × 1500 步）===")
print(f"  {'v15 单段专家':>14} {1490.2:>9.1f} {1500:>7} {15:>4}/16 "
      f"{5.62:>8.2f} {4.36:>9.2f}")
print()
print(f"  同一段动作、同一网络结构 (512,256,128)：")
print(f"    单段专家 1490.2  vs  八段通才 {s2.mean():.1f}  "
      f"= {1490.2/max(s2.mean(),1e-9):.1f}x 差距")
PYEOF
