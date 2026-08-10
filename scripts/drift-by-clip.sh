#!/usr/bin/env bash
cd "$HOME/tools/rl" || exit 1
source .venv/bin/activate
cp /mnt/d/g1-imitation/src/rl/*.py .
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python - <<'PYEOF' 2>&1 | grep -vE 'UserWarning|warnings.warn|Failed to import|cuda_timer'
"""逐段对比策略与前馈的漂移——位置回正到底在哪些段失效了？

判据：策略漂移 ≈ 前馈漂移 -> 回正完全没起作用
      策略漂移 << 前馈漂移 -> 回正正常
"""
import numpy as np, pathlib
import jax_compat  # noqa
import rl_env, rl_play
from rl_eval_multi import pick_clips
from rl_eval_mjx import eval_starts, rollout

names = pick_clips("可用", "", 8)
ref, refv, refb, refc, refl = rl_env.load_reference(names)
env = rl_env.G1Imitate(ref, refv, refb, refc, refl, ep_len=800)
lens = np.asarray(refl)

pol = rl_play.build_policy(
    pathlib.Path.home()/"tools"/"rl"/"runs"/"multi"/"policy_latest.pkl",
    rl_env.OBS_SIZE, rl_env.NU)

# 各段参考的速度尺度——用于检验「速度尺度冲突」这个猜想
print("=== 各段参考的根速度量级 ===")
print(f"  {'动作':<26}{'中位m/s':>9}{'p95':>8}{'最大':>8}")
spd = {}
for ci, nm in enumerate(names):
    v = np.asarray(refv[ci])[:int(lens[ci]), 0:3]
    s = np.linalg.norm(v, axis=1)
    spd[nm] = np.median(s)
    print(f"  {nm:<26}{np.median(s):>9.2f}{np.percentile(s,95):>8.2f}{s.max():>8.2f}")

print()
print("=== 漂移：策略 vs 前馈（4 起点 × 800 步）===")
print(f"  {'动作':<26}{'前馈cm':>9}{'策略cm':>9}{'改善':>8}{'存活':>8}  判定")
rows = []
for ci, nm in enumerate(names):
    st = eval_starts(int(lens[ci]), 4)
    _, _, r0, _, _ = rollout(env, st, None, 800, ci)
    s1, _, r1, _, _ = rollout(env, st, pol, 800, ci)
    d0, d1 = np.nanmedian(r0), np.nanmedian(r1)
    ratio = d1/max(d0, 1e-9)
    verdict = ("回正失效" if ratio > 0.85 else
               "部分有效" if ratio > 0.5 else "回正正常")
    rows.append((nm, d0, d1, ratio, s1.mean(), verdict))
    print(f"  {nm:<26}{d0:>9.2f}{d1:>9.2f}{ratio:>8.2f}{s1.mean():>8.0f}  {verdict}",
          flush=True)

print()
fail = [r for r in rows if r[3] > 0.85]
print(f"  回正失效的段: {len(fail)}/{len(rows)}")
if fail:
    print(f"    {', '.join(r[0] for r in fail)}")

print()
print("=== 速度尺度与回正效果的关系 ===")
sp = np.array([spd[r[0]] for r in rows])
ra = np.array([r[3] for r in rows])
c = np.corrcoef(sp, ra)[0, 1]
print(f"  段的中位速度 vs 漂移比值，相关系数 {c:+.2f}")
print(f"  {'-> 速度越快回正越差，支持速度尺度冲突' if c > 0.5 else '-> 与速度无关，冲突假设不成立'}")
PYEOF
