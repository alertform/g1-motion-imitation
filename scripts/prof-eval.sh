#!/usr/bin/env bash
cd "$HOME/tools/rl" || exit 1
source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python - <<'PYEOF' 2>&1 | grep -vE 'UserWarning|warnings.warn|Failed to import|cuda_timer'
"""逐段评估到底慢在哪——分阶段计时，不再靠猜。"""
import time, numpy as np, jax
import jax_compat  # noqa
import rl_env
from rl_eval_multi import pick_clips
from rl_eval_mjx import eval_starts, rollout

def t(label, fn):
    t0 = time.perf_counter()
    r = fn()
    print(f"  {label:<34}{time.perf_counter()-t0:>8.1f}s", flush=True)
    return r

names = pick_clips("可用", "", 8)
ref, refv, refb, refc, refl = t("load_reference（8 段，有缓存）",
                                lambda: rl_env.load_reference(names))
env = t("G1Imitate 构造（含 mjcf.load_model）",
        lambda: rl_env.G1Imitate(ref, refv, refb, refc, refl, ep_len=1000))

import rl_play, pathlib
ck = pathlib.Path.home()/"tools"/"rl"/"runs"/"multi"/"policy_latest.pkl"
pol = t("build_policy", lambda: rl_play.build_policy(ck, rl_env.OBS_SIZE, rl_env.NU))

lens = np.asarray(refl)
starts = eval_starts(int(lens[0]), 6)
print(f"\n  第 1 段 {names[0]}，6 起点 × 1000 步：")
s0 = t("  rollout 前馈（含 jit 编译）", lambda: rollout(env, starts, None, 1000, 0))
s1 = t("  rollout 策略（含 jit 编译）", lambda: rollout(env, starts, pol, 1000, 0))
print(f"\n  第 2 段（jit 已缓存，看稳态速度）：")
starts2 = eval_starts(int(lens[1]), 6)
t("  rollout 前馈", lambda: rollout(env, starts2, None, 1000, 1))
t("  rollout 策略", lambda: rollout(env, starts2, pol, 1000, 1))
print(f"\n  前馈存活 {s0[0].tolist()}")
print(f"  策略存活 {s1[0].tolist()}")
PYEOF
