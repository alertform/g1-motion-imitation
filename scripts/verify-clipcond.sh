#!/usr/bin/env bash
cd "$HOME/tools/rl" || exit 1
source .venv/bin/activate
cp /mnt/d/g1-imitation/src/rl/*.py .
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python - <<'PYEOF' 2>&1 | grep -vE 'UserWarning|warnings.warn|Failed to import|cuda_timer'
import numpy as np, jax
import jax_compat  # noqa
import rl_env, rl_play

NAMES = ["walk1_subject1", "sprint1_subject2", "aiming2_subject2"]
ref, refv, refb, refc, refl = rl_env.load_reference(NAMES)
env = rl_env.G1Imitate(ref, refv, refb, refc, refl, ep_len=500)

print(f"=== 观测维度 {rl_env.OBS_SIZE}（v16 是 215，+{rl_env.CLIP_EMBED_DIM} 为动作条件）===")
print()
print("=== 1. 各段的身份指纹 ===")
emb = np.asarray(env._clip_emb)
for i, n in enumerate(NAMES):
    print(f"  {n:<24} |v|={np.linalg.norm(emb[i]):.4f}  {emb[i][:4].round(3)}…")
print()
print("  两两余弦相似度（应当明显小于 1，否则策略分不开）：")
for i in range(len(NAMES)):
    for j in range(i+1, len(NAMES)):
        c = float(emb[i] @ emb[j])
        print(f"    {NAMES[i][:14]:<15} vs {NAMES[j][:14]:<15} {c:+.3f}")

print()
print("=== 2. jax 与 numpy 两侧的 embedding 必须逐位一致 ===")
for i in range(len(NAMES)):
    roll = rl_play.NumpyRollout(np.asarray(ref[i]), np.asarray(refv[i]),
                                clip=i, n_clip=len(NAMES))
    d = np.abs(roll.clip_emb - emb[i]).max()
    print(f"  段{i}: 最大差 {d:.2e}  {'OK' if d == 0 else '不一致！'}")

print()
print("=== 3. 观测里 embedding 落在正确的位置 ===")
# 布局: grav3 + angvel3 + qpos29 + qvel29 + last29 + vt3 + perr3 + emb8 + fut
OFF = 3+3+29+29+29+3+3
for i in range(len(NAMES)):
    q, v = env._ref_at(i, 100)
    data = env.pipeline_init(q, v)
    o = np.asarray(env._obs(data, i, 100, jax_zeros := __import__("jax").numpy.zeros(29)))
    got = o[OFF:OFF+rl_env.CLIP_EMBED_DIM]
    d = np.abs(got - emb[i]).max()
    print(f"  段{i}: 观测[{OFF}:{OFF+8}] 与指纹差 {d:.2e}  {'OK' if d < 1e-6 else '位置错了'}")

print()
print("=== 4. numpy / jax 完整观测一致性 ===")
i = 1
roll = rl_play.NumpyRollout(np.asarray(ref[i]), np.asarray(refv[i]),
                            clip=i, n_clip=len(NAMES))
roll.reset(100, np.asarray(refv[i]))
mine = roll.obs(100, np.zeros(29))
import jax.numpy as jp
st = env.pipeline_init(*env._ref_at(i, 100))
st = st.replace(qpos=jp.asarray(roll.d.qpos), qvel=jp.asarray(roll.d.qvel))
theirs = np.asarray(env._obs(st, i, 100, jp.zeros(29)))
print(f"  维度 numpy={len(mine)} jax={len(theirs)}   最大差 {np.abs(mine-theirs).max():.3e}")
print(f"  {'一致' if np.abs(mine-theirs).max() < 1e-4 else '不一致！'}")
PYEOF
