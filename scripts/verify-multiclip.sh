#!/usr/bin/env bash
cd "$HOME/tools/rl" || exit 1
source .venv/bin/activate
cp /mnt/d/g1-imitation/src/rl/*.py .
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python - <<'PYEOF' 2>&1 | grep -vE 'UserWarning|warnings.warn|Failed to import|cuda_timer'
"""多段支持验证：补齐是否污染训练、短段索引是否正确钳制。"""
import numpy as np, jax, jax.numpy as jp
import jax_compat  # noqa
import rl_env

# 故意挑长度差异最大的两段
NAMES = ["fallAndGetUp3_subject1", "aiming2_subject2", "walk1_subject1"]
ref, refv, refb, refc, refl = rl_env.load_reference(NAMES)
lens = np.asarray(refl)
T = ref.shape[1]
print(f"=== 载入 {len(NAMES)} 段 ===")
for n, L in zip(NAMES, lens):
    print(f"  {n:<28} {L:>6} 帧  补齐区 {T-L:>6} 帧 ({(T-L)/T*100:.0f}%)")
print(f"  补齐到 T={T}，利用率 {lens.sum()/(len(lens)*T)*100:.1f}%")

print()
print("=== 1. 补齐区确实是末帧重复（不是 0）===")
for c, L in enumerate(lens):
    if L >= T:
        print(f"  段{c}: 无补齐区")
        continue
    last = np.asarray(ref[c, L-1])
    pad0 = np.asarray(ref[c, L])
    pad9 = np.asarray(ref[c, min(L+99, T-1)])
    print(f"  段{c}: |补齐首帧-末帧|={np.abs(pad0-last).max():.2e}  "
          f"|补齐+99-末帧|={np.abs(pad9-last).max():.2e}  "
          f"{'OK 是末帧重复' if np.abs(pad0-last).max() < 1e-6 else '异常'}")

print()
env = rl_env.G1Imitate(ref, refv, refb, refc, refl, ep_len=500)
print("=== 2. 各段的 RSI 起点上界按自己的长度算 ===")
ms = np.asarray(env._max_start_arr)
for c, (n, L) in enumerate(zip(NAMES, lens)):
    print(f"  {n:<28} 长度 {L:>6}  起点上界 {ms[c]:>6}  "
          f"(= L-500-6, {'OK' if ms[c] == max(1, L-508) else '不符'})")

print()
print("=== 3. 参考索引按各段长度钳制，不会读到补齐区 ===")
for c, L in enumerate(lens):
    q_in, _ = env._ref_at(c, int(L)-1)          # 最后有效帧
    q_out, _ = env._ref_at(c, T+1000)           # 远超范围
    same = np.abs(np.asarray(q_in) - np.asarray(q_out)).max()
    print(f"  段{c}: 越界索引 -> 钳到末帧，差 {same:.2e}  "
          f"{'OK' if same < 1e-6 else '异常'}")

print()
print("=== 4. reset 抽 2048 次：起点都在各段有效范围内 ===")
keys = jax.random.split(jax.random.PRNGKey(0), 2048)
st = jax.jit(jax.vmap(env.reset))(keys)
clips = np.asarray(st.info["clip"]); starts = np.asarray(st.info["start"])
bad = 0
for c in range(len(lens)):
    sel = starts[clips == c]
    if len(sel) == 0:
        continue
    over = (sel >= ms[c]).sum()
    ncon_bad = sum(1 for s in sel if refc[c][s] < rl_env.MIN_START_CONTACTS)
    bad += over + ncon_bad
    print(f"  段{c}: 抽中 {len(sel):>5} 次  范围[{sel.min()},{sel.max()}]  "
          f"越界 {over}  接触不足 {ncon_bad}")
print(f"  违规合计 {bad}   {'OK' if bad == 0 else '有问题'}")

print()
print("=== 5. 各段被抽中的概率是否均匀 ===")
cnt = np.bincount(clips, minlength=len(lens))
print(f"  {cnt.tolist()}  期望各约 {2048//len(lens)}")
print(f"  注意：均匀抽段意味着**短段的每一帧被抽中的概率更高**，")
print(f"        长段 {lens.max()} 帧 vs 短段 {lens.min()} 帧，单帧概率差 "
      f"{lens.max()/lens.min():.1f} 倍")
PYEOF
