#!/usr/bin/env bash
cd "$HOME/tools/rl" || exit 1
source .venv/bin/activate
export JAX_PLATFORMS=cpu
python - <<'PYEOF' 2>&1 | grep -vE 'UserWarning|warnings.warn|Failed to import|cuda|CUDA'
"""「动作僵硬」的来源诊断。

僵硬在视觉上可能来自三个完全不同的原因，对应完全不同的修法：
  A. 伺服增益过高      -> 关节死跟目标，没有柔顺性
  B. 动作抖动          -> 高频微修正，看起来紧绷
  C. 力矩饱和          -> 一直顶在上限，动作生硬
分别量出来，别猜。
"""
import pathlib, numpy as np, mujoco, jax
import rl_env, rl_play

ref, refv = rl_env.load_reference(["walk1_subject1"])
refn, refvn = np.asarray(ref[0]), np.asarray(refv[0])
roll = rl_play.NumpyRollout(refn, refvn)

ck = pathlib.Path.home()/"tools"/"rl"/"runs"/"walk"/"policy.pkl"
pol = rl_play.build_policy(ck, rl_env.OBS_SIZE, rl_env.NU)
key = jax.random.PRNGKey(0)

# 跑一段，记录动作、力矩、关节角
roll.reset(0, refvn)
last = np.zeros(rl_env.NU)
acts, taus, qs, refs = [], [], [], []
for i in range(400):
    a = np.asarray(pol(roll.obs(i, last), key)[0])
    roll.apply(a, i)
    acts.append(a.copy()); last = a
    taus.append(roll.d.qfrc_actuator[6:].copy())
    qs.append(roll.d.qpos[7:].copy())
    refs.append(refn[min(i+1, roll.T-1), 7:].copy())
    if roll.fell():
        break
acts, taus = np.array(acts), np.array(taus)
qs, refs = np.array(qs), np.array(refs)
n = len(acts)
print(f"采样 {n} 步\n")

print("=== A. 伺服增益：实际力矩 / 关节力矩上限 ===")
m = roll.m
lim = np.abs(m.jnt_actfrcrange[1:]).max(axis=1)
sat = np.abs(taus) / np.maximum(lim, 1e-9)
print(f"  力矩幅值   中位 {np.median(np.abs(taus)):6.1f} N·m   p95 {np.percentile(np.abs(taus),95):6.1f}")
print(f"  占上限比例 中位 {np.median(sat)*100:6.1f}%      p95 {np.percentile(sat,95)*100:6.1f}%")
print(f"  饱和帧占比 {100*np.mean(sat > 0.95):.2f}%  （>0.95 视为顶到上限）")
kdv = -m.actuator_biasprm[:, 2]
print(f"  当前 kp={rl_env.KP_SCALE:.0f}  kd per-joint [{kdv.min():.1f}, {kdv.max():.1f}]")

print()
print("=== B. 动作抖动：相邻两步的动作变化 ===")
da = np.abs(np.diff(acts, axis=0))
print(f"  |Δa| 中位 {np.median(da):.4f}   p95 {np.percentile(da,95):.4f}  （满幅=2）")
print(f"  折合关节目标角变化 中位 {np.median(da)*rl_env.ACT_SCALE*57.3:5.2f}°/步"
      f"   p95 {np.percentile(da,95)*rl_env.ACT_SCALE*57.3:5.2f}°/步")
print(f"  50Hz 控制下 p95 相当于 {np.percentile(da,95)*rl_env.ACT_SCALE*57.3*50:.0f}°/秒 的目标角抖动")

print()
print("=== C. 频谱：机器人关节角 vs 参考，高频成分对比 ===")
def hf_ratio(x):
    """高频能量占比：>5Hz 的功率 / 总功率。人走路主频约 1~2Hz。"""
    x = x - x.mean(axis=0)
    f = np.fft.rfftfreq(len(x), d=0.02)
    p = np.abs(np.fft.rfft(x, axis=0))**2
    return (p[f > 5].sum() / max(p.sum(), 1e-12))
print(f"  参考动作   高频(>5Hz)能量占比 {hf_ratio(refs)*100:6.3f}%")
print(f"  机器人实际 高频(>5Hz)能量占比 {hf_ratio(qs)*100:6.3f}%")
r = hf_ratio(qs) / max(hf_ratio(refs), 1e-12)
print(f"  比值 {r:.2f}   {'-> 机器人明显更抖' if r > 3 else '-> 频谱接近参考，抖动不是主因'}")

print()
print("=== D. 各关节的跟踪误差排行（找僵硬集中在哪）===")
err = np.degrees(np.abs(qs - refs)).mean(axis=0)
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i,0])
         for i in range(m.nu)]
order = np.argsort(-err)[:8]
for i in order:
    print(f"  {str(names[i]):<30} {err[i]:6.2f}°   力矩中位 {np.median(np.abs(taus[:,i])):6.1f} N·m")
PYEOF
