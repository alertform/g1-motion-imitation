#!/usr/bin/env bash
cd "$HOME/tools/rl" || exit 1
source .venv/bin/activate
export JAX_PLATFORMS=cpu
python - <<'PYEOF' 2>&1 | grep -vE 'UserWarning|warnings.warn|Failed to import|cuda|CUDA'
"""摔倒模式分析：是随机失稳，还是参考动作里有系统性的不可行时刻？

判据：
  失败点集中在某几个步态相位 -> 参考在那些时刻对 G1 不可行
  失败点均匀分散           -> 控制鲁棒性问题，与参考无关
"""
import pathlib, numpy as np, mujoco, jax
import rl_env, rl_play

ref, refv = rl_env.load_reference(["walk1_subject1"])
refn, refvn = np.asarray(ref[0]), np.asarray(refv[0])
roll = rl_play.NumpyRollout(refn, refvn)
ck = pathlib.Path.home()/"tools"/"rl"/"runs"/"walk"/"policy.pkl"
pol = rl_play.build_policy(ck, rl_env.OBS_SIZE, rl_env.NU)
key = jax.random.PRNGKey(0)

m = roll.m
FID = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
       for n in ("left_ankle_roll_link", "right_ankle_roll_link")]

# 参考动作里每一帧的支撑状态（用参考自带的接触标注推不出，用脚高度判定）
starts = np.linspace(0, roll.T - 520, 24).astype(int)
fails = []
for s in starts:
    roll.reset(s, refvn); last = np.zeros(rl_env.NU)
    tilt_hist = []
    for i in range(500):
        a = np.asarray(pol(roll.obs(s+i, last), key)[0])
        roll.apply(a, s+i); last = a
        up = 1.0 - 2.0*(roll.d.qpos[4]**2 + roll.d.qpos[5]**2)
        tilt_hist.append(np.degrees(np.arccos(np.clip(up, -1, 1))))
        if roll.fell():
            fz = roll.d.xpos[FID][:, 2]
            fails.append(dict(start=s, step=i+1, absframe=s+i+1,
                              z=roll.d.qpos[2], tilt=tilt_hist[-1],
                              nfoot=int((fz < 0.10).sum()),
                              tilt10=tilt_hist[max(0,len(tilt_hist)-10)]))
            break
    else:
        fails.append(dict(start=s, step=500, absframe=s+500, z=roll.d.qpos[2],
                          tilt=tilt_hist[-1], nfoot=-1, tilt10=tilt_hist[-10]))

surv = [f["step"] for f in fails]
print(f"=== {len(starts)} 个起点，存活步数 ===")
print(f"  {surv}")
print(f"  均值 {np.mean(surv):.1f}  中位 {np.median(surv):.0f}"
      f"  最短 {min(surv)}  最长 {max(surv)}")
print(f"  跑满 500 的有 {sum(1 for x in surv if x>=500)} 个")

print()
print("=== 摔倒瞬间的状态 ===")
real = [f for f in fails if f["step"] < 500]
print(f"  {'起点':>7} {'存活':>6} {'绝对帧':>8} {'躯干z':>7} {'倾角':>7} "
      f"{'10步前倾角':>10} {'触地脚数':>8}")
for f in sorted(real, key=lambda x: x["step"])[:12]:
    print(f"  {f['start']:>7} {f['step']:>6} {f['absframe']:>8} {f['z']:>7.3f} "
          f"{f['tilt']:>6.1f}° {f['tilt10']:>9.1f}° {f['nfoot']:>8}")

print()
print("=== 失败点在参考轨迹上的分布 ===")
frames = np.array([f["absframe"] for f in real])
print(f"  绝对帧位置: {sorted(frames.tolist())}")
hist, edges = np.histogram(frames, bins=8, range=(0, roll.T))
for h, e in zip(hist, edges[:-1]):
    print(f"  帧 {int(e):>6}-{int(e+roll.T/8):>6}: {'#'*h} ({h})")
print(f"  {'-> 集中在少数区段，参考可能有不可行时刻' if hist.max() >= 4 else '-> 分布较散，更像鲁棒性问题'}")

print()
print("=== 摔倒是突发还是渐进 ===")
sudden = sum(1 for f in real if f["tilt10"] < 30)
print(f"  摔倒前 10 步（0.2 秒）倾角仍小于 30° 的: {sudden}/{len(real)}")
print(f"  {'-> 多为突发失稳（0.2 秒内从正常到倒地）' if sudden > len(real)*0.6 else '-> 多为渐进倾倒，有征兆'}")
PYEOF
