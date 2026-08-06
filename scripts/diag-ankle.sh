#!/usr/bin/env bash
cd "$HOME/tools/rl" || exit 1
source .venv/bin/activate
export JAX_PLATFORMS=cpu
python - <<'PYEOF' 2>&1 | grep -vE 'UserWarning|warnings.warn|Failed to import|cuda|CUDA'
import pathlib, numpy as np, mujoco, jax
import rl_env, rl_play

m = rl_env.configure_model(mujoco.MjModel.from_xml_path(str(rl_env.XML)))
kd = -m.actuator_biasprm[:, 2]

print("=== 腿部关节的阻尼变化（v9 常数 20.2 -> v10 per-joint）===")
print(f"  {'关节':<28} {'v10 kd':>8} {'v9 kd':>8} {'变化':>8}")
for nm in ("left_ankle_roll_joint","left_ankle_pitch_joint","left_knee_joint",
           "left_hip_pitch_joint","left_hip_roll_joint","left_elbow_joint"):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
    i = m.jnt_dofadr[jid]-6
    print(f"  {nm:<28} {kd[i]:>8.2f} {20.2:>8.1f} {kd[i]/20.2:>7.2f}x")

print()
print("=== CPU 引擎下的实际存活（就是你在 viewer 里看到的）===")
ck = pathlib.Path.home()/"tools"/"rl"/"runs"/"walk"/"policy.pkl"
pol = rl_play.build_policy(ck, rl_env.OBS_SIZE, rl_env.NU)
key = jax.random.PRNGKey(0)
ref, refv = rl_env.load_reference(["walk1_subject1"])
refn, refvn = np.asarray(ref[0]), np.asarray(refv[0])
roll = rl_play.NumpyRollout(refn, refvn)
out = []
for s in (0, 1794, 3588, 5382, 7177, 8971, 10765, 12560):
    roll.reset(s, refvn); last = np.zeros(rl_env.NU); n = 0
    for i in range(500):
        a = np.asarray(pol(roll.obs(s+i, last), key)[0])
        roll.apply(a, s+i); last = a; n = i+1
        if roll.fell(): break
    out.append(n)
print(f"  各起点 {out}")
print(f"  均值 {np.mean(out):.1f} 步 = {np.mean(out)*0.02:.1f} 秒")
print(f"  （v9 同口径为 275.9 步 / MJX；CPU 与 MJX 不同引擎，只看 v10 自身）")
PYEOF
