#!/usr/bin/env bash
cd "$HOME/tools/rl" || exit 1
source .venv/bin/activate
cp /mnt/d/g1-imitation/src/rl/rl_env.py /mnt/d/g1-imitation/src/rl/rl_train.py /mnt/d/g1-imitation/src/rl/rl_play.py \
   /mnt/d/g1-imitation/src/rl/rl_eval.py /mnt/d/g1-imitation/src/rl/rl_eval_mjx.py .
export JAX_PLATFORMS=cpu
python - <<'PYEOF' 2>&1 | grep -vE 'UserWarning|warnings.warn|Failed to import|cuda|CUDA'
import numpy as np, mujoco
import rl_env, rl_play

m = rl_env.configure_model(mujoco.MjModel.from_xml_path(str(rl_env.XML)))
kd = -m.actuator_biasprm[:, 2]
print(f"=== 新的 per-joint kd（KD_RATIO={rl_env.KD_RATIO}）===")
print(f"  范围 [{kd.min():.2f}, {kd.max():.2f}]   旧的是常数 20.2")
for nm in ("left_wrist_roll_joint","left_elbow_joint","left_knee_joint","left_hip_pitch_joint"):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
    i = m.jnt_dofadr[jid]-6
    print(f"  {nm:<26} kd={kd[i]:6.2f}   （旧 20.2，比值 {20.2/kd[i]:.2f}x）")

print()
print("=== 稳定性回归：静态站立 5 秒（当初引入常数 kd 就是为了这个）===")
d = mujoco.MjData(m)
k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(m, d, k); d.ctrl[:] = m.key_ctrl[k]
ok, t = True, 5.0
for i in range(1250):
    mujoco.mj_step(m, d)
    if d.qpos[2] < 0.2 or 1-2*(d.qpos[4]**2+d.qpos[5]**2) < 0:
        ok, t = False, i*0.004; break
print(f"  {'通过，站住 5 秒' if ok else f'失败，{t:.2f}s 倒地'}   末态 z={d.qpos[2]:.3f}")

print()
print("=== 零动作前馈（旧配置基线 149.4 / 5 起点）===")
ref, refv = rl_env.load_reference(["walk1_subject1"])
refn, refvn = np.asarray(ref[0]), np.asarray(refv[0])
roll = rl_play.NumpyRollout(refn, refvn)
out = []
for s in (0, 3000, 6000, 9000, 12000):
    roll.reset(s, refvn); n = 0
    for i in range(500):
        roll.apply(np.zeros(rl_env.NU), s+i); n = i+1
        if roll.fell(): break
    out.append(n)
print(f"  存活 {out}   均值 {np.mean(out):.1f}")
PYEOF
