#!/usr/bin/env bash
cd "$HOME/tools/rl" || exit 1
source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python - <<'PYEOF' 2>&1 | grep -vE 'UserWarning|warnings.warn|Failed to import|cuda_timer'
"""深蹲段在 MJX 与 CPU 下的差异，以及摔倒瞬间发生了什么。"""
import pathlib, numpy as np, mujoco, jax, jax.numpy as jp
import jax_compat  # noqa
import rl_env, rl_play
from brax.envs.base import State

CK = pathlib.Path.home()/"tools"/"rl"/"runs"/"walk_v11_final"/"policy.pkl"
S = 12060
ref, refv = rl_env.load_reference(["walk1_subject1"])
refn, refvn = np.asarray(ref[0]), np.asarray(refv[0])
pol = rl_play.build_policy(CK, rl_env.OBS_SIZE, rl_env.NU)
key = jax.random.PRNGKey(0)

# ---------- CPU ----------
roll = rl_play.NumpyRollout(refn, refvn)
m = roll.m
FID = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
       for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
roll.reset(S, refvn)
last = np.zeros(rl_env.NU)
log = []
for i in range(400):
    a = np.asarray(pol(roll.obs(S+i, last), key)[0])
    roll.apply(a, S+i); last = a
    up = 1.0 - 2.0*(roll.d.qpos[4]**2 + roll.d.qpos[5]**2)
    fz = roll.d.xpos[FID][:, 2]
    log.append(dict(i=i+1, z=roll.d.qpos[2],
                    tilt=np.degrees(np.arccos(np.clip(up,-1,1))),
                    nfoot=int((fz < 0.10).sum()),
                    ncon=roll.d.ncon,
                    jerr=np.degrees(np.abs(roll.d.qpos[7:]-refn[min(S+i+1,len(refn)-1),7:]).mean())))
    if roll.fell():
        break
cpu_n = len(log)

# ---------- MJX ----------
env = rl_env.G1Imitate(ref, refv, ep_len=1500)
q, v = env._ref_at(0, S)
data = env.pipeline_init(q, v)
st = State(data, env._obs(data, 0, S, jp.zeros(rl_env.NU)),
           jp.zeros(()), jp.zeros(()),
           {k: jp.zeros(()) for k in ("r_pose","r_orient","r_root","r_rvel",
                                       "r_jvel","r_alive","r_effort",
                                       "pose_err","root_err","rvel_err")},
           {"clip": jp.int32(0), "start": jp.int32(S), "step": jp.int32(S),
            "last_act": jp.zeros(rl_env.NU), "rng": key})
step_fn = jax.jit(env.step)
mjx_n = 400
for i in range(400):
    a = pol(st.obs, key)[0]
    st = step_fn(st, a)
    if float(st.done) > 0.5:
        mjx_n = i+1; break

print(f"=== 深蹲段（起点 {S}，{S*0.02:.1f}s）存活对比 ===")
print(f"  CPU MuJoCo : {cpu_n} 步 = {cpu_n*0.02:.2f} 秒")
print(f"  MJX        : {mjx_n} 步 = {mjx_n*0.02:.2f} 秒")
print(f"  比值 {mjx_n/max(cpu_n,1):.1f}x")

print()
print("=== CPU 下摔倒前的演化（每 20 步）===")
print(f"  {'步':>5} {'躯干z':>7} {'倾角':>7} {'触地脚':>7} {'接触点':>7} {'关节误差':>9}")
for r in log[::20] + [log[-1]]:
    print(f"  {r['i']:>5} {r['z']:>7.3f} {r['tilt']:>6.1f}° {r['nfoot']:>7} "
          f"{r['ncon']:>7} {r['jerr']:>8.2f}°")
PYEOF
