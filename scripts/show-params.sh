#!/usr/bin/env bash
cd "$HOME/tools/rl" || exit 1
source .venv/bin/activate
export JAX_PLATFORMS=cpu
python - <<'PYEOF' 2>&1 | grep -vE 'UserWarning|warnings.warn|Failed to import|cuda|CUDA'
import numpy as np, mujoco
import rl_env

# 原始模型（未经我们改动）和当前配置各取一份
m0 = mujoco.MjModel.from_xml_path(str(rl_env.XML))
m1 = rl_env.configure_model(mujoco.MjModel.from_xml_path(str(rl_env.XML)))

def jname(m, i):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i,0])

SAMPLE = ["left_hip_pitch_joint","left_knee_joint","left_ankle_pitch_joint",
          "left_ankle_roll_joint","left_shoulder_pitch_joint","left_elbow_joint",
          "left_wrist_roll_joint","waist_yaw_joint"]
idx = {}
for i in range(m0.nu):
    n = jname(m0, i)
    if n in SAMPLE:
        idx[n] = i

print("="*78)
print("第一层：关节本身的物理属性（<joint>）—— 描述机械结构，通常不该乱动")
print("="*78)
print(f"{'关节':<26}{'限位(度)':>16}{'armature':>10}{'damping':>9}{'frictionloss':>13}")
for n in SAMPLE:
    i = idx[n]
    jid = m0.actuator_trnid[i,0]
    lo, hi = np.degrees(m0.jnt_range[jid])
    print(f"{n:<26}{f'[{lo:.0f}, {hi:.0f}]':>16}"
          f"{m0.dof_armature[m0.jnt_dofadr[jid]]:>10.3f}"
          f"{m0.dof_damping[m0.jnt_dofadr[jid]]:>9.3f}"
          f"{m0.dof_frictionloss[m0.jnt_dofadr[jid]]:>13.3f}")

print()
print(f"  armature      电机转子惯量折算到关节侧。全模型统一 {m0.dof_armature[6]:.2f}")
print(f"  damping       关节被动阻尼（机械摩擦/润滑），与伺服 kd 是两回事")
print(f"  frictionloss  库仑摩擦（与速度无关的恒定阻力）")

print()
print("="*78)
print("第二层：执行器 / 控制器（<actuator>）—— 我们一直在调的就是这层")
print("="*78)
GAIN = {0:"FIXED"}; BIAS = {0:"NONE",1:"AFFINE"}
print(f"  执行器类型: gaintype={GAIN.get(m0.actuator_gaintype[0])} "
      f"biastype={BIAS.get(m0.actuator_biastype[0])} -> 位置伺服")
print(f"  控制律: tau = kp*(ctrl - q) - kd*qd，然后被 forcerange 和 jnt_actfrcrange 截断")
print()
print(f"{'关节':<26}{'kp原':>7}{'kp今':>7}{'kd原':>8}{'kd今':>8}{'力矩上限':>10}{'ctrlrange(度)':>16}")
for n in SAMPLE:
    i = idx[n]
    jid = m0.actuator_trnid[i,0]
    lo, hi = np.degrees(m0.actuator_ctrlrange[i])
    print(f"{n:<26}{m0.actuator_gainprm[i,0]:>7.0f}{m1.actuator_gainprm[i,0]:>7.0f}"
          f"{-m0.actuator_biasprm[i,2]:>8.2f}{-m1.actuator_biasprm[i,2]:>8.2f}"
          f"{np.abs(m0.jnt_actfrcrange[jid]).max():>10.0f}"
          f"{f'[{lo:.0f}, {hi:.0f}]':>16}")
print()
print(f"  kp              位置增益，越大跟得越死（我们 500 -> 250）")
print(f"  kd              速度增益/阻尼，越大越粘滞（我们改成按惯量算的 per-joint）")
print(f"  jnt_actfrcrange 关节能承受的力矩上限，硬件规格，不该改")
print(f"  actuator_forcerange 全零 = 不额外限制（由上面那个管）")
print(f"  gear            传动比，当前全为 {m0.actuator_gear[0,0]:.0f}")

print()
print("="*78)
print("第三层：接触（<geom> / <pair>）—— 影响脚地交互")
print("="*78)
gid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_GEOM, "left_foot1_collision")
fid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_GEOM, "floor")
for p in range(m0.npair):
    if {m0.pair_geom1[p], m0.pair_geom2[p]} == {gid, fid}:
        print(f"  脚-地 pair:  condim={m0.pair_dim[p]}  friction={m0.pair_friction[p][:3]}")
        print(f"               solref={m0.pair_solref[p]}   solimp={m0.pair_solimp[p]}")
        break
print(f"  condim   接触维度：1=无摩擦 3=有切向摩擦 4/6=含扭转/滚动")
print(f"  friction [滑动, 滑动, 自旋, 滚动, 滚动]")
print(f"  solref   [时间常数, 阻尼比] —— 接触的软硬，时间常数越小越硬")
print(f"  solimp   接触约束的刚度曲线")

print()
print("="*78)
print("第四层：全局仿真设置（<option>）")
print("="*78)
INT = {0:"EULER",1:"RK4",2:"IMPLICIT",3:"IMPLICITFAST"}
print(f"  timestep      {m1.opt.timestep}  （物理步长，我们用 0.004 = 250Hz）")
print(f"  integrator    {INT.get(int(m1.opt.integrator))}")
print(f"  solver        NEWTON   iterations={m1.opt.iterations}  ls_iterations={m1.opt.ls_iterations}")
print(f"  gravity       {m1.opt.gravity}")
print(f"  总质量        {m1.body_mass.sum():.2f} kg")
PYEOF
