"""第 8 课：真实机器人 —— 把前 7 课串起来

拿 menagerie 的宇树 Go2，走一遍完整流程：
    加载模型 -> 读结构 -> 设初始姿势 -> 写控制器 -> 跑仿真 -> 出视频

这一课的重点是**流程**，不是控制算法。真正的行走控制器要么是
MPC（模型预测控制），要么是 RL 训出来的策略网络，都远超本课范围。
但你会看到它们插进来的位置在哪。

运行:
    python 08_real_robot.py            # 全程离屏，产出 GIF（默认）
    python 08_real_robot.py --view     # 最后额外开一个交互窗口看实时效果
"""

import os
import pathlib
import sys

import imageio.v3 as iio
import mujoco
import mujoco.viewer
import numpy as np

ROOT = pathlib.Path(__file__).parent.parent
OUT = pathlib.Path(__file__).parent / "out"
MODEL_PATH = ROOT / "mujoco_menagerie" / "unitree_go2" / "scene.xml"


def inspect(model: mujoco.MjModel) -> None:
    print("=" * 70)
    print("1. 读懂模型结构（第 1~3 课的内容）")
    print("=" * 70)
    print(f"  刚体 {model.nbody}   几何体 {model.ngeom}   关节 {model.njnt}   执行器 {model.nu}")
    print(f"  nq={model.nq}  nv={model.nv}")
    print(f"    = freejoint(7/6) + {model.nq-7} 个腿关节")
    print()
    print("  自由度分区（写控制器时反复用到）：")
    print(f"    qpos[0:3]  躯干位置        qpos[3:7]  躯干姿态四元数")
    print(f"    qpos[7:]   12 个关节角     qvel[0:3]  躯干线速度")
    print(f"    qvel[3:6]  躯干角速度      qvel[6:]   12 个关节速度")
    print()
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             for i in range(model.nu)]
    print("  执行器（每条腿 3 个：髋外展 / 大腿 / 小腿）：")
    for i in range(0, model.nu, 3):
        print(f"    {' '.join(f'{n:<12}' for n in names[i:i+3])}")


def check_actuator(model: mujoco.MjModel) -> str:
    print()
    print("=" * 70)
    print("2. 判断执行器类型（第 4 课的核心，写错就全废）")
    print("=" * 70)
    is_pos = (model.actuator_biastype[0] == mujoco.mjtBias.mjBIAS_AFFINE
              and model.actuator_biasprm[0][1] != 0)
    kind = "position 位置伺服" if is_pos else "motor 力矩电机"
    print(f"  biastype[0] = {model.actuator_biastype[0]}  ->  {kind}")
    print(f"  ctrlrange   = {model.actuator_ctrlrange[0]}  "
          f"({'rad' if is_pos else 'N·m'})")
    if is_pos:
        print(f"  内置 kp={-model.actuator_biasprm[0][1]:.0f} "
              f"kv={-model.actuator_biasprm[0][2]:.0f}")
        print("  -> ctrl 直接写目标角度即可")
    else:
        print("  -> ctrl 是力矩，要保持姿势必须自己写 PD")
    return kind


def initial_pose(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    print()
    print("=" * 70)
    print("3. 设置初始姿势 —— keyframe")
    print("=" * 70)
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    print(f"  模型有 {model.nkey} 个 keyframe，'home' 的 id = {kid}")
    mujoco.mj_resetDataKeyframe(model, data, kid)
    q_home = model.key_qpos[kid][7:].copy()
    print(f"  躯干高度 z = {data.qpos[2]:.3f} m")
    print(f"  每条腿的关节角 = {np.round(q_home[:3], 3)} (髋, 大腿, 小腿)")
    print()
    print("  对比：直接 mj_resetData 会回到 qpos0（关节全 0，腿绷直）")
    d2 = mujoco.MjData(model)
    mujoco.mj_resetData(model, d2)
    print(f"    qpos0 的关节角 = {np.round(d2.qpos[7:10], 3)}  <- 不是站立姿势")
    return q_home


def no_control(model, data, q_home):
    print()
    print("=" * 70)
    print("4. 对照组：不加控制会怎样")
    print("=" * 70)
    mujoco.mj_resetDataKeyframe(model, data,
                                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
    z0 = data.qpos[2]
    for _ in range(1000):
        data.ctrl[:] = 0
        mujoco.mj_step(model, data)
    print(f"  ctrl 全 0，跑 2 秒：躯干高度 {z0:.3f} -> {data.qpos[2]:.3f}")
    print("  瘫倒了。这是**正确的物理**：没有电机出力，腿撑不住自重。")
    print("  真实的 Go2 断电时也是这样趴下去的。")


def pd_stand(model, data, q_home, kp=100.0, kd=5.0):
    print()
    print("=" * 70)
    print("5. 加 PD 控制器让它站住")
    print("=" * 70)
    print(f"  tau = kp*(q_target - q) - kd*dq    kp={kp:g} kd={kd:g}")
    mujoco.mj_resetDataKeyframe(model, data,
                                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    z0 = data.qpos[2]
    print()
    print(f"  {'时间(s)':>8}{'躯干高度':>10}{'最大关节误差(rad)':>18}{'最大力矩(N·m)':>15}")
    print("  " + "-" * 51)
    for step in range(2501):
        q, dq = data.qpos[7:], data.qvel[6:]
        tau = kp * (q_home - q) - kd * dq
        data.ctrl[:] = np.clip(tau, lo, hi)
        if step % 500 == 0:
            print(f"  {data.time:>8.2f}{data.qpos[2]:>10.4f}"
                  f"{np.max(np.abs(q_home - q)):>18.4f}"
                  f"{np.max(np.abs(data.ctrl)):>15.2f}")
        mujoco.mj_step(model, data)
    print()
    print(f"  站住了：高度从 {z0:.3f} 稳定在 {data.qpos[2]:.3f}")
    print("  这就是最简单的「站立控制器」。行走则需要在 q_target 上加时变的步态。")


def read_state(model, data):
    print()
    print("=" * 70)
    print("6. 读状态 —— RL 的观测向量长什么样")
    print("=" * 70)
    quat = data.qpos[3:7]
    # 把重力方向投影到机身系：这是四足 RL 最常用的姿态表示
    grav = np.zeros(3)
    mujoco.mju_rotVecQuat(grav, np.array([0.0, 0.0, -1.0]),
                          np.array([quat[0], -quat[1], -quat[2], -quat[3]]))
    obs = np.concatenate([
        grav,                    # 3  机身系下的重力方向（等价于 roll/pitch）
        data.qvel[3:6],          # 3  机身角速度（陀螺仪）
        data.qpos[7:],           # 12 关节角（编码器）
        data.qvel[6:],           # 12 关节速度
        data.ctrl,               # 12 上一步动作
    ])
    print(f"  典型观测维度 = 3 + 3 + 12 + 12 + 12 = {obs.shape[0]}")
    print(f"    重力投影   {np.round(grav, 3)}   <- 站直时应接近 [0,0,-1]")
    print(f"    机身角速度 {np.round(data.qvel[3:6], 3)}")
    print(f"    关节角前3  {np.round(data.qpos[7:10], 3)}")
    print()
    print("  注意这些量**真机全都有**（IMU + 编码器），所以策略能迁移。")
    print("  如果观测里混进了 qpos[0:3]（世界绝对位置），真机上没有，必翻车。")


def contacts(model, data):
    print()
    print("=" * 70)
    print("7. 足端接触 —— 判断哪只脚着地")
    print("=" * 70)
    foot_names = ["FL", "FR", "RL", "RR"]
    print(f"  当前接触点数 = {data.ncon}")
    forces = {}
    f6 = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        for g in (c.geom1, c.geom2):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            for foot in foot_names:
                if gname.startswith(foot):
                    mujoco.mj_contactForce(model, data, i, f6)
                    forces[foot] = forces.get(foot, 0.0) + f6[0]
    print()
    print(f"  {'足':>6}{'法向力(N)':>12}   状态")
    print("  " + "-" * 32)
    for foot in foot_names:
        f = forces.get(foot, 0.0)
        print(f"  {foot:>6}{f:>12.2f}   {'着地' if f > 1 else '腾空'}")
    total = sum(forces.values())
    weight = float(np.sum(model.body_mass)) * 9.81
    print("  " + "-" * 32)
    print(f"  {'合计':>6}{total:>12.2f}   整机重量 {weight:.2f} N")
    print()
    print("  四足站立时四条腿分担体重。行走时着地的腿会交替变化，")
    print("  这个信号就是步态相位的依据。")


def record(model, data, q_home, kp=100.0, kd=5.0):
    print()
    print("=" * 70)
    print("8. 录一段视频（第 6 课的内容）")
    print("=" * 70)
    OUT.mkdir(exist_ok=True)
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 540
    mujoco.mj_resetDataKeyframe(model, data,
                                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]

    FPS = 30
    every = max(1, round(1.0 / (FPS * model.opt.timestep)))
    frames = []
    with mujoco.Renderer(model, 540, 960) as r:
        for step in range(int(4.0 / model.opt.timestep)):
            t = data.time
            # 站立姿势上叠加一个正弦"下蹲-起立"，让画面有东西看
            wave = 0.25 * np.sin(2 * np.pi * 0.5 * t)
            target = q_home.copy()
            target[1::3] += wave          # 大腿关节
            target[2::3] -= wave * 2      # 小腿关节
            tau = kp * (target - data.qpos[7:]) - kd * data.qvel[6:]
            data.ctrl[:] = np.clip(tau, lo, hi)
            mujoco.mj_step(model, data)
            if step % every == 0:
                r.update_scene(data, camera=-1)
                frames.append(r.render())
    path = OUT / "08_go2_squat.gif"
    iio.imwrite(path, np.stack(frames), duration=1000 / FPS, loop=0)
    print(f"  已保存 {path.name}  {len(frames)} 帧  {path.stat().st_size/1024:.0f} KB")
    print(f"  最终躯干高度 {data.qpos[2]:.3f} m")
    print()
    print("  这个正弦「下蹲-起立」就是最原始的『步态』——")
    print("  真正的行走是给 12 个关节设计相位差合适的周期轨迹，")
    print("  或者干脆让 RL 去学 target 该是多少。")


def live_view(model, data, q_home, kp=100.0, kd=5.0):
    """开一个交互窗口，用 PD 控制器让 Go2 站着，同时做正弦下蹲。

    和上面 record() 的物理完全一样，只是把画面送到窗口而不是 GIF。
    这就是「离屏渲染」和「交互式 viewer」的唯一区别——控制逻辑一行都不用改。
    """
    print()
    print("=" * 70)
    print("9. 交互式窗口（--view）")
    print("=" * 70)
    mujoco.mj_resetDataKeyframe(model, data,
                                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]

    print("  鼠标左键拖动旋转视角，右键平移，滚轮缩放")
    print("  关闭窗口即结束")

    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            wave = 0.25 * np.sin(2 * np.pi * 0.5 * data.time)
            target = q_home.copy()
            target[1::3] += wave
            target[2::3] -= wave * 2
            tau = kp * (target - data.qpos[7:]) - kd * data.qvel[6:]
            data.ctrl[:] = np.clip(tau, lo, hi)
            mujoco.mj_step(model, data)
            v.sync()
    print("  窗口已关闭")


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"模型不存在: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    inspect(model)
    check_actuator(model)
    q_home = initial_pose(model, data)
    no_control(model, data, q_home)
    pd_stand(model, data, q_home)
    read_state(model, data)
    contacts(model, data)
    record(model, data, q_home)

    if "--view" in sys.argv[1:]:
        live_view(model, data, q_home)

    print()
    print("=" * 70)
    print("完整流程回顾")
    print("=" * 70)
    print("  1. from_xml_path 加载       -> 第 1 课 MjModel/MjData")
    print("  2. 看 nq/nv/nu 分区         -> 第 3 课 自由度")
    print("  3. 判断执行器类型           -> 第 4 课 ctrl 单位")
    print("  4. mj_resetDataKeyframe     -> 初始姿势")
    print("  5. 循环里写 data.ctrl       -> 你的控制器插在这里")
    print("  6. mj_step 推进             -> 第 1 课")
    print("  7. 读 sensordata / contact  -> 第 5、7 课")
    print("  8. Renderer 出图出视频      -> 第 6 课")
    print()
    print("  下一步想做什么：")
    print("    行走控制  -> 学 MPC 或直接上 RL")
    print("    RL 训练   -> uv pip install \"mujoco[mjx]\" jax[cuda12]")
    print("                 MJX 能在 GPU 上并行几千个环境")
    print("    标准环境  -> uv pip install gymnasium[mujoco]")
    if "--view" not in sys.argv[1:]:
        print()
        print("  想看实时窗口而不是 GIF: python 08_real_robot.py --view")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)   # 规避 WSLg d3d12 在解释器退出时清理 GL 上下文的段错误
