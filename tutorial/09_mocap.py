"""第 9 课：用动捕数据驱动模型

"动捕驱动"有四种截然不同的做法，复杂度差好几个数量级。
这一课把四种都跑一遍，让你知道自己需要哪一种。

    1. mocap body      MuJoCo 原生机制。无限刚性的运动学物体，
                       用来当"目标点"或拖拽物体。最简单。
    2. qpos 直接回放    把关节角逐帧写进 qpos。纯运动学，没有物理，
                       机器人会穿模、悬空。用来快速看动作对不对。
    3. IK + PD 跟踪     动捕给末端位置 -> 解 IK 得关节角 -> 当 PD 目标。
                       物理生效，但机器人可能站不住。
    4. RL 模仿          训一个策略去跟踪参考动作。唯一能让人形真正
                       走起来的方法，但要 GPU 训几小时到几天。

运行: python 09_mocap.py
"""

import pathlib

import mujoco
import numpy as np

OUT = pathlib.Path(__file__).parent / "out"


# ---------------------------------------------------------------- 1. mocap body
MOCAP_XML = """
<mujoco model="mocap_demo">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" size="3 3 0.1"/>

    <!-- mocap body：不受物理影响，位置由 data.mocap_pos 直接指定 -->
    <body name="target" mocap="true" pos="0.3 0 0.6">
      <geom type="sphere" size="0.04" rgba="1 0 0 0.5" contype="0" conaffinity="0"/>
    </body>

    <!-- 普通刚体，会被 weld 约束拉向 mocap body -->
    <body name="cube" pos="0.3 0 0.6">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" rgba="0.2 0.6 0.9 1" mass="0.5"/>
    </body>
  </worldbody>

  <equality>
    <!-- weld：把 cube 焊到 target 上。这是"动捕拖动物体"的标准做法 -->
    <weld body1="cube" body2="target" solref="0.02 1"/>
  </equality>
</mujoco>
"""


def demo_mocap_body():
    print("=" * 72)
    print("1. mocap body —— MuJoCo 的原生动捕接口")
    print("=" * 72)
    m = mujoco.MjModel.from_xml_string(MOCAP_XML)
    d = mujoco.MjData(m)

    print(f"  模型里的 mocap body 数: m.nmocap = {m.nmocap}")
    print(f"  它们的状态存在 data.mocap_pos (shape {d.mocap_pos.shape}) "
          f"和 data.mocap_quat (shape {d.mocap_quat.shape})")
    print()
    print("  关键性质：mocap body 不受重力、不受碰撞，**你写什么它就在哪**。")
    print("  它不占 qpos/qvel —— 不是自由度，是外部输入。")
    print(f"    nq={m.nq} nv={m.nv}  <- 只有 cube 的 freejoint(7/6)，target 不占")
    print()
    print("  真实用途：")
    print("    - 动捕标记点：每帧把 marker 世界坐标写进 mocap_pos")
    print("    - VR 手柄 / 遥操作目标位姿")
    print("    - 用 weld/connect 约束把机器人末端拉向它")

    print()
    print("  演示：让 target 画圆，看被 weld 的 cube 跟不跟得上")
    print(f"  {'时间':>6}{'target x':>10}{'target z':>10}{'cube x':>9}{'cube z':>9}{'误差':>9}")
    print("  " + "-" * 53)
    for step in range(1501):
        t = d.time
        # 这一行就是"喂动捕数据"——真实场景里换成从文件/网络读来的位姿
        d.mocap_pos[0] = [0.3 * np.cos(2 * t), 0.3 * np.sin(2 * t), 0.6]
        mujoco.mj_step(m, d)
        if step % 300 == 0:
            err = np.linalg.norm(d.qpos[:3] - d.mocap_pos[0])
            print(f"  {t:>6.2f}{d.mocap_pos[0][0]:>10.3f}{d.mocap_pos[0][2]:>10.3f}"
                  f"{d.qpos[0]:>9.3f}{d.qpos[2]:>9.3f}{err:>9.4f}")
    print()
    print("  cube 紧跟 target，误差极小 —— weld 约束在做这件事。")
    print("  想要柔顺一点（碰到障碍会让步），把 solref 第一个数调大。")


# ---------------------------------------------------------------- 2. qpos 回放
def demo_qpos_playback():
    print()
    print("=" * 72)
    print("2. qpos 直接回放 —— 最快看到动作，但没有物理")
    print("=" * 72)
    p = pathlib.Path(__file__).parent.parent / "mujoco_menagerie" / "unitree_go2" / "scene.xml"
    if not p.exists():
        print("  menagerie 未找到，跳过")
        return
    m = mujoco.MjModel.from_xml_path(str(p))
    d = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    q_home = m.key_qpos[kid].copy()

    # 造一段"动捕轨迹"：真实场景里这是从 BVH/CSV/npz 读来的 (T, nq) 数组
    T = 200
    traj = np.tile(q_home, (T, 1))
    ts = np.linspace(0, 2 * np.pi, T)
    traj[:, 2] = q_home[2] + 0.06 * np.sin(ts)          # 躯干上下起伏
    traj[:, 8::3] = q_home[8] + 0.3 * np.sin(ts)[:, None]   # 大腿关节摆动

    print(f"  参考轨迹 shape = {traj.shape}   (帧数, nq)")
    print("  回放只有两行：")
    print("    d.qpos[:] = traj[i]")
    print("    mujoco.mj_forward(m, d)     # 注意是 forward 不是 step")
    print()
    print("  mj_forward 只更新运动学，**不做积分、不算接触力**。")
    print("  所以机器人会完全按数据走，哪怕穿进地面也照样穿。")

    penetrate = 0
    for i in range(T):
        d.qpos[:] = traj[i]
        mujoco.mj_forward(m, d)
        if d.qpos[2] < 0.22:
            penetrate += 1
    print()
    print(f"  这段轨迹里有 {penetrate}/{T} 帧躯干低于 0.22m（正常站立高度 0.27）")
    print("  纯运动学下这不会有任何后果 —— 这既是优点也是缺点：")
    print("    优点：动作一定和数据一致，用来验证「retarget 对不对」")
    print("    缺点：物理上可能根本不可行（脚穿地、力矩超限）")


# ---------------------------------------------------------------- 3. IK + PD
def demo_ik_tracking():
    print()
    print("=" * 72)
    print("3. IK + PD 跟踪 —— 动捕给末端位置，物理生效")
    print("=" * 72)
    print("  流程: 动捕末端位姿 -> 逆运动学解关节角 -> 当 PD 目标 -> mj_step")
    print()
    print("  MuJoCo 没有内置 IK 求解器，但提供雅可比，几行就能写阻尼最小二乘:")
    print("    mujoco.mj_jacSite(m, d, jacp, jacr, site_id)")
    print("    dq = Jᵀ (J Jᵀ + λ²I)⁻¹ · (目标位置 - 当前位置)")

    XML = """
    <mujoco>
      <compiler angle="radian"/>
      <option gravity="0 0 -9.81" timestep="0.002"/>
      <worldbody>
        <light pos="0 0 3"/>
        <geom type="plane" size="2 2 .1"/>
        <body name="target" mocap="true" pos="0.4 0 0.5">
          <geom type="sphere" size="0.03" rgba="1 0 0 .6" contype="0" conaffinity="0"/>
        </body>
        <body pos="0 0 0.1">
          <geom type="cylinder" size="0.06 0.1"/>
          <body pos="0 0 0.1">
            <!-- 三个关节都写 range：无限位关节的 jnt_range 是 [0,0]，
                 拿它去 clip 会把自由度直接焊死（我第一版就踩了这个坑） -->
            <joint name="j1" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="1"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.04"/>
            <body pos="0 0 0.3">
              <joint name="j2" type="hinge" axis="0 1 0" range="-2 2" damping="1"/>
              <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.035"/>
              <body pos="0.3 0 0">
                <joint name="j3" type="hinge" axis="0 1 0" range="-2.5 2.5" damping="1"/>
                <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.03"/>
                <site name="tip" pos="0.25 0 0" size="0.02" rgba="0 1 0 1"/>
              </body>
            </body>
          </body>
        </body>
      </worldbody>
      <actuator>
        <position joint="j1" kp="120" kv="12"/>
        <position joint="j2" kp="120" kv="12"/>
        <position joint="j3" kp="120" kv="12"/>
      </actuator>
    </mujoco>"""

    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)          # 物理状态
    d_ik = mujoco.MjData(m)       # IK 专用的运动学草稿，不碰物理状态
    tip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tip")

    # 只对真正有限位的关节做 clip
    limited = m.jnt_limited.astype(bool)
    lo = np.where(limited, m.jnt_range[:, 0], -np.inf)
    hi = np.where(limited, m.jnt_range[:, 1], np.inf)
    print()
    print(f"  jnt_limited = {m.jnt_limited}   -> 只对有限位的关节 clip")

    def solve_ik(target, q_init, iters=8, damping=0.1):
        """在草稿 data 上迭代求解，返回关节角。不影响物理状态。"""
        q = q_init.copy()
        jacp = np.zeros((3, m.nv))
        for _ in range(iters):
            d_ik.qpos[:] = q
            mujoco.mj_forward(m, d_ik)
            err = target - d_ik.site_xpos[tip]
            if np.linalg.norm(err) < 1e-5:
                break
            mujoco.mj_jacSite(m, d_ik, jacp, None, tip)
            JJt = jacp @ jacp.T + damping ** 2 * np.eye(3)
            q = np.clip(q + jacp.T @ np.linalg.solve(JJt, err), lo, hi)
        return q

    print()
    print("  演示：末端跟踪一个画圆的 mocap 目标")
    print(f"  {'时间':>6}{'目标 (x, y, z)':>26}{'末端 (x, y, z)':>26}{'误差(mm)':>10}")
    print("  " + "-" * 68)

    q_cmd = d.qpos.copy()
    errors = []
    for step in range(2001):
        t = d.time
        target = np.array([0.35 + 0.12 * np.cos(1.5 * t),
                           0.12 * np.sin(1.5 * t),
                           0.42 + 0.08 * np.sin(1.5 * t)])
        d.mocap_pos[0] = target

        q_cmd = solve_ik(target, q_cmd)   # 解 IK（不动物理状态）
        d.ctrl[:] = q_cmd                 # 位置伺服：ctrl 就是目标角
        mujoco.mj_step(m, d)              # 物理推进

        e = float(np.linalg.norm(d.site_xpos[tip] - target))
        if t > 0.5:
            errors.append(e)              # 跳过起步瞬态
        if step % 400 == 0:
            print(f"  {t:>6.2f}   ({target[0]:>6.3f},{target[1]:>6.3f},{target[2]:>6.3f})"
                  f"   ({d.site_xpos[tip][0]:>6.3f},{d.site_xpos[tip][1]:>6.3f},"
                  f"{d.site_xpos[tip][2]:>6.3f}){e*1000:>10.2f}")

    print()
    print(f"  稳态跟踪误差: 平均 {np.mean(errors)*1000:.2f} mm，"
          f"最大 {np.max(errors)*1000:.2f} mm")
    print("  y 方向也跟上了 —— 三个自由度都在工作。")
    print()
    print("  两个实现要点:")
    print("    1. IK 在**独立的 MjData** 上迭代，别在物理状态上改 qpos 再 step")
    print("       —— 那等于每帧把机器人瞬移，动力学就废了")
    print("    2. clip 只对 jnt_limited 为真的关节做")
    print("       —— 无限位关节的 jnt_range 是 [0,0]，误 clip 会焊死自由度")
    print()
    print("  真实场景把 target 换成动捕流即可。想要更完整的 IK（含姿态、")
    print("  多目标、碰撞规避），用 mink 库，别自己造。")


# ---------------------------------------------------------------- 4. RL 模仿
def explain_rl_imitation():
    print()
    print("=" * 72)
    print("4. RL 模仿 —— 人形行走唯一实用的路")
    print("=" * 72)
    print("  前三种对机械臂/物体够用，但对**双足人形**都会倒。")
    print("  原因回顾第 8 课：双足站立是不稳定平衡点，")
    print("  照着动捕角度做 PD 跟踪，抵不住整体倾覆。")
    print()
    print("  实用方案是 DeepMimic 式的 RL 模仿:")
    print("    观测 = 机器人本体状态 + 参考动作的未来几帧")
    print("    奖励 = 关节角相似度 + 末端位置相似度 + 姿态 + 存活")
    print("    动作 = 关节目标角（叠加在参考动作上）")
    print("    训练 = PPO，数千并行环境，GPU 上几小时到几天")
    print()
    print("  你的 RTX 4060 跑 MJX 可以并行约 2000~4000 个环境。")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    demo_mocap_body()
    demo_qpos_playback()
    demo_ik_tracking()
    explain_rl_imitation()

    print()
    print("=" * 72)
    print("选哪一种？")
    print("=" * 72)
    print("  只想看动作对不对              -> 2. qpos 回放")
    print("  拖动物体 / 遥操作目标位姿      -> 1. mocap body + weld")
    print("  机械臂跟踪末端轨迹             -> 3. IK + PD")
    print("  人形/四足真的走起来            -> 4. RL 模仿")
    print()
    print("  另外：人体动捕 -> 机器人，中间必须做 retarget（骨骼不一样长、")
    print("  自由度对不上）。别自己写，用 GMR 这类现成工具。")


if __name__ == "__main__":
    main()
