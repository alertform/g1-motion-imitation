"""第 3 课：关节与自由度 —— qpos / qvel 的坑

MuJoCo 有 4 种关节：

    hinge   铰链  绕轴转    nq=1  nv=1   最常用（机器人关节）
    slide   滑动  沿轴移    nq=1  nv=1
    ball    球关节 3 转     nq=4  nv=3   位置用四元数
    free    自由  6 自由度  nq=7  nv=6   位置=3平移+4四元数

**nq ≠ nv 是新手第一大坑**：只要模型里有 ball 或 free 关节，
qpos 就比 qvel 长。写控制器时用 qpos[7:] 而不是 qpos[6:]，
用 qvel[6:] 而不是 qvel[7:]，错了就全乱套。

运行: python 03_joints.py
"""

import mujoco
import numpy as np

JOINTS = """
<mujoco model="joint_types">
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="5 5 0.1"/>

    <body name="hinge_body" pos="-2 0 1">
      <joint name="j_hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.04" rgba="1 0 0 1"/>
    </body>

    <body name="slide_body" pos="-1 0 1">
      <joint name="j_slide" type="slide" axis="0 0 1" range="-0.5 0.5"/>
      <geom type="box" size="0.1 0.1 0.1" rgba="0 1 0 1"/>
    </body>

    <body name="ball_body" pos="0 0 1">
      <joint name="j_ball" type="ball"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.04" rgba="0 0 1 1"/>
    </body>

    <body name="free_body" pos="1 0 1">
      <freejoint name="j_free"/>
      <geom type="box" size="0.12 0.12 0.12" rgba="1 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

TYPE_NAME = {
    mujoco.mjtJoint.mjJNT_FREE: "free",
    mujoco.mjtJoint.mjJNT_BALL: "ball",
    mujoco.mjtJoint.mjJNT_SLIDE: "slide",
    mujoco.mjtJoint.mjJNT_HINGE: "hinge",
}
NQ = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4,
      mujoco.mjtJoint.mjJNT_SLIDE: 1, mujoco.mjtJoint.mjJNT_HINGE: 1}
NV = {mujoco.mjtJoint.mjJNT_FREE: 6, mujoco.mjtJoint.mjJNT_BALL: 3,
      mujoco.mjtJoint.mjJNT_SLIDE: 1, mujoco.mjtJoint.mjJNT_HINGE: 1}


def main() -> None:
    m = mujoco.MjModel.from_xml_string(JOINTS)
    d = mujoco.MjData(m)

    print("=" * 66)
    print("四种关节的自由度账本")
    print("=" * 66)
    print(f"  {'关节名':<12}{'类型':<8}{'qpos 起点':>10}{'占 nq':>7}{'qvel 起点':>11}{'占 nv':>7}")
    print("  " + "-" * 60)
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        t = m.jnt_type[j]
        print(f"  {name:<12}{TYPE_NAME[t]:<8}{m.jnt_qposadr[j]:>10}{NQ[t]:>7}"
              f"{m.jnt_dofadr[j]:>11}{NV[t]:>7}")
    print("  " + "-" * 60)
    print(f"  {'合计':<20}{'':>10}{m.nq:>7}{'':>11}{m.nv:>7}")
    print()
    print(f"  nq={m.nq}  nv={m.nv}   差 {m.nq - m.nv} = ball(1) + free(1)")
    print("  每个 ball/free 关节都会让 nq 比 nv 多 1（四元数的约束）")

    print()
    print("=" * 66)
    print("坑演示：拿 qvel 的下标去索引 qpos")
    print("=" * 66)
    print("  假设你想读 free_body 的速度和位置：")
    j_free = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "j_free")
    qadr, vadr = m.jnt_qposadr[j_free], m.jnt_dofadr[j_free]
    print(f"    正确: qpos[{qadr}:{qadr+7}]  qvel[{vadr}:{vadr+6}]")
    print(f"    错误: 两边都用同一个下标 —— 会读到隔壁关节的数据")
    print()
    print("  健壮写法：永远用 model.jnt_qposadr / model.jnt_dofadr 查地址，")
    print("  不要手算偏移量。")

    print()
    print("=" * 66)
    print("四元数不能随便赋值")
    print("=" * 66)
    d.qpos[qadr:qadr+7] = [0, 0, 1, 0.5, 0.5, 0.5, 0.5]   # 模长正好为 1
    mujoco.mj_forward(m, d)
    q = d.qpos[qadr+3:qadr+7]
    print(f"  合法四元数 {q}  模长={np.linalg.norm(q):.4f}")

    d.qpos[qadr+3:qadr+7] = [1, 1, 1, 1]                  # 模长 = 2，非法
    print(f"  非法赋值   [1 1 1 1]  模长={np.linalg.norm([1,1,1,1]):.4f}")
    print("  MuJoCo 不会自动归一化 —— 姿态会失真。正确做法：")
    print("    mujoco.mju_axisAngle2Quat(quat, axis, angle)   # 轴角转四元数")
    print("    mujoco.mju_euler2Quat(quat, euler, 'xyz')      # 欧拉角转四元数")
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), np.pi/4)
    print(f"    绕 z 转 45°  ->  {np.round(quat, 4)}  模长={np.linalg.norm(quat):.4f}")

    print()
    print("=" * 66)
    print("关节限位 range")
    print("=" * 66)
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if m.jnt_limited[j]:
            lo, hi = m.jnt_range[j]
            print(f"  {name:<12} 限位 [{lo:.3f}, {hi:.3f}]")
        else:
            print(f"  {name:<12} 无限位（可无限转动）")
    print()
    print("  限位是软约束，靠 solimp/solref 参数决定「多硬」。")
    print("  真实机器人一定要设限位，否则关节会转到物理上不可能的角度。")

    print()
    print("=" * 66)
    print("实战：Go2 的自由度账本")
    print("=" * 66)
    import pathlib
    p = pathlib.Path(__file__).parent.parent / "mujoco_menagerie" / "unitree_go2" / "scene.xml"
    if p.exists():
        go2 = mujoco.MjModel.from_xml_path(str(p))
        print(f"  nq={go2.nq}  nv={go2.nv}  nu={go2.nu}")
        print(f"  = freejoint(7/6) + 12 个腿关节(12/12)")
        print(f"  写控制器时：")
        print(f"    关节角  = qpos[7:]   长度 {go2.nq - 7}")
        print(f"    关节速度 = qvel[6:]   长度 {go2.nv - 6}")
        print(f"    躯干位置 = qpos[0:3]   躯干姿态 = qpos[3:7]")
        print(f"    躯干线速度 = qvel[0:3] 躯干角速度 = qvel[3:6]")
    else:
        print("  (menagerie 未找到，跳过)")

    print()
    print("小结：")
    print("  hinge/slide 各占 1 维；ball 占 4/3；free 占 7/6")
    print("  有 ball/free 时 nq > nv，索引千万别混")
    print("  查地址用 jnt_qposadr / jnt_dofadr，别手算")


if __name__ == "__main__":
    main()
