"""第 2 课：MJCF 建模 —— 怎么描述一个机器人

MJCF 是 XML，核心是一棵**运动学树**：

    <worldbody>              世界，树根，固定不动
      └── <body>             刚体
            ├── <joint>      这个刚体相对父级怎么动
            ├── <geom>       形状（同时是外观和碰撞体）
            ├── <site>       标记点（无碰撞，用于传感器/参考系）
            └── <body>       子刚体，可无限嵌套

关键认知：**关节属于子刚体，描述的是它相对父刚体的运动**。
没有 joint 的 body 就是焊死在父级上。

运行: python 02_mjcf.py
"""

import mujoco
import numpy as np

# ---------------------------------------------------------------- 几何体类型
GEOM_DEMO = """
<mujoco model="geom_types">
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="5 5 0.1"/>
    <body pos="-1.5 0 0.5"><freejoint/>
      <geom type="sphere" size="0.2" rgba="0.9 0.2 0.2 1"/></body>
    <body pos="-0.5 0 0.5"><freejoint/>
      <geom type="box" size="0.2 0.15 0.1" rgba="0.2 0.9 0.2 1"/></body>
    <body pos="0.5 0 0.5"><freejoint/>
      <geom type="capsule" size="0.1" fromto="0 -0.2 0 0 0.2 0" rgba="0.2 0.2 0.9 1"/></body>
    <body pos="1.5 0 0.5"><freejoint/>
      <geom type="cylinder" size="0.15 0.2" rgba="0.9 0.9 0.2 1"/></body>
    <body pos="2.5 0 0.5"><freejoint/>
      <geom type="ellipsoid" size="0.2 0.15 0.1" rgba="0.9 0.2 0.9 1"/></body>
  </worldbody>
</mujoco>
"""

# ---------------------------------------------------------------- 运动学树
# 一条三节机械臂。注意 body 的嵌套 —— 每一节挂在前一节下面。
ARM = """
<mujoco model="arm">
  <compiler angle="degree"/>
  <default>
    <joint damping="0.5"/>
    <geom rgba="0.6 0.6 0.9 1"/>
  </default>

  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="3 3 0.1" rgba="0.8 0.8 0.8 1"/>

    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.12 0.1"/>

      <body name="link1" pos="0 0 0.1">
        <joint name="j1" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.4" size="0.05"/>

        <body name="link2" pos="0 0 0.4">
          <joint name="j2" type="hinge" axis="0 1 0" range="-90 90"/>
          <geom type="capsule" fromto="0 0 0 0.35 0 0" size="0.04"/>

          <body name="link3" pos="0.35 0 0">
            <joint name="j3" type="hinge" axis="0 1 0" range="-120 120"/>
            <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.03"/>
            <site name="tip" pos="0.25 0 0" size="0.02" rgba="1 0 0 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def show_geoms() -> None:
    print("=" * 62)
    print("几何体类型 —— geom 既是外观也是碰撞体")
    print("=" * 62)
    m = mujoco.MjModel.from_xml_string(GEOM_DEMO)
    types = {
        mujoco.mjtGeom.mjGEOM_PLANE: "plane    平面（无限大，只能固定）",
        mujoco.mjtGeom.mjGEOM_SPHERE: "sphere   球     size=[半径]",
        mujoco.mjtGeom.mjGEOM_BOX: "box      长方体 size=[半长,半宽,半高]",
        mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule  胶囊   用 fromto 定两端 + size=[半径]",
        mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder 圆柱   size=[半径,半高]",
        mujoco.mjtGeom.mjGEOM_ELLIPSOID: "ellipsoid 椭球  size=[a,b,c]",
    }
    for i in range(m.ngeom):
        t = m.geom_type[i]
        print(f"  {types.get(t, str(t))}")
    print()
    print("  实践建议：capsule 的碰撞检测最快最稳，机器人连杆首选。")
    print("  box 的角点接触容易抖，mesh 最贵。")


def show_tree() -> None:
    print()
    print("=" * 62)
    print("运动学树 —— body 嵌套决定了谁带着谁动")
    print("=" * 62)
    m = mujoco.MjModel.from_xml_string(ARM)
    data = mujoco.MjData(m)

    # 打印树结构
    def name_of(obj, i):
        return mujoco.mj_id2name(m, obj, i) or f"<{i}>"

    print("  结构：")
    for b in range(m.nbody):
        depth = 0
        p = m.body_parentid[b]
        while p != 0 and b != 0:
            depth += 1
            if p == m.body_parentid[p]:
                break
            p = m.body_parentid[p]
        indent = "    " + "  " * depth
        jnts = [name_of(mujoco.mjtObj.mjOBJ_JOINT, j)
                for j in range(m.njnt) if m.jnt_bodyid[j] == b]
        j_str = f"  [关节: {', '.join(jnts)}]" if jnts else ""
        print(f"{indent}{name_of(mujoco.mjtObj.mjOBJ_BODY, b)}{j_str}")

    print()
    print("  这条臂有 3 个 hinge 关节，所以 nq=nv=3（hinge 每个占 1 维）")
    print(f"  实际: nq={m.nq}  nv={m.nv}")

    print()
    print("  转动 j1 会带着 link2、link3 一起转 —— 父级运动传递给所有子级：")
    print(f"  {'j1 角度':>10} {'末端 tip 位置 (x, y, z)':>34}")
    print("  " + "-" * 46)
    tip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tip")
    for deg in (0, 30, 60, 90):
        data.qpos[:] = [np.deg2rad(deg), 0, 0]
        # mj_forward 只算运动学/动力学量，不推进时间。想"看当前状态"就用它。
        mujoco.mj_forward(m, data)
        pos = data.site_xpos[tip_id]
        print(f"  {deg:>8}°   ({pos[0]:>7.3f}, {pos[1]:>7.3f}, {pos[2]:>7.3f})")

    print()
    print("  mj_forward vs mj_step：")
    print("    mj_forward  只更新派生量（位置/雅可比/接触），不推进时间")
    print("    mj_step     = mj_forward + 积分一步，时间前进")
    print("    改了 qpos 想立刻看效果 -> mj_forward")


def show_defaults() -> None:
    print()
    print("=" * 62)
    print("<default> —— 避免重复，机器人模型必用")
    print("=" * 62)
    m = mujoco.MjModel.from_xml_string(ARM)
    print("  上面 XML 里写了：")
    print("    <default><joint damping=\"0.5\"/><geom rgba=\"...\"/></default>")
    print("  于是所有 joint 自动带 damping=0.5，不用逐个写。")
    print(f"  实际读到的阻尼: {m.dof_damping}")
    print()
    print("  真实机器人模型（menagerie）大量用 <default class=\"...\">")
    print("  给不同部位分组，比如 Go2 的 abduction/hip/knee 三类关节。")


if __name__ == "__main__":
    show_geoms()
    show_tree()
    show_defaults()
    print()
    print("小结：")
    print("  worldbody -> body -> body 嵌套 = 运动学树")
    print("  joint 挂在子 body 上，描述它相对父 body 怎么动")
    print("  geom 同时负责外观和碰撞；site 是无碰撞的标记点")
