"""第 7 课：接触与摩擦 —— 腿足机器人的一切都在这里

机器人能走路，靠的是脚和地面的接触力。理解接触是理解腿足仿真的前提。

三件事要搞明白：
    1. 谁和谁会碰？        —— contype/conaffinity 位掩码
    2. 碰上了力有多大？    —— solref/solimp 决定"软硬"
    3. 会不会打滑？        —— friction

运行: python 07_contacts.py
"""

import mujoco
import numpy as np

XML = """
<mujoco model="contacts">
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" size="5 5 0.1" friction="1.0 0.005 0.0001"/>

    <body name="box_a" pos="-1 0 0.3">
      <freejoint/>
      <geom name="ga" type="box" size="0.1 0.1 0.1" rgba="0.9 0.3 0.2 1" mass="1"/>
    </body>
    <body name="box_b" pos="0 0 0.3">
      <freejoint/>
      <geom name="gb" type="box" size="0.1 0.1 0.1" rgba="0.3 0.9 0.2 1" mass="1"/>
    </body>
    <body name="box_c" pos="1 0 0.3">
      <freejoint/>
      <geom name="gc" type="box" size="0.1 0.1 0.1" rgba="0.2 0.3 0.9 1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

# 斜坡上放不同摩擦的方块，看谁滑下去
SLOPE = """
<mujoco model="slope">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <compiler angle="degree"/>
  <worldbody>
    <light pos="0 0 5"/>
    <!-- 斜坡绕 y 轴倾斜 20 度 -->
    <geom name="ramp" type="box" size="3 2 0.05" pos="0 0 0" euler="0 -20 0"
          friction="{mu} 0.005 0.0001" rgba="0.7 0.7 0.7 1"/>
    <!-- 方块也转 20 度，并贴着斜面放，否则会先掉下来再弹几下，测到的就不是纯滑移 -->
    <body name="block" pos="0 0 0.16" euler="0 -20 0">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" friction="{mu} 0.005 0.0001"
            rgba="0.9 0.4 0.2 1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def demo_contact_data():
    print("=" * 70)
    print("1. 接触信息藏在 data.contact 里")
    print("=" * 70)
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    for _ in range(500):
        mujoco.mj_step(m, d)

    print(f"  当前接触点数 data.ncon = {d.ncon}")
    print()
    print(f"  {'#':>3}{'geom1':>8}{'geom2':>8}{'穿透深度':>12}{'接触点位置':>28}")
    print("  " + "-" * 60)
    for i in range(d.ncon):
        c = d.contact[i]
        g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or c.geom1
        g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or c.geom2
        p = c.pos
        print(f"  {i:>3}{str(g1):>8}{str(g2):>8}{c.dist:>12.6f}"
              f"   ({p[0]:>6.2f},{p[1]:>6.2f},{p[2]:>6.2f})")
    print()
    print("  dist < 0 表示穿透 —— MuJoCo 允许微小穿透，靠约束求解器推开")
    print("  一个方块着地通常有 4 个接触点（4 个角）")


def demo_contact_force():
    print()
    print("=" * 70)
    print("2. 读接触力 —— 足底力传感器的实现方式")
    print("=" * 70)
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    for _ in range(800):
        mujoco.mj_step(m, d)

    total_normal = 0.0
    print(f"  {'#':>3}{'法向力(N)':>12}{'切向力1':>11}{'切向力2':>11}")
    print("  " + "-" * 38)
    force = np.zeros(6)
    for i in range(d.ncon):
        mujoco.mj_contactForce(m, d, i, force)
        # force = [法向, 切向1, 切向2, 力矩x, y, z]，在接触坐标系里
        total_normal += force[0]
        if i < 6:
            print(f"  {i:>3}{force[0]:>12.3f}{force[1]:>11.3f}{force[2]:>11.3f}")
    print("  " + "-" * 38)
    print(f"  法向力合计 = {total_normal:.2f} N")
    print(f"  理论值 3 个 1kg 方块 = {3 * 1 * 9.81:.2f} N")
    print()
    print("  对得上说明求解器收敛良好。差太多说明步长太大或约束太软。")


def demo_friction():
    print()
    print("=" * 70)
    print("3. 摩擦 —— 20° 斜坡上，多大摩擦系数才不滑")
    print("=" * 70)
    mu_crit = np.tan(np.deg2rad(20))
    print(f"  理论临界值 μ = tan(20°) = {mu_crit:.3f}")
    print("  μ > 临界值 -> 静摩擦足以平衡重力分量 -> 不滑")
    print()
    print("  测法：先跑 0.5 秒让方块落定，再测之后 3 秒的位移，")
    print("       这样测到的才是稳态滑移，不含初始的沉降和弹跳。")
    print()
    print(f"  {'摩擦系数 μ':>12}{'稳态滑移(m)':>16}{'预期':>8}   实测")
    print("  " + "-" * 50)
    for mu in [0.05, 0.2, 0.3, 0.34, 0.4, 0.5, 1.0]:
        m = mujoco.MjModel.from_xml_string(SLOPE.format(mu=mu))
        d = mujoco.MjData(m)
        for _ in range(250):              # 0.5 秒沉降
            mujoco.mj_step(m, d)
        p0 = d.qpos[:3].copy()
        for _ in range(1500):             # 3 秒测量
            mujoco.mj_step(m, d)
        slip = float(np.linalg.norm(d.qpos[:3] - p0))
        expect = "滑" if mu < mu_crit else "不滑"
        actual = "滑" if slip > 0.02 else "不滑"
        flag = "" if expect == actual else "   <- 与理论不符"
        print(f"  {mu:>12.2f}{slip:>16.4f}{expect:>8}   {actual}{flag}")
    print()
    print("  分界点落在 0.34~0.4 之间，与理论值 0.364 吻合。")
    print("  friction 三个数是 [滑动, 自旋, 滚动]，通常只调第一个。")
    print("  两个 geom 接触时，实际摩擦按 solmix 规则混合（默认取几何平均）。")


def demo_collision_filter():
    print()
    print("=" * 70)
    print("4. 碰撞过滤 —— 为什么机器人自己的腿不会卡住自己")
    print("=" * 70)
    print("  MuJoCo 判断两个 geom 是否检测碰撞：")
    print("    (contype1 & conaffinity2) || (contype2 & conaffinity1)  非零才检测")
    print()
    print("  常用套路：")
    print("    contype=1 conaffinity=1   默认，什么都碰")
    print("    contype=0 conaffinity=0   纯视觉，不参与碰撞（省算力）")
    print("    contype=1 conaffinity=0   只被别人碰，自己不主动碰")
    print()
    print("  另外父子 body 之间默认不检测碰撞（相邻连杆本来就挨着）。")
    print("  想强制排除某两个 geom：")
    print('    <contact><exclude body1="thigh" body2="calf"/></contact>')

    import pathlib
    p = pathlib.Path(__file__).parent.parent / "mujoco_menagerie" / "unitree_go2" / "scene.xml"
    if p.exists():
        go2 = mujoco.MjModel.from_xml_path(str(p))
        vis_only = int(np.sum((go2.geom_contype == 0) & (go2.geom_conaffinity == 0)))
        print()
        print(f"  Go2 实例：{go2.ngeom} 个 geom，其中 {vis_only} 个是纯视觉不参与碰撞")
        print(f"    —— 外壳网格只用来好看，碰撞用简化的 capsule/sphere，快得多")


def demo_solver():
    print()
    print("=" * 70)
    print("5. 接触「软硬」—— solref / solimp")
    print("=" * 70)
    print("  solref = [时间常数, 阻尼比]，默认 [0.02, 1]")
    print("    时间常数越小 -> 接触越硬 -> 穿透越少，但越容易数值不稳")
    print()
    tpl = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="5 5 .1" solref="{sr} 1"/>
    <body pos="0 0 0.5"><freejoint/>
      <geom type="sphere" size="0.1" mass="10" solref="{sr} 1"/></body>
  </worldbody>
</mujoco>"""
    print(f"  {'solref[0]':>11}{'稳定后穿透深度(m)':>20}   相对默认(0.02)")
    print("  " + "-" * 48)
    for sr in [0.002, 0.01, 0.02, 0.1]:
        m = mujoco.MjModel.from_xml_string(tpl.format(sr=sr))
        d = mujoco.MjData(m)
        for _ in range(1500):
            mujoco.mj_step(m, d)
        pen = 0.1 - d.qpos[2]     # 球半径 - 球心高度
        note = "默认" if sr == 0.02 else ("更硬" if sr < 0.02 else "更软")
        print(f"  {sr:>11.3f}{pen:>20.6f}   {note}")
    print()
    print("  腿足机器人一般保持默认。太硬会让求解器发散，")
    print("  症状是机器人突然弹飞或 qpos 出现 NaN。")


def main() -> None:
    demo_contact_data()
    demo_contact_force()
    demo_friction()
    demo_collision_filter()
    demo_solver()
    print()
    print("小结：")
    print("  data.ncon / data.contact[i] 读接触，mj_contactForce 读力")
    print("  friction[0] 是滑动摩擦，斜坡不滑的条件 μ > tan(坡角)")
    print("  碰撞过滤靠 contype/conaffinity 位掩码，视觉网格设 0 省算力")
    print("  接触发散/弹飞 -> 先查 solref 是不是调太硬、timestep 是不是太大")


if __name__ == "__main__":
    main()
