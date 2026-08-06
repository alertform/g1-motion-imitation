"""第 1 课：MuJoCo 的核心二分 —— MjModel 与 MjData

这是理解 MuJoCo 的地基。所有 API 都围绕这两个对象转。

    MjModel  = 不变的东西。质量、几何、关节结构、执行器参数。
               从 XML 编译而来，仿真过程中只读。
    MjData   = 会变的东西。位置、速度、力、接触、时间。
               每次 mj_step 都会被改写。

一个 MjModel 可以配多个 MjData —— 这就是并行环境采样的基础：
模型只编译一次、共享，每个环境一份 MjData。

运行: python 01_hello.py
"""

import mujoco
import numpy as np

# ---------------------------------------------------------------- 模型定义
# MJCF 是 MuJoCo 的模型格式。最小可用模型需要：
#   <worldbody>  场景根节点，所有物体挂在它下面
#   <geom>       几何体，既是外观也是碰撞体
#   <body>       刚体，可以嵌套，形成运动学树
#   <freejoint>  自由关节，让刚体能在空间中自由运动（6 自由度）
XML = """
<mujoco model="hello">
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" size="5 5 0.1"/>

    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="0.1" rgba="0.9 0.3 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def main() -> None:
    # 编译模型。这一步把 XML 变成高效的数值结构，只做一次。
    model = mujoco.MjModel.from_xml_string(XML)
    # 分配状态容器。
    data = mujoco.MjData(model)

    print("=" * 60)
    print("MjModel —— 静态结构（仿真中不变）")
    print("=" * 60)
    print(f"  nbody  刚体数     = {model.nbody}   (含 world 本身)")
    print(f"  ngeom  几何体数   = {model.ngeom}")
    print(f"  njnt   关节数     = {model.njnt}")
    print(f"  nq     位置维度   = {model.nq}   <- freejoint 占 7")
    print(f"  nv     速度维度   = {model.nv}   <- freejoint 占 6")
    print(f"  nu     执行器数   = {model.nu}   (这个模型没有电机)")
    print(f"  timestep          = {model.opt.timestep} 秒")
    print(f"  gravity           = {model.opt.gravity}")

    print()
    print("  为什么 nq=7 而 nv=6？")
    print("    位置用「3 平移 + 4 四元数」表示 = 7")
    print("    速度用「3 线速度 + 3 角速度」表示 = 6")
    print("    四元数有 1 个约束(模长=1)，所以位置比速度多 1 维。")
    print("    这是新手最常踩的坑：qpos 和 qvel 的长度不一样，不能直接对齐。")

    print()
    print("=" * 60)
    print("MjData —— 动态状态（每步都变）")
    print("=" * 60)
    print(f"  初始 qpos = {data.qpos}")
    print(f"       前 3 个是位置 xyz，后 4 个是姿态四元数 wxyz")
    print(f"  初始 qvel = {data.qvel}")
    print(f"  初始 time = {data.time}")

    print()
    print("=" * 60)
    print("仿真循环 —— mj_step 推进一个时间步")
    print("=" * 60)
    print(f"  {'步数':>6} {'时间(s)':>9} {'高度 z':>9} {'垂直速度':>10}")
    print("  " + "-" * 38)

    for i in range(501):
        if i % 100 == 0:
            print(f"  {i:>6} {data.time:>9.3f} {data.qpos[2]:>9.4f} {data.qvel[2]:>10.4f}")
        mujoco.mj_step(model, data)

    print()
    print("  小球从 z=1.0 自由落体，撞到地面后停在 z≈0.1（球半径）。")
    print("  注意速度在触地瞬间反号——那是接触力在起作用。")

    print()
    print("=" * 60)
    print("一个模型配多个 data —— 并行环境的基础")
    print("=" * 60)
    envs = [mujoco.MjData(model) for _ in range(3)]
    for i, d in enumerate(envs):
        d.qpos[2] = 1.0 + i * 0.5          # 三个不同的初始高度
    for _ in range(200):
        for d in envs:
            mujoco.mj_step(model, d)
    for i, d in enumerate(envs):
        print(f"  环境 {i}: 初始高度 {1.0 + i*0.5:.1f}  ->  0.4 秒后 z={d.qpos[2]:.4f}")
    print()
    print("  模型只编译一次、三份状态独立演化。RL 并行采样就是这么做的。")

    print()
    print("小结：")
    print("  model = 蓝图（只读）   data = 当前状态（可写）")
    print("  mj_step(model, data) 把 data 往前推一个 timestep")


if __name__ == "__main__":
    main()
