"""第 5 课：传感器 —— 机器人怎么"感知"

真实机器人只能靠传感器读数做决策，不能直接读仿真的真值。
写 RL 或控制器时，**观测量必须来自 sensordata**，
否则策略在仿真里能跑、上真机就废（因为真机没有上帝视角）。

所有传感器读数扁平地存在一个数组里：`data.sensordata`
用 `sensor_adr[i]` 和 `sensor_dim[i]` 查每个传感器占哪一段。

运行: python 05_sensors.py
"""

import mujoco
import numpy as np

XML = """
<mujoco model="sensors">
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" size="5 5 0.1"/>

    <body name="pendulum" pos="0 0 1.5">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.05"/>
      <geom name="rod" type="capsule" fromto="0 0 0 0 0 -0.6" size="0.03" mass="0.5"/>
      <body name="bob" pos="0 0 -0.6">
        <geom name="ball" type="sphere" size="0.08" mass="1"/>
        <site name="tip" size="0.02" rgba="1 0 0 1"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="torque" joint="hinge" ctrlrange="-5 5"/>
  </actuator>

  <sensor>
    <!-- 关节层 -->
    <jointpos    name="q"        joint="hinge"/>
    <jointvel    name="dq"       joint="hinge"/>
    <actuatorfrc name="tau"      actuator="torque"/>
    <!-- 站点层：位置/速度/加速度 -->
    <framepos    name="tip_pos"  objtype="site" objname="tip"/>
    <framelinvel name="tip_vel"  objtype="site" objname="tip"/>
    <!-- IMU：真实机器人上最常用的两个 -->
    <accelerometer name="imu_acc"  site="tip"/>
    <gyro          name="imu_gyro" site="tip"/>
    <framequat     name="tip_quat" objtype="site" objname="tip"/>
  </sensor>
</mujoco>
"""

SENSOR_NOTE = {
    "q": "关节角 (rad) —— 编码器",
    "dq": "关节角速度 (rad/s)",
    "tau": "执行器实际出力 (N·m)",
    "tip_pos": "末端世界坐标 (m) —— 真机上没有，动捕才有",
    "tip_vel": "末端线速度 (m/s)",
    "imu_acc": "加速度计 (m/s²) —— 含重力分量",
    "imu_gyro": "陀螺仪 (rad/s)",
    "tip_quat": "姿态四元数",
}


def main() -> None:
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)

    print("=" * 72)
    print("传感器清单 —— sensordata 的内存布局")
    print("=" * 72)
    print(f"  {'名称':<12}{'起始下标':>9}{'维度':>6}   说明")
    print("  " + "-" * 66)
    for i in range(m.nsensor):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i)
        adr, dim = m.sensor_adr[i], m.sensor_dim[i]
        print(f"  {name:<12}{adr:>9}{dim:>6}   {SENSOR_NOTE.get(name,'')}")
    print("  " + "-" * 66)
    print(f"  sensordata 总长度 = {m.nsensordata}")

    print()
    print("  取某个传感器的正确姿势（别手算偏移）：")
    print("    i   = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, 'imu_gyro')")
    print("    adr = m.sensor_adr[i]; dim = m.sensor_dim[i]")
    print("    val = d.sensordata[adr : adr+dim]")

    def read(name):
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
        a, n = m.sensor_adr[i], m.sensor_dim[i]
        return d.sensordata[a:a + n]

    print()
    print("=" * 72)
    print("摆动过程中的读数变化")
    print("=" * 72)
    d.qpos[0] = 1.0                       # 抬起 1 rad 放手
    mujoco.mj_forward(m, d)
    print(f"  {'时间':>6}{'关节角':>9}{'角速度':>9}{'末端高度':>10}"
          f"{'加速度模':>10}{'陀螺 y':>9}")
    print("  " + "-" * 55)
    for step in range(1501):
        if step % 250 == 0:
            print(f"  {d.time:>6.2f}{read('q')[0]:>9.3f}{read('dq')[0]:>9.3f}"
                  f"{read('tip_pos')[2]:>10.3f}"
                  f"{np.linalg.norm(read('imu_acc')):>10.3f}"
                  f"{read('imu_gyro')[1]:>9.3f}")
        mujoco.mj_step(m, d)

    print()
    print("  加速度计的反直觉之处：静止时读数不是 0。实测一下——")
    d2 = mujoco.MjData(m)
    d2.qpos[0] = 0.0                  # 垂直下垂，就是静止平衡位置
    mujoco.mj_forward(m, d2)
    for _ in range(2000):             # 让它彻底静下来
        mujoco.mj_step(m, d2)
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_acc")
    a = d2.sensordata[m.sensor_adr[i]:m.sensor_adr[i] + 3]
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "dq")
    print(f"    完全静止时(角速度={d2.sensordata[m.sensor_adr[j]]:.6f})")
    print(f"    加速度计读数 = {np.round(a, 3)}   模长 = {np.linalg.norm(a):.4f}")
    print(f"    正好等于重力加速度 9.81，方向朝上。")
    print()
    print("  ——它测的是「比力」(specific force)，不是运动加速度。")
    print("    静止时支撑力向上抵消重力，读数就是 +9.81。")
    print("    自由落体时反而读 0。真机 IMU 完全一样，这不是 bug。")

    print()
    print("=" * 72)
    print("为什么不能直接读 data.qpos 当观测")
    print("=" * 72)
    print("  仿真里 d.qpos / d.qvel 是「上帝视角真值」，真机上拿不到：")
    print("    - 躯干在世界系的绝对位置：真机只有 IMU 积分，会漂移")
    print("    - 关节速度：真机是编码器差分，有噪声和延迟")
    print()
    print("  正确做法：观测只用真机也有的量")
    print("    关节角/角速度 (编码器)、IMU 加速度+角速度、足底力传感器")
    print("  想更真实，还可以给传感器加噪声：")
    print('    <jointpos name="q" joint="hinge" noise="0.01"/>')

    print()
    print("=" * 72)
    print("实战：Go2 自带哪些传感器")
    print("=" * 72)
    import pathlib
    p = pathlib.Path(__file__).parent.parent / "mujoco_menagerie" / "unitree_go2" / "scene.xml"
    if p.exists():
        go2 = mujoco.MjModel.from_xml_path(str(p))
        if go2.nsensor == 0:
            print("  Go2 的 scene.xml 没定义传感器 —— 需要自己加。")
            print("  真实四足 RL 的典型观测组合：")
            print("    躯干姿态(framequat) + 角速度(gyro) + 重力方向投影")
            print("    + 12 关节角 + 12 关节速度 + 上一步动作")
        else:
            names = [mujoco.mj_id2name(go2, mujoco.mjtObj.mjOBJ_SENSOR, i)
                     for i in range(go2.nsensor)]
            print(f"  {go2.nsensor} 个传感器: {', '.join(names[:12])}")
            print(f"  sensordata 长度 = {go2.nsensordata}")

    print()
    print("小结：")
    print("  sensordata 是扁平数组，用 sensor_adr/sensor_dim 定位")
    print("  加速度计静止读 9.81 不是 bug")
    print("  观测只用真机拿得到的量，否则 sim2real 必翻车")


if __name__ == "__main__":
    main()
