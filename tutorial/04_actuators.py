"""第 4 课：执行器与控制 —— ctrl 到底是什么单位

**这一课是最容易出错的地方**，我自己搭环境时就写错过。

同一个数字写进 `data.ctrl`，含义完全取决于执行器类型：

    <motor>     ctrl = 力矩 (N·m) 或力 (N)      需要你自己做闭环
    <position>  ctrl = 目标角度 (rad)            伺服内部自带 PD
    <velocity>  ctrl = 目标角速度 (rad/s)        伺服内部自带 P

往 position 伺服里写「力矩」，或往 motor 里写「角度」，
不会报错，只会得到**完全错误的行为**。

怎么在代码里判断？看 biastype：
    biastype == mjBIAS_NONE   -> motor（纯力矩）
    biastype == mjBIAS_AFFINE -> position/velocity（内置反馈）

运行: python 04_actuators.py
"""

import mujoco
import numpy as np

# 同一个单摆，只有执行器类型不同
BASE = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3"/>
    <body name="arm" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.04" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    {actuator}
  </actuator>
</mujoco>
"""

MODELS = {
    "motor":    '<motor    name="a" joint="hinge" ctrlrange="-20 20"/>',
    "position": '<position name="a" joint="hinge" kp="50" kv="5" ctrlrange="-3.14 3.14"/>',
    "velocity": '<velocity name="a" joint="hinge" kv="10" ctrlrange="-5 5"/>',
}


def kind(m: mujoco.MjModel) -> str:
    if m.nu == 0:
        return "无执行器"
    bt = m.actuator_biastype[0]
    if bt == mujoco.mjtBias.mjBIAS_NONE:
        return "motor 力矩"
    gp = m.actuator_gainprm[0]
    bp = m.actuator_biasprm[0]
    # position: gainprm[0]=kp, biasprm[1]=-kp ; velocity: biasprm[2]=-kv, biasprm[1]=0
    return "position 位置" if bp[1] != 0 else "velocity 速度"


def run(name: str, ctrl_value: float, seconds: float = 2.0):
    m = mujoco.MjModel.from_xml_string(BASE.format(actuator=MODELS[name]))
    d = mujoco.MjData(m)
    for _ in range(int(seconds / m.opt.timestep)):
        d.ctrl[0] = ctrl_value
        mujoco.mj_step(m, d)
    return m, d


def main() -> None:
    print("=" * 70)
    print("同一个数字 ctrl=1.0，三种执行器给出三种完全不同的结果")
    print("=" * 70)
    print(f"  {'执行器':<16}{'内部类型':<16}{'ctrl 含义':<18}{'2 秒后关节角':>12}")
    print("  " + "-" * 64)
    meanings = {"motor": "力矩 1.0 N·m", "position": "目标角 1.0 rad",
                "velocity": "目标角速度 1.0"}
    for name in MODELS:
        m, d = run(name, 1.0)
        print(f"  <{name}>{'':<{15-len(name)-2}}{kind(m):<16}{meanings[name]:<18}"
              f"{d.qpos[0]:>12.4f}")

    print()
    print("  position 停在 1.0 rad —— 因为那就是目标角度，伺服帮你闭环了")
    print("  motor 停在别处   —— 1.0 N·m 只是一个恒定力矩，跟重力/阻尼平衡")
    print("  velocity 一直在转 —— 目标是角速度不是角度")

    print()
    print("=" * 70)
    print("内部机制：MuJoCo 统一的执行器公式")
    print("=" * 70)
    print("    力 = gain * ctrl + bias·(1, 关节角, 关节速度)")
    print()
    print(f"  {'执行器':<12}{'gainprm[0]':>12}{'biasprm[1]':>12}{'biasprm[2]':>12}  说明")
    print("  " + "-" * 66)
    for name in MODELS:
        m = mujoco.MjModel.from_xml_string(BASE.format(actuator=MODELS[name]))
        g = m.actuator_gainprm[0][0]
        b1, b2 = m.actuator_biasprm[0][1], m.actuator_biasprm[0][2]
        note = {"motor": "无反馈，力=ctrl", "position": "b1=-kp b2=-kv 即 PD",
                "velocity": "只有 b2=-kv 即 P"}[name]
        print(f"  <{name}>{'':<{11-len(name)-2}}{g:>12.1f}{b1:>12.1f}{b2:>12.1f}  {note}")
    print()
    print("  position 展开就是:  力 = kp*ctrl - kp*q - kv*dq = kp*(ctrl-q) - kv*dq")
    print("  ——这正是 PD 控制器。所以位置伺服 = 内置了 PD 的 motor。")

    print()
    print("=" * 70)
    print("写错会怎样：往 position 伺服里写「力矩」")
    print("=" * 70)
    m, d = run("position", 0.0, seconds=2.0)
    print(f"  以为在写「0 力矩，让它自由摆动」-> 实际是「目标角度 0」")
    print(f"  结果关节被死死拉到 0：qpos={d.qpos[0]:.6f}")
    m2, d2 = run("motor", 0.0, seconds=2.0)
    print(f"  真正的「0 力矩」应该用 motor：qpos={d2.qpos[0]:.4f}（自由下垂）")

    print()
    print("=" * 70)
    print("motor 要保持姿势，必须自己写 PD")
    print("=" * 70)
    m = mujoco.MjModel.from_xml_string(BASE.format(actuator=MODELS["motor"]))
    target = 0.8
    print(f"  目标角度 {target} rad，用 tau = kp*(target-q) - kd*dq")
    print(f"  {'kp':>6}{'kd':>6}{'2秒后角度':>12}{'误差':>10}")
    print("  " + "-" * 34)
    for kp, kd in [(5, 0.5), (20, 1), (50, 2), (100, 5)]:
        d = mujoco.MjData(m)
        lo, hi = m.actuator_ctrlrange[0]
        for _ in range(1000):
            tau = kp * (target - d.qpos[0]) - kd * d.qvel[0]
            d.ctrl[0] = np.clip(tau, lo, hi)
            mujoco.mj_step(m, d)
        print(f"  {kp:>6}{kd:>6}{d.qpos[0]:>12.4f}{abs(d.qpos[0]-target):>10.4f}")
    print()
    print("  kp 越大跟踪越准，但太大会震荡/发散。kd 提供阻尼抑制震荡。")

    print()
    print("=" * 70)
    print("实战：menagerie 里两种都有，用前必须先判断")
    print("=" * 70)
    import pathlib
    root = pathlib.Path(__file__).parent.parent / "mujoco_menagerie"
    for name in ["unitree_go2", "unitree_a1", "boston_dynamics_spot",
                 "unitree_g1", "unitree_h1", "franka_emika_panda"]:
        p = root / name / "scene.xml"
        if not p.exists():
            continue
        mm = mujoco.MjModel.from_xml_path(str(p))
        kp = -mm.actuator_biasprm[0][1] if mm.nu else 0
        extra = f"  内置 kp={kp:.0f}" if kp else "  需自己写 PD"
        print(f"  {name:<24}{kind(mm):<14}{extra}")

    print()
    print("小结：")
    print("  ctrl 的单位由执行器类型决定，不看类型就写 = 埋雷")
    print("  判断方法: model.actuator_biastype[i] == mjBIAS_NONE 即 motor")
    print("  motor 想保持姿势 -> 自己写 PD；position 直接给目标角")


if __name__ == "__main__":
    main()
