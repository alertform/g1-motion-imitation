"""第 10 课：用真实动捕数据驱动 G1

数据来源: openhe/g1-retargeted-motions (MIT)
          AMASS/ACCAD 人体动捕 -> 重定向到 Unitree G1

这一课把第 9 课的"方式 2（qpos 回放）"用在真实数据上，
过程中会撞到动捕流程里最常见的三个坑：

    1. 四元数顺序    数据用 xyzw（SMPL/scipy 惯例），MuJoCo 用 wxyz
    2. 自由度不匹配  数据是 23-DOF 的 G1，menagerie 的模型是 29-DOF
    3. 根位置偏移    动捕的世界原点不一定在地面

运行:
    python 10_real_mocap.py                    # 回放并出 GIF
    python 10_real_mocap.py --view             # 开交互窗口
    python 10_real_mocap.py --list             # 列出已下载的动作
"""

import os
import pathlib
import pickle
import sys

import imageio.v3 as iio
import mujoco
import mujoco.viewer
import numpy as np

ROOT = pathlib.Path(__file__).parent.parent
MOTIONS = ROOT / "motions"
MODEL_PATH = ROOT / "mujoco_menagerie" / "unitree_g1" / "scene.xml"
OUT = pathlib.Path(__file__).parent / "out"

# 23-DOF 数据 -> 29-DOF 模型的下标映射。
#
# 数据布局（由标准差分析 + 数据集卡片确认）：
#   [0:12]  双腿 6+6
#   [12:15] 腰 yaw/roll/pitch     <- 曾误以为只有 yaw 一个
#   [15:19] 左臂 shoulder p/r/y + elbow   （只有 4 个，没有腕）
#   [19:23] 右臂 shoulder p/r/y + elbow
#
# 模型（29-DOF）比数据多出 6 个腕关节，补 0。
DOF_MAP = {
    **{i: i for i in range(19)},          # 腿 12 + 腰 3 + 左臂 4，下标恰好一致
    **{19 + k: 22 + k for k in range(4)},  # 右臂：数据 19..22 -> 模型 22..25
}
MISSING = [19, 20, 21, 26, 27, 28]         # 六个腕关节，数据里没有


def load_motion(path: pathlib.Path) -> dict:
    """读取重定向动作，自动识别 pickle / joblib 两种格式。

    这个数据集里两种格式混着：ACCAD 子集是裸 pickle，
    LAFAN1 子集是 joblib（带压缩头，pickle.load 会报
    "invalid load key"）。先试 joblib，它对裸 pickle 也兼容。

    注意：两种格式都会执行任意代码，只加载可信来源的文件。
    """
    try:
        import joblib
        outer = joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            outer = pickle.load(f)
    # 外层通常是单键字典，键是原始 npz 路径；也可能已经是数据本身
    if isinstance(outer, dict) and "root_trans_offset" not in outer:
        return outer[next(iter(outer))]
    return outer


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """SMPL/scipy 用 (x,y,z,w)，MuJoCo 用 (w,x,y,z)。

    顺序搞错不会报错——机器人只是姿态诡异（躺着、歪着），
    很容易被误判成「重定向没做好」。这是动捕流程第一大坑。
    """
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)


def build_qpos(motion: dict, model: mujoco.MjModel) -> np.ndarray:
    """把动作数据组装成 (T, nq) 的 qpos 轨迹。"""
    trans = np.asarray(motion["root_trans_offset"], dtype=np.float64)
    rot = np.asarray(motion["root_rot"], dtype=np.float64)
    dof = np.asarray(motion["dof"], dtype=np.float64)
    T = len(trans)

    qpos = np.zeros((T, model.nq))
    qpos[:, 0:3] = trans
    qpos[:, 3:7] = quat_xyzw_to_wxyz(rot)
    for src, dst in DOF_MAP.items():
        qpos[:, 7 + dst] = dof[:, src]
    return qpos


def ground_align(qpos: np.ndarray, model: mujoco.MjModel,
                 samples: int = 250, percentile: float = 50.0) -> np.ndarray:
    """把轨迹在竖直方向平移，使「典型触地时刻」脚正好贴地。

    只看足底接触球，不看别的 geom——手撑地、躯干贴地这些会误导。

    取「每帧最低脚高度」的**中位数**作为地面。依据：只要大部分时间有一只
    脚踩在地上，这个量的中位数就是地面高度。

    为什么不用最小值：重定向数据常有个别帧穿透很深（Capoeira 某帧脚陷到
    地下 20cm），按最小值对齐会把整段抬高 20cm，站立时脚明显悬空。
    为什么不用低分位：Capoeira 这类动作的低分位被深蹲/贴地帧占据，
    用 5% 分位对齐后起始站立姿态仍悬空 6.5cm（实测）。
    九段动作对比下来中位数最稳：起始帧偏差普遍在 ±1.3cm 内。

    返回的轨迹里仍可能有少数帧穿透，那是源数据本身的瑕疵，
    单一平移无法消除（要消除得做逐帧校正，但那会破坏跳跃的竖直动力学）。
    """
    feet = [g for g in range(model.ngeom)
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
            and (model.geom_contype[g] or model.geom_conaffinity[g])]
    if not feet:
        return qpos.copy()

    d = mujoco.MjData(model)
    step = max(1, len(qpos) // samples)
    bottoms = []
    for i in range(0, len(qpos), step):
        d.qpos[:] = qpos[i]
        mujoco.mj_forward(model, d)
        bottoms.append(min(float(d.geom_xpos[g][2]) - float(model.geom_size[g][0])
                           for g in feet))
    contact_level = float(np.percentile(bottoms, percentile))

    out = qpos.copy()
    out[:, 2] -= contact_level
    return out


def available() -> list[pathlib.Path]:
    return sorted(MOTIONS.glob("*.pkl"))


def main() -> None:
    args = sys.argv[1:]
    motions = available()
    if not motions:
        raise SystemExit(
            f"没有动作文件。先下载到 {MOTIONS}/\n"
            "  见 README 的「获取动捕数据」一节"
        )

    if "--list" in args:
        print(f"已下载 {len(motions)} 个动作:")
        for p in motions:
            m = load_motion(p)
            T = len(m["root_trans_offset"])
            fps = int(np.asarray(m["fps"]))
            print(f"  {p.stem:<32} {T:>5} 帧 @ {fps} fps = {T/fps:.1f} 秒")
        return

    path = motions[1] if len(motions) > 1 else motions[0]
    for a in args:
        if not a.startswith("-"):
            hits = [p for p in motions if a.lower() in p.stem.lower()]
            if hits:
                path = hits[0]

    print("=" * 72)
    print(f"1. 加载动作: {path.name}")
    print("=" * 72)
    motion = load_motion(path)
    fps = int(np.asarray(motion["fps"]))
    T = len(motion["root_trans_offset"])
    print(f"  帧数 {T} @ {fps} fps = {T/fps:.1f} 秒")
    print(f"  字段: {', '.join(motion.keys())}")
    print(f"  dof shape = {np.asarray(motion['dof']).shape}   <- 23 自由度")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    print(f"  模型 nq={model.nq} nv={model.nv} nu={model.nu}   <- 29 自由度")

    print()
    print("=" * 72)
    print("2. 坑一：四元数顺序")
    print("=" * 72)
    q_raw = np.asarray(motion["root_rot"])[0]
    q_mj = quat_xyzw_to_wxyz(q_raw)
    print(f"  数据 (xyzw) = {np.round(q_raw, 4)}")
    print(f"  MuJoCo(wxyz)= {np.round(q_mj, 4)}")
    print("  最大分量从末位挪到首位 —— 这是绕 z 的偏航角，人站着朝某个方向")
    print("  顺序写反不会报错，机器人只会躺着或歪着，极易误判成重定向失败")

    print()
    print("=" * 72)
    print("3. 坑二：自由度不匹配 23 -> 29")
    print("=" * 72)
    print(f"  数据 23 DOF (G1 基础版) vs 模型 29 DOF (G1 增强版)")
    print(f"  能对上的 {len(DOF_MAP)} 个，模型独有的 {len(MISSING)} 个补 0:")
    for j in MISSING:
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j + 1)
        print(f"    [{j:>2}] {n}")
    print()
    print("  教训：重定向数据绑定具体硬件配置，换机器人必须重新映射。")

    qpos = build_qpos(motion, model)
    print()
    print("=" * 72)
    print("4. 坑三：根位置对地")
    print("=" * 72)
    z_before = qpos[0, 2]
    qpos = ground_align(qpos, model)
    print(f"  校正前躯干高度 {z_before:.4f} m -> 校正后 {qpos[0, 2]:.4f} m"
          f"  (平移 {qpos[0,2]-z_before:+.4f} m)")
    print("  动捕世界原点未必在地面，不校正会悬空或陷地")

    print()
    print("=" * 72)
    print("5. 回放并渲染")
    print("=" * 72)
    OUT.mkdir(exist_ok=True)
    model.vis.global_.offwidth, model.vis.global_.offheight = 640, 480
    data = mujoco.MjData(model)

    # 相机跟着人走，否则走出画面
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 3.5
    cam.elevation = -15
    cam.azimuth = 135

    # GIF 按 15fps 抽帧、480x360 输出——原分辨率 30fps 会到 50MB+
    gif_fps = 15
    stride = max(1, round(fps / gif_fps))
    frames = []
    with mujoco.Renderer(model, 360, 480) as r:
        for i in range(0, T, stride):
            data.qpos[:] = qpos[i]
            mujoco.mj_forward(model, data)      # 运动学回放：forward 不是 step
            cam.lookat[:] = data.qpos[:3]
            r.update_scene(data, camera=cam)
            frames.append(r.render())

    gif = OUT / f"10_{path.stem[:24]}.gif"
    iio.imwrite(gif, np.stack(frames), duration=1000 / gif_fps, loop=0)
    print(f"  已保存 {gif.name}  {len(frames)} 帧 @{gif_fps}fps  "
          f"{gif.stat().st_size/1024/1024:.1f} MB")

    print()
    print("  注意：这是**纯运动学回放**，没有物理。")
    print("  机器人不会倒，因为根本没在算重力和接触——")
    print("  这正是第 9 课说的：想让它物理上真站住，必须上 RL。")

    if "--view" in args:
        print()
        print("=" * 72)
        print("6. 交互窗口（循环播放）")
        print("=" * 72)
        print("  关闭窗口结束")
        with mujoco.viewer.launch_passive(model, data) as v:
            i = 0
            while v.is_running():
                data.qpos[:] = qpos[i % T]
                mujoco.mj_forward(model, data)
                v.sync()
                i += 1
        print("  窗口已关闭")
    else:
        print()
        print("  想看实时循环: python 10_real_mocap.py --view")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
