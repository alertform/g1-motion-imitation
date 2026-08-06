#!/usr/bin/env python3
"""动捕数据浏览与预览工具。

用法:
    python motions.py                      # 列出本地 + 远端目录
    python motions.py --search walk        # 搜索远端匹配的动作
    python motions.py --get walk1          # 下载匹配的动作（可多个）
    python motions.py --sheet              # 给本地全部动作出预览图（每个一张）
    python motions.py --sheet dance        # 只给匹配的出
    python motions.py --grid               # 所有本地动作合成一张总览图
    python motions.py --gif walk1          # 出 GIF
    python motions.py --play walk1         # 开交互窗口播放
"""

import argparse
import json
import os
import pathlib
import pickle
import sys
import urllib.request
import warnings

warnings.filterwarnings("ignore")

import imageio.v3 as iio
import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
MOTIONS = ROOT / "motions"
PREVIEW = MOTIONS / "preview"
MODEL_PATH = ROOT / "mujoco_menagerie" / "unitree_g1" / "scene.xml"

REPO = "openhe/g1-retargeted-motions"
API = f"https://huggingface.co/api/datasets/{REPO}"
RAW = f"https://huggingface.co/datasets/{REPO}/resolve/main"

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


# ---------------------------------------------------------------- 数据
def remote_catalog() -> list[str]:
    """远端全部 .pkl 的相对路径。"""
    with urllib.request.urlopen(API, timeout=60) as r:
        meta = json.load(r)
    return sorted(s["rfilename"] for s in meta.get("siblings", [])
                  if s["rfilename"].endswith(".pkl"))


def local_motions() -> list[pathlib.Path]:
    """本地可播放的动作：pkl（预重定向数据）+ npz（GMR 自己跑的）。"""
    return sorted(list(MOTIONS.glob("*.pkl")) + list(MOTIONS.glob("*.npz")))


def download(rel: str) -> pathlib.Path:
    MOTIONS.mkdir(parents=True, exist_ok=True)
    out = MOTIONS / pathlib.PurePosixPath(rel).name
    if out.exists():
        return out
    urllib.request.urlretrieve(f"{RAW}/{rel}", out)
    return out


def load(path: pathlib.Path) -> dict:
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


def motion_fps(path: pathlib.Path) -> int:
    if path.suffix == ".npz":
        return int(np.asarray(np.load(path, allow_pickle=True)["fps"]))
    return int(np.asarray(load(path)["fps"]))


def qpos_from_file(path: pathlib.Path, model: mujoco.MjModel) -> np.ndarray:
    """按文件类型取 qpos。

    .npz  —— GMR 直接产出的机器人 qpos，已在求解阶段完成接触约束和对地，
             不能再做 ground_align，否则会把已经对好的接触又推歪。
    .pkl  —— 预重定向的 23-DOF 数据，需要映射 + 事后对地。
    """
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=True)
        q = np.asarray(z["qpos"], dtype=np.float64)
        if q.shape[1] != model.nq:
            raise SystemExit(f"{path.name}: qpos 宽度 {q.shape[1]} != 模型 nq {model.nq}")
        return q
    return to_qpos(load(path), model)


def to_qpos(motion: dict, model: mujoco.MjModel) -> np.ndarray:
    trans = np.asarray(motion["root_trans_offset"], float)
    rot = np.asarray(motion["root_rot"], float)
    dof = np.asarray(motion["dof"], float)
    q = np.zeros((len(trans), model.nq))
    q[:, 0:3] = trans
    q[:, 3:7] = np.concatenate([rot[:, 3:4], rot[:, 0:3]], axis=-1)  # xyzw -> wxyz
    for s, t in DOF_MAP.items():
        q[:, 7 + t] = dof[:, s]
    return ground_align(q, model)


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


# ---------------------------------------------------------------- 渲染
def make_camera(distance=3.2, elevation=-12, azimuth=135):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.elevation, cam.azimuth = distance, elevation, azimuth
    return cam


def contact_sheet(name: str, qpos: np.ndarray, model: mujoco.MjModel,
                  n: int = 6, size: int = 260) -> np.ndarray:
    """把一段动作抽 n 帧横向拼成一张预览条。"""
    d = mujoco.MjData(model)
    cam = make_camera()
    idx = np.linspace(0, len(qpos) - 1, n).astype(int)
    tiles = []
    with mujoco.Renderer(model, size, size) as r:
        for i in idx:
            d.qpos[:] = qpos[i]
            mujoco.mj_forward(model, d)
            cam.lookat[:] = d.qpos[:3]
            r.update_scene(d, camera=cam)
            tiles.append(r.render())
    return np.hstack(tiles)


def label_strip(width: int, text: str, height: int = 22) -> np.ndarray:
    """纯色标签条。不引入字体依赖，用色块 + 文件名写在控制台。"""
    bar = np.full((height, width, 3), 32, dtype=np.uint8)
    bar[:2, :, :] = 90
    return bar


# ---------------------------------------------------------------- 命令
def cmd_list():
    local = local_motions()
    print(f"本地 {MOTIONS}  ({len(local)} 个)")
    for p in local:
        fps = motion_fps(p)
        if p.suffix == ".npz":
            T = len(np.load(p, allow_pickle=True)["qpos"])
            tag = "GMR"
        else:
            T = len(load(p)["root_trans_offset"])
            tag = "预重定向"
        print(f"  {p.stem:<34}{tag:<10}{T:>6} 帧 @{fps}fps = {T/fps:5.1f}s")
    try:
        remote = remote_catalog()
    except Exception as e:
        print(f"\n远端目录取不到: {e}")
        return
    have = {p.name for p in local}
    print(f"\n远端共 {len(remote)} 个，未下载 {sum(1 for r in remote if pathlib.PurePosixPath(r).name not in have)} 个")
    print("用 --search <关键词> 搜索，--get <关键词> 下载")


def cmd_search(pattern: str):
    remote = remote_catalog()
    hits = [r for r in remote if pattern.lower() in r.lower()]
    have = {p.name for p in local_motions()}
    print(f"匹配 '{pattern}' 的 {len(hits)} 个:")
    for r in hits:
        n = pathlib.PurePosixPath(r).name
        mark = "已下载" if n in have else "      "
        print(f"  {mark}  {n}")


def cmd_get(patterns: list[str]):
    remote = remote_catalog()
    todo = []
    for pat in patterns:
        todo += [r for r in remote if pat.lower() in r.lower()]
    todo = sorted(set(todo))
    if not todo:
        print("没有匹配项")
        return
    print(f"将下载 {len(todo)} 个:")
    for r in todo:
        p = download(r)
        print(f"  {p.name}  {p.stat().st_size/1024:.0f} KB")


def cmd_sheet(pattern: str | None):
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.vis.global_.offwidth = model.vis.global_.offheight = 512
    PREVIEW.mkdir(parents=True, exist_ok=True)
    targets = [p for p in local_motions()
               if pattern is None or pattern.lower() in p.stem.lower()]
    if not targets:
        print("本地没有匹配的动作，先用 --get 下载")
        return
    for p in targets:
        q = qpos_from_file(p, model)
        img = contact_sheet(p.stem, q, model)
        out = PREVIEW / f"{p.stem}.png"
        iio.imwrite(out, img)
        print(f"  {out.relative_to(ROOT)}   {img.shape[1]}x{img.shape[0]}")


def cmd_grid():
    """所有本地动作纵向叠成一张总览。"""
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.vis.global_.offwidth = model.vis.global_.offheight = 512
    PREVIEW.mkdir(parents=True, exist_ok=True)
    rows, names = [], []
    for p in local_motions():
        q = qpos_from_file(p, model)
        rows.append(contact_sheet(p.stem, q, model, n=6, size=200))
        names.append(p.stem)
    if not rows:
        print("本地没有动作")
        return
    w = min(r.shape[1] for r in rows)
    stacked = []
    for r, n in zip(rows, names):
        stacked.append(r[:, :w])
        stacked.append(label_strip(w, n))
    img = np.vstack(stacked)
    out = PREVIEW / "_overview.png"
    iio.imwrite(out, img)
    print(f"总览图: {out.relative_to(ROOT)}   {img.shape[1]}x{img.shape[0]}")
    print("从上到下依次是:")
    for i, n in enumerate(names, 1):
        print(f"  {i}. {n}")


def pick(pattern: str) -> pathlib.Path:
    hits = [p for p in local_motions() if pattern.lower() in p.stem.lower()]
    if not hits:
        raise SystemExit(f"本地没有匹配 '{pattern}' 的动作，先 --get {pattern}")
    return hits[0]


def cmd_gif(pattern: str, fps_out: int = 15):
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.vis.global_.offwidth, model.vis.global_.offheight = 480, 360
    p = pick(pattern)
    q = qpos_from_file(p, model)
    fps = motion_fps(p)
    stride = max(1, round(fps / fps_out))
    d = mujoco.MjData(model)
    cam = make_camera()
    frames = []
    with mujoco.Renderer(model, 360, 480) as r:
        for i in range(0, len(q), stride):
            d.qpos[:] = q[i]
            mujoco.mj_forward(model, d)
            cam.lookat[:] = d.qpos[:3]
            r.update_scene(d, camera=cam)
            frames.append(r.render())
    PREVIEW.mkdir(parents=True, exist_ok=True)
    out = PREVIEW / f"{p.stem}.gif"
    iio.imwrite(out, np.stack(frames), duration=1000 / fps_out, loop=0)
    print(f"  {out.relative_to(ROOT)}  {len(frames)} 帧  {out.stat().st_size/1024/1024:.1f} MB")


# 长按滚动的手感参数
SCRUB_NEW_PRESS_GAP = 0.25  # 距上次方向键超过这么久，就算一次全新的按下(秒)
SCRUB_STEP_MIN = 1.0        # 刚进入长按时，每个重复事件走几帧
SCRUB_STEP_MAX = 8.0        # 持续按住后的最大步长
SCRUB_RAMP = 1.5            # 从最小步长加速到最大步长所需的按住时长(秒)


def cmd_play(pattern: str):
    """交互播放。

    键位:
        空格    暂停 / 继续
        ← →     轻点=单帧；按住=连续滚动（越按越快）
        R       回到开头
        C       切换「跟随躯干」/「自由相机」
        Esc     viewer 自带的自由相机

    三个必须自己实现的点：

    1. 暂停：launch_passive 下 viewer 的 Pause 管的是它内部的物理步进，
       而我们自己在写 qpos，所以得自己接管，否则按空格没反应。

    2. 帧率：不做时间控制的话渲多快就播多快（60Hz 显示 = 2 倍速）。

    3. 长按：MuJoCo 的 key_callback 只给 keycode，**不区分按下/松开/重复**，
       也拿不到 Shift 之类的修饰键。所以只能把「事件还在持续到达」当作
       「键还按着」。

       关键设计：滚动**由事件驱动**，不是靠墙钟连续推进。
       每收到一个重复事件就走 step 帧，step 随按住时长从 1 涨到 8。
       这样松手即停、零过冲——先前按墙钟滚动的写法，因为要等
       0.25 秒才能判定松手，那段时间仍在高速滚动，实测过冲 50+ 帧。

       代价：从按下到开始滚动有操作系统的按键重复延迟（约 0.5 秒），
       这在拿不到 key-up 事件的前提下无法消除。
    """
    import time
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    p = pick(pattern)
    q = qpos_from_file(p, model)
    fps = motion_fps(p)
    dt = 1.0 / fps
    n = len(q)

    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    state = {
        "paused": False,
        "frame": 0.0,
        "track": pelvis >= 0,
        "cam_dirty": True,
        "scrub_dir": 0,
        "scrub_since": 0.0,
        "scrub_last": 0.0,
        "follow": pelvis >= 0,
    }

    def scrub_step(now: float) -> float:
        """按住时长 -> 每个重复事件前进多少帧。"""
        k = min(1.0, (now - state["scrub_since"]) / SCRUB_RAMP)
        return SCRUB_STEP_MIN + k * (SCRUB_STEP_MAX - SCRUB_STEP_MIN)

    def on_key(keycode: int) -> None:
        now = time.perf_counter()
        if keycode == 32:                                   # 空格
            state["paused"] = not state["paused"]
            state["scrub_dir"] = 0
            print(f"  {'暂停' if state['paused'] else '继续'} "
                  f"@ 第 {int(state['frame'])} 帧 / {n}")
        elif keycode in (262, 263):                         # → / ←
            direction = 1 if keycode == 262 else -1
            fresh = (direction != state["scrub_dir"]
                     or now - state["scrub_last"] > SCRUB_NEW_PRESS_GAP)
            if fresh:
                step = 1.0                 # 轻点就是干脆的一帧
                state["scrub_since"] = now
            else:
                step = scrub_step(now)     # 重复事件 = 键按着，按住越久走越快
            state["frame"] = (state["frame"] + direction * step) % n
            state["scrub_dir"] = direction
            state["scrub_last"] = now
            state["paused"] = True
        elif keycode in (ord("R"), ord("r")):
            state["frame"] = 0.0
            state["scrub_dir"] = 0
            print("  回到开头")
        elif keycode in (ord("C"), ord("c")):
            if pelvis < 0:
                print("  该模型没有 pelvis，无法跟随")
                return
            state["track"] = not state["track"]
            state["cam_dirty"] = True
            print(f"  相机: {'跟随躯干' if state['track'] else '自由'}")

    print(f"  播放 {p.stem}   {n} 帧 @{fps}fps = {n/fps:.1f}s（循环）")
    print("  空格=暂停/继续   ←/→=轻点单帧·按住连续滚动   R=开头   C=相机跟随")

    d = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, d, key_callback=on_key) as v:
        last = time.perf_counter()
        while v.is_running():
            now = time.perf_counter()
            elapsed = now - last
            last = now

            if not state["paused"]:
                # 按数据自己的帧率播放，而不是渲染多快就播多快
                state["frame"] = (state["frame"] + elapsed / dt) % n

            d.qpos[:] = q[int(state["frame"])]
            mujoco.mj_forward(model, d)

            # 只在切换时写相机（边沿触发），这样你用 Esc / [ ] 自己调的
            # 视角不会被每帧强行改回来
            if state["cam_dirty"]:
                if state["track"]:
                    # 用自由相机手动跟随，而不是 mjCAMERA_TRACKING。
                    # 跟踪相机的视点锁在 body 原点上，趴地动作里躯干只有
                    # 0.1m 高，配上俯视角相机就会穿到地板下面（画面里地板消失）。
                    # 这里把视点抬到躯干上方固定高度，保证始终在地面之上。
                    v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                    v.cam.distance = 3.5
                    v.cam.elevation = -12
                    v.cam.azimuth = 135
                    state["follow"] = True
                else:
                    v.cam.lookat[:] = d.qpos[:3]
                    v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                    state["follow"] = False
                state["cam_dirty"] = False

            if state.get("follow"):
                # 视点跟着躯干走，但高度锁在 0.5m —— 趴地时不至于把
                # 相机压到地板以下
                v.cam.lookat[0] = d.qpos[0]
                v.cam.lookat[1] = d.qpos[1]
                v.cam.lookat[2] = max(0.5, d.qpos[2])

            v.sync()
            time.sleep(max(0.0, dt * 0.5))


def main():
    ap = argparse.ArgumentParser(description="G1 动捕数据浏览与预览")
    ap.add_argument("--search", metavar="KW", help="搜索远端动作")
    ap.add_argument("--get", nargs="+", metavar="KW", help="下载匹配的动作")
    ap.add_argument("--sheet", nargs="?", const="", metavar="KW", help="出预览条（每个动作一张）")
    ap.add_argument("--grid", action="store_true", help="所有本地动作合成总览图")
    ap.add_argument("--gif", metavar="KW", help="出 GIF")
    ap.add_argument("--play", metavar="KW", help="开窗口播放")
    a = ap.parse_args()

    if a.search:
        cmd_search(a.search)
    elif a.get:
        cmd_get(a.get)
    elif a.sheet is not None:
        cmd_sheet(a.sheet or None)
    elif a.grid:
        cmd_grid()
    elif a.gif:
        cmd_gif(a.gif)
    elif a.play:
        cmd_play(a.play)
    else:
        cmd_list()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
