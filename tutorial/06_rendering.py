"""第 6 课：渲染 —— 出图、出视频、多相机

MuJoCo 有两套渲染出口，用途完全不同：

    mujoco.viewer     交互式窗口，人看的。会阻塞，不能批量跑。
    mujoco.Renderer   离屏渲染，代码用的。返回 numpy 数组。
                      采视觉数据集、录视频、RL 的图像观测都靠它。

**离屏分辨率受模型限制**：默认只有 640x480，
想要更大必须改 model.vis.global_.offwidth/offheight（编译后可改）。

运行: python 06_rendering.py
输出: frames/ 目录下的 PNG，以及 pendulum.gif
"""

import pathlib

import imageio.v3 as iio
import mujoco
import numpy as np

OUT = pathlib.Path(__file__).parent / "out"

XML = """
<mujoco model="render_demo">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4"
             rgb2="0.3 0.4 0.5" width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="8 8" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="1 1 3" dir="-0.3 -0.3 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="grid_mat"/>

    <body name="pendulum" pos="0 0 1.5">
      <joint name="j1" type="hinge" axis="0 1 0" damping="0.02"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.03"
            rgba="0.8 0.3 0.2 1" mass="0.3"/>
      <body pos="0 0 -0.5">
        <joint name="j2" type="hinge" axis="0 1 0" damping="0.02"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.03"
              rgba="0.2 0.5 0.8 1" mass="0.3"/>
      </body>
    </body>

    <camera name="side"  pos="2.5 0 1.2"  xyaxes="0 1 0 -0.4 0 1"/>
    <camera name="front" pos="0 -2.5 1.2" xyaxes="1 0 0 0 0.4 1"/>
    <camera name="top"   pos="0 0 3.5"    xyaxes="1 0 0 0 1 0"/>
  </worldbody>
</mujoco>
"""


def demo_single_frame(model, data):
    print("=" * 66)
    print("1. 最基本：渲染一帧")
    print("=" * 66)
    # Renderer 要用 with，否则 GL 资源不释放
    with mujoco.Renderer(model, height=480, width=640) as r:
        r.update_scene(data)          # 把当前 data 的状态送进渲染器
        pixels = r.render()           # 返回 (H, W, 3) uint8
    print(f"  返回类型 {type(pixels).__name__}  shape={pixels.shape}  dtype={pixels.dtype}")
    iio.imwrite(OUT / "01_default.png", pixels)
    print(f"  已保存 out/01_default.png")
    print()
    print("  注意 update_scene 必须在 render 之前调 —— 它才是把状态同步过去的一步")


def demo_cameras(model, data):
    print()
    print("=" * 66)
    print("2. 多相机 —— 同一时刻不同视角")
    print("=" * 66)
    with mujoco.Renderer(model, 480, 640) as r:
        for cam in ["side", "front", "top"]:
            r.update_scene(data, camera=cam)
            iio.imwrite(OUT / f"02_cam_{cam}.png", r.render())
            print(f"  out/02_cam_{cam}.png")
    print()
    print("  camera 参数可以是名字、id，或 -1（默认自由视角）")
    print("  RL 的图像观测就是固定一个机身相机，每步渲一帧")


def demo_resolution(model, data):
    print()
    print("=" * 66)
    print("3. 分辨率上限 —— 最常见的报错")
    print("=" * 66)
    print(f"  模型声明的离屏缓冲: {model.vis.global_.offwidth} x {model.vis.global_.offheight}")
    try:
        with mujoco.Renderer(model, 2000, 2000) as r:
            r.update_scene(data)
            r.render()
        print("  请求 2000x2000 成功")
    except Exception as e:
        print(f"  请求 2000x2000 失败: {type(e).__name__}")
        print(f"    {str(e)[:100]}")
        print()
        print("  解法：编译后直接改（不用改 XML）")
        print("    model.vis.global_.offwidth  = 2000")
        print("    model.vis.global_.offheight = 2000")
        model.vis.global_.offwidth = 2000
        model.vis.global_.offheight = 2000
        with mujoco.Renderer(model, 2000, 2000) as r:
            r.update_scene(data)
            px = r.render()
        print(f"  改完再试: shape={px.shape}  成功")
        model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 720


def demo_options(model, data):
    print()
    print("=" * 66)
    print("4. 可视化选项 —— 看见平时看不见的东西")
    print("=" * 66)
    opts = {
        "contact": (mujoco.mjtVisFlag.mjVIS_CONTACTPOINT, "接触点"),
        "com": (mujoco.mjtVisFlag.mjVIS_COM, "质心"),
        "joint": (mujoco.mjtVisFlag.mjVIS_JOINT, "关节轴"),
        "transparent": (mujoco.mjtVisFlag.mjVIS_TRANSPARENT, "半透明"),
    }
    with mujoco.Renderer(model, 480, 640) as r:
        for key, (flag, note) in opts.items():
            opt = mujoco.MjvOption()
            opt.flags[flag] = True
            r.update_scene(data, camera="side", scene_option=opt)
            iio.imwrite(OUT / f"04_vis_{key}.png", r.render())
            print(f"  out/04_vis_{key}.png   显示{note}")
    print()
    print("  调试物理问题时，打开 contact 看接触点在哪，往往一眼看出问题")


def demo_depth_seg(model, data):
    print()
    print("=" * 66)
    print("5. 深度图与分割图 —— 视觉任务的标注来源")
    print("=" * 66)
    with mujoco.Renderer(model, 480, 640) as r:
        r.enable_depth_rendering()
        r.update_scene(data, camera="side")
        depth = r.render()
        r.disable_depth_rendering()
        finite = depth[np.isfinite(depth)]
        print(f"  深度图 shape={depth.shape} dtype={depth.dtype}")
        print(f"    范围 {finite.min():.3f} ~ {finite.max():.3f} 米")
        # 归一化成可看的灰度图
        vis = np.clip((depth - finite.min()) / (np.ptp(finite) + 1e-9), 0, 1)
        iio.imwrite(OUT / "05_depth.png", (vis * 255).astype(np.uint8))
        print(f"  out/05_depth.png")

        r.enable_segmentation_rendering()
        r.update_scene(data, camera="side")
        seg = r.render()
        r.disable_segmentation_rendering()
        print(f"  分割图 shape={seg.shape}  通道0=geom id, 通道1=geom type")
        ids = np.unique(seg[:, :, 0])
        print(f"    画面里的 geom id: {ids}")
        # 上色
        rng = np.random.default_rng(0)
        lut = rng.integers(0, 255, (int(ids.max()) + 2, 3), dtype=np.uint8)
        colored = lut[np.clip(seg[:, :, 0] + 1, 0, len(lut) - 1)]
        iio.imwrite(OUT / "05_segmentation.png", colored)
        print(f"  out/05_segmentation.png")
    print()
    print("  深度+分割 = 免费的完美标注，这是 sim2real 视觉训练的核心优势")


def demo_video(model, data):
    print()
    print("=" * 66)
    print("6. 录视频 —— 关键是渲染帧率 ≠ 仿真帧率")
    print("=" * 66)
    FPS = 30
    DURATION = 3.0
    # 仿真步长 0.002s = 500Hz，视频 30fps，所以每 500/30 ≈ 17 步渲一帧
    steps_per_frame = max(1, round(1.0 / (FPS * model.opt.timestep)))
    print(f"  仿真频率 {1/model.opt.timestep:.0f} Hz，视频 {FPS} fps")
    print(f"  -> 每 {steps_per_frame} 个仿真步渲 1 帧")
    print("  （每步都渲会慢 17 倍，而且视频会变成慢动作）")

    mujoco.mj_resetData(model, data)
    data.qpos[:] = [2.0, -1.0]
    frames = []
    with mujoco.Renderer(model, 480, 640) as r:
        for step in range(int(DURATION / model.opt.timestep)):
            mujoco.mj_step(model, data)
            if step % steps_per_frame == 0:
                r.update_scene(data, camera="side")
                frames.append(r.render())
    path = OUT / "06_pendulum.gif"
    iio.imwrite(path, np.stack(frames), duration=1000 / FPS, loop=0)
    print(f"  已保存 {path.name}  ({len(frames)} 帧, {path.stat().st_size/1024:.0f} KB)")
    print()
    print("  想要 mp4: uv pip install imageio-ffmpeg，然后 iio.imwrite('x.mp4', frames, fps=30)")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    data.qpos[:] = [1.2, -0.6]
    mujoco.mj_forward(model, data)

    demo_single_frame(model, data)
    demo_cameras(model, data)
    demo_resolution(model, data)
    demo_options(model, data)
    demo_depth_seg(model, data)
    demo_video(model, data)

    print()
    print("=" * 66)
    print(f"全部输出在 {OUT}")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name:<28}{f.stat().st_size/1024:>8.0f} KB")
    print()
    print("小结：")
    print("  Renderer 用 with；update_scene 必须在 render 前")
    print("  分辨率受 model.vis.global_.offwidth/offheight 限制，可编译后改")
    print("  录视频要按 fps 抽帧，不是每步都渲")
    print("  深度图/分割图是免费的完美标注")


if __name__ == "__main__":
    main()
