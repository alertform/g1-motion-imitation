#!/usr/bin/env python3
"""多个 G1 同屏，每个跑不同的动作段。

你在别处看到的「一个场景里几千个机器人同时训练」是 Isaac Lab 的效果：
它的仿真与渲染共用同一套 GPU 管线，训练时天然能出画面。
MJX 不同——它是**纯计算**的，4096 个环境只是显存里的张量，
整个训练过程不生成任何图像（这也是它快的原因之一）。

但状态随时可以读出来渲染。这个脚本用 MjSpec 把 N 个 G1 合并进一个
场景横向排开，各自跑不同的动作段，效果和那种展示一样——只是数量
按可读性取 4~8 个，不是几千个。

物理用 CPU MuJoCo（N 个实例而已），不占显存、不影响正在跑的训练。

用法:
    python rl_play_multi.py --ckpt runs/multi/policy.pkl --limit 6
    python rl_play_multi.py --clips walk1_subject1,sprint1_subject2 --speed 0.5
"""
import argparse
import os
import pathlib
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import mujoco


def build_multi(n, spacing=1.8):
    """用 MjSpec 把 n 个 G1 合并进一个场景。

    MjSpec.attach 会自动给 body/joint/geom 加前缀避免重名，
    比手工拼 XML 可靠得多。返回 (model, 每个实例的 qpos 起始下标)。
    """
    import rl_env
    scene = mujoco.MjSpec.from_file(str(rl_env.XML))       # 含地面与光照
    robot_xml = str(rl_env.XML.parent/"g1_mjx.xml")

    # scene_mjx 自身 include 了一个 g1，先记下它占的自由度
    base = mujoco.MjSpec.from_file(robot_xml)
    nq_one = base.compile().nq

    for i in range(1, n):                                   # 第 0 个已在场景里
        child = mujoco.MjSpec.from_file(robot_xml)
        frame = scene.worldbody.add_frame()
        frame.pos = [0.0, i * spacing, 0.0]                 # 沿 y 排开
        frame.attach_body(child.worldbody.first_body(),
                          f"r{i}_", "")                     # 前缀避免重名

    m = scene.compile()
    # 不调 configure_model：那是给单机器人的（按 29 个执行器读基准增益），
    # 合并后有 n×29 个会越界。这个场景只用 mj_forward 更新可视化位姿、
    # 不做物理积分，增益/求解器设置在这里无意义——真实物理各自在
    # NumpyRollout 里算，那边才需要与训练一致。
    offsets = [i * nq_one for i in range(n)]
    return m, nq_one, offsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/multi/policy_latest.pkl")
    ap.add_argument("--clips", default="")
    ap.add_argument("--grade", default="可用")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args()

    global mujoco                    # 下面 build_multi 也用它，避免被当局部名
    import jax
    import mujoco.viewer
    import rl_env, rl_play
    from rl_eval_multi import pick_clips

    names = pick_clips(a.grade, a.clips, a.limit)
    n = len(names)
    print(f"同屏 {n} 个 G1：")
    for i, nm in enumerate(names):
        print(f"  #{i+1} {nm}")

    ref, refv, refb, refc, refl = rl_env.load_reference(names)
    lens = np.asarray(refl)

    ck = pathlib.Path(a.ckpt)
    if not ck.is_absolute():
        ck = pathlib.Path.home()/"tools"/"rl"/ck
    if not ck.exists():
        raise SystemExit(f"存档不存在: {ck}")
    pol = rl_play.build_policy(ck, rl_env.OBS_SIZE, rl_env.NU)
    key = jax.random.PRNGKey(0)

    # 每个实例一份独立的单机器人物理（算各自的状态），
    # 再把结果拷进合并场景里渲染——比在合并模型上直接跑简单可靠，
    # 因为观测/动作的下标都按单机器人定义。
    rolls = [rl_play.NumpyRollout(np.asarray(ref[i]), np.asarray(refv[i]),
                                 clip=i, n_clip=n)
             for i in range(n)]
    steps = [0]*n
    lasts = [np.zeros(rl_env.NU) for _ in range(n)]
    for i in range(n):
        rolls[i].reset(0, np.asarray(refv[i]))

    print("\n构建合并场景…")
    m, nq_one, offs = build_multi(n)
    d = mujoco.MjData(m)
    print(f"  {m.nbody} 个 body，nq={m.nq}（单机器人 {nq_one}）")

    ctrl_dt = rolls[0].n_frames * rolls[0].m.opt.timestep
    print(f"回放速度 {a.speed}×   一步 = {ctrl_dt*1000:.0f}ms 仿真时间\n")

    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.distance = 3.0 + n*0.9
        v.cam.elevation = -14.0
        v.cam.lookat[:] = [0.0, (n-1)*0.9, 0.7]
        wall = time.perf_counter()

        while v.is_running():
            for i in range(n):
                r = rolls[i]
                act = np.asarray(pol(r.obs(steps[i], lasts[i]), key)[0])
                r.apply(act, steps[i])
                lasts[i] = act
                steps[i] += 1

                # 状态拷进合并场景：位置加上该实例的横向偏移
                q = r.d.qpos.copy()
                q[1] += i * 1.8
                d.qpos[offs[i]:offs[i]+nq_one] = q

                if r.fell() or steps[i] >= int(lens[i]) - rl_env.LOOKAHEAD - 1:
                    print(f"  #{i+1} {names[i]}: {steps[i]} 步 = "
                          f"{steps[i]*ctrl_dt:.1f}s"
                          f"（{'摔倒' if r.fell() else '播完'}），重来",
                          flush=True)
                    r.reset(0, np.asarray(refv[i]))
                    steps[i] = 0
                    lasts[i] = np.zeros(rl_env.NU)

            mujoco.mj_forward(m, d)      # 只更新可视化用的位姿，不做积分
            v.sync()

            wall += ctrl_dt/max(a.speed, 1e-6)
            lag = wall - time.perf_counter()
            if lag > 0:
                time.sleep(lag)
            else:
                wall = time.perf_counter()


if __name__ == "__main__":
    main()
