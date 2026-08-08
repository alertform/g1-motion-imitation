#!/usr/bin/env python3
"""无头评估策略的跟踪质量。

奖励是几项 exp(-k·err) 的加权和，数值本身不可读——奖励 80 分说明不了
膝关节差几度。这里把它拆回物理量：

    关节跟踪误差   度      策略复现参考姿态的精度
    根位置漂移     厘米    有没有走偏
    根朝向误差     度
    脚部滑移       厘米/步 支撑脚在触地期间的水平位移，衡量脚打滑
    存活率         %       跑满整段的比例

用法:
    python rl_eval.py --ckpt runs/walk/policy_latest.pkl --episodes 8
"""
import argparse
import os
import pathlib

os.environ.setdefault("JAX_PLATFORMS", "cpu")   # 不和训练抢显存

import numpy as np
import mujoco


def run_episode(roll, policy, key, start, refv, max_steps):
    """跑一段，返回逐帧的跟踪量。"""
    import rl_env
    roll.reset(start, refv)
    last_act = np.zeros(rl_env.NU)

    fid = [mujoco.mj_name2id(roll.m, mujoco.mjtObj.mjOBJ_BODY, n)
           for n in ("left_ankle_roll_link", "right_ankle_roll_link")]

    jnt_err, pos_err, quat_err = [], [], []
    slip, prev_xy, prev_down = [], None, None

    for i in range(max_steps):
        step = start + i
        o = roll.obs(step, last_act)
        act, _ = policy(o, key)
        act = np.asarray(act)
        roll.apply(act, step)
        last_act = act

        ref = roll.ref[min(step + 1, roll.T - 1)]
        jnt_err.append(np.abs(roll.d.qpos[7:] - ref[7:]))
        pos_err.append(np.linalg.norm(roll.d.qpos[:3] - ref[:3]))
        dot = abs(float(np.dot(roll.d.qpos[3:7], ref[3:7])))
        quat_err.append(2.0 * np.arccos(min(1.0, dot)))

        # 脚滑：只统计「上一帧和这一帧都贴地」的脚的水平位移
        fz = roll.d.xpos[fid][:, 2]
        xy = roll.d.xpos[fid][:, :2].copy()
        down = fz < 0.10
        if prev_xy is not None:
            both = down & prev_down
            if both.any():
                slip.append(float(np.linalg.norm(xy[both]-prev_xy[both], axis=1).mean()))
        prev_xy, prev_down = xy, down

        if roll.fell():
            return dict(steps=i+1, alive=False, jnt=np.array(jnt_err),
                        pos=np.array(pos_err), quat=np.array(quat_err),
                        slip=np.array(slip))

    return dict(steps=max_steps, alive=True, jnt=np.array(jnt_err),
                pos=np.array(pos_err), quat=np.array(quat_err),
                slip=np.array(slip))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/walk/policy_latest.pkl")
    ap.add_argument("--clip", default="walk1_subject1")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--starts", default="",
                    help="逗号分隔的起始帧，指定后忽略 --episodes（用于定点复查）")
    a = ap.parse_args()

    import jax
    import rl_env, rl_play

    ref, refv, refb, refc = rl_env.load_reference([a.clip])
    ref, refv = np.asarray(ref[0]), np.asarray(refv[0])
    roll = rl_play.NumpyRollout(ref, refv)

    ckpt = pathlib.Path(a.ckpt)
    if not ckpt.is_absolute():
        ckpt = pathlib.Path.home()/"tools"/"rl"/ckpt
    if not ckpt.exists():
        raise SystemExit(f"存档不存在: {ckpt}")

    policy = rl_play.build_policy(ckpt, rl_env.OBS_SIZE, rl_env.NU)
    key = jax.random.PRNGKey(0)

    # 起点集与 max_steps 无关，两个评估脚本共用同一定义（见 rl_eval_mjx）
    from rl_eval_mjx import eval_starts
    if a.starts:
        starts = np.array([int(s) for s in a.starts.split(",")])
    else:
        starts = eval_starts(roll.T, a.episodes)

    print(f"存档 {ckpt.name}   动作 {a.clip}   {len(starts)} 段 × {a.max_steps} 步")
    print(f"起始帧: {starts.tolist()}")
    print()
    print(f"  {'起点':>7} {'步数':>6} {'跑满':>5} {'关节误差°':>10} "
          f"{'p95°':>7} {'根漂移cm':>9} {'朝向°':>7} {'脚滑cm/步':>10}")
    rows = []
    for s in starts:
        r = run_episode(roll, policy, key, int(s), refv, a.max_steps)
        jd = np.degrees(r["jnt"])
        row = dict(start=int(s), steps=r["steps"], alive=r["alive"],
                   jnt=jd.mean(), jnt95=np.percentile(jd, 95),
                   pos=r["pos"].mean()*100,
                   quat=np.degrees(r["quat"]).mean(),
                   slip=r["slip"].mean()*100 if len(r["slip"]) else float("nan"))
        rows.append(row)
        print(f"  {row['start']:>7} {row['steps']:>6} {'是' if row['alive'] else '否':>5} "
              f"{row['jnt']:>10.2f} {row['jnt95']:>7.2f} {row['pos']:>9.2f} "
              f"{row['quat']:>7.2f} {row['slip']:>10.3f}")

    print()
    alive = sum(r["alive"] for r in rows)
    print(f"  存活率 {alive}/{len(rows)} = {100*alive/len(rows):.0f}%")
    print(f"  平均回合长 {np.mean([r['steps'] for r in rows]):.1f} / {a.max_steps}")
    print(f"  关节跟踪 中位 {np.median([r['jnt'] for r in rows]):.2f}°  "
          f"p95 {np.median([r['jnt95'] for r in rows]):.2f}°")
    print(f"  根漂移   中位 {np.median([r['pos'] for r in rows]):.2f} cm")
    print(f"  脚滑     中位 {np.nanmedian([r['slip'] for r in rows]):.3f} cm/步")
    print()
    print("  参考：关节误差 <5° 算跟得紧，>15° 肉眼就能看出动作变形；")
    print("        脚滑 >0.5cm/步（=25cm/s）在真机上会明显打滑。")


if __name__ == "__main__":
    main()
